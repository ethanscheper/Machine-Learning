#### Load in necessary libraries ####
library(tidyverse)
library(caret)
library(leaps)
library(glmnet)
library(ggplot2)
library(earth)
library(mgcv)
library(InformationValue)
library(randomForest)
library(xgboost)
library(Ckmeans.1d.dp)
library(pdp)
library(varhandle)
library(imputeMissings)

#### Read in the data ####
setwd("/Users/ethanscheper/Documents/AA 502/Machine Learning/HW 2")
train <- read.csv("insurance_t.csv")

#### Classify each of the variables by type and factor categoricals ####
binary <- c("DDA", "DIRDEP", "NSF", "SAV", "ATM", "CD", "IRA", "MM", "SDB", 
            "INAREA", "INV", "CC")
ordinal <- c("MMCRED", "CCPURC")
nominal <- c("BRANCH")
continuous <- c("ACCTAGE", "DDABAL", "DEP", "DEPAMT", "CHECKS", "NSFAMT", 
                "PHONE", "TELLER", "SAVBAL", "ATMAMT", "POS", "POSAMT", 
                "CDBAL", "IRABAL", "INVBAL", "MMBAL", "CCBAL", "INCOME", 
                "LORES", "HMVAL", "AGE", "CRSCORE")

train[binary] <- lapply(train[binary], as.factor)
train[ordinal] <- lapply(train[ordinal], as.factor)
train[nominal] <- lapply(train[nominal], as.factor)
train$INS <- as.factor(train$INS)

#### Explore which variables have missing values and how many they have ####
# Find variables with missing values
train_col <- data.frame(colnames(train))
train_col <- train_col %>%
  rename(name = `colnames.train.`)

for (i in 1:nrow(train_col)) {
  train_col$num_missing[i] <- sum(is.na(train[,i]))
}

missing_vars <- train_col %>%
  filter(num_missing > 0)

#### Impute missing values ####
train <- impute(train, method = "median/mode", flag = T)

# Check to make sure it worked
sum(is.na(train))

#### Check for and correct separation issues ####
# Subset data to only have categorical variables
train_cat <- train %>%
  dplyr::select(all_of(binary), all_of(nominal), all_of(ordinal))

# Check each variable for separation concerns
sep_list <- list()
for (col in colnames(train_cat)) {
  freqs <- table(train[[col]], train$INS)
  test <- sapply(freqs, function(x) x==0)
  if (TRUE %in% test) {
    sep_list <- c(sep_list, col)
    print(col)
    print(freqs)
  }
}

# Correct the separation problem for MMCRED by creating a 3+ category
train$MMCRED <- unfactor(train$MMCRED)
train <- train %>%
  mutate(MMCRED = ifelse(MMCRED >= 3, '3+', MMCRED))

# Check to make sure it worked
table(train$MMCRED, train$INS)

# Refcator MMCRED
train$MMCRED <- as.factor(train$MMCRED)

#### Build a random forest model ####
# Fit a model
set.seed(121299)
rf1 <- randomForest(INS ~ ., data = train, ntree = 500, importance = TRUE)

# Plot the change in error across different number of trees
plot(rf1, main = "Number of Trees Compared to MSE")
# 300 or so trees should be good for fitting future models

# See how well this initial model predicts
rf1_pred <- predict(rf1, type = "prob")
plotROC(train$INS, rf1_pred[,2])
# AUROC: 0.7942

# Tune a random forest mtry value
set.seed(121299)
tuneRF(x = train[,-37], y = train[,37], 
       plot = TRUE, ntreeTry = 300, stepFactor = 0.5)
# mtry = 7

# Fit another model with the new mtry value
set.seed(121299)
rf2 <- randomForest(INS ~ ., data = train, ntree = 300, mtry = 7, 
                    importance = TRUE)

# See how well this model predicts
rf2_pred <- predict(rf2, type = "prob")
plotROC(train$INS, rf2_pred[,2])
# AUROC: 0.7913

# Fit another model with variable selection
set.seed(121299)
train$random <- rnorm(nrow(train))
set.seed(121299)
rf3 <- randomForest(INS ~ ., data = train, ntree = 300, importance = TRUE)

# See which variables were more important than the random one
varImpPlot(rf3,
           sort = TRUE,
           n.var = 15,
           main = "Look for Variables Below Random Variable")
importance(rf3)
# In terms of mean decrease in Gini, only SAVBAL, BRANCH, and DDABAL
# In terms of mean decrease in accuracy, all of the variables did better

# Looks like the very first model was our best in terms of prediction
# We will use that model as our "final RF model"

#Look at variable importance
varImpPlot(rf1,
           sort = TRUE,
           n.var = 10,
           main = "Top 10 - Variable Importance")

# Get importance for all of the variables for report
importance(rf1)

# Remove the random variable from the data
train <- train %>%
  dplyr::select(-random)

#### Build an XGBoost model ####
# Prepare data for XGBoost function
train_x <- model.matrix(INS ~ ., data = train)[, -1]
train_y <- train$INS

# Fit a model
train_y_nonfactor <- unfactor(train_y) # Needs a non-factor for binary response
set.seed(121299)
xgb1 <- xgboost(data = train_x, label = train_y_nonfactor, subsample = 0.5, 
                nrounds = 100, objective = "binary:logistic")

# Use caret::train function so we can predict and see ROC curve
set.seed(121299)
xgb.caret1 <- caret::train(x = train_x, y = train_y,
                           method = "xgbTree",
                           trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                                    number = 10))

# See how well this model predicts
xgb.caret_pred1 <- predict(xgb.caret1, type = "prob")
plotROC(train$INS, xgb.caret_pred1[,2])
# AUROC = 0.8396

# Tuning XGBoost nrounds parameter for maximum AUROC
set.seed(121299)
xgbcv1 <- xgb.cv(data = train_x, label = train_y_nonfactor, metrics = "auc", 
                 subsample = 0.5, nrounds = 100, nfold = 10, 
                 objective = "binary:logistic")
which.max(xgbcv1[["evaluation_log"]]$test_auc_mean)
# 8 was the highest

# Grid search for the best tuning parameters
tune_grid <- expand.grid(
  nrounds = 50, 
  eta = c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4),
  max_depth = c(1:10),
  gamma = c(0),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.25, 0.5, 0.75, 1)
)

# Use caret::train function to fit an XGBoost with the optimal values from the tune grid
set.seed(121299)
xgb.caret2 <- caret::train(x = train_x, y = train_y,
                        method = "xgbTree",
                        tuneGrid = tune_grid,
                        trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                                 number = 10))

# Plot and check the best tune values
plot(xgb.caret2)
xgb.caret2$bestTune

# See how well this model predicts
xgb.caret_pred2 <- predict(xgb.caret2, type = "prob")
plotROC(train$INS, xgb.caret_pred2[,2])
# AUROC = 0.8592

# Fit a model that we can feed to variable importance function w best tune values
xgb.caret2$bestTune
set.seed(121299)
xgb2 <- xgboost(data = train_x, label = train_y_nonfactor, subsample = 0.5, 
                eta = 0.1, nrounds = 50, max_depth = 5, gamma = 0, 
                colsample_bytree = 1, min_child_weight = 1, 
                objective = "binary:logistic")

# Variable importance
xgb.importance(feature_names = colnames(train_x), model = xgb2)
xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb2))

# Create a random variable for variable selection
set.seed(121299)
train$random <- rnorm(nrow(train))
train_x <- model.matrix(INS ~ ., data = train)[, -1]
train_y <- train$INS
train_y_nonfactor <- unfactor(train_y)

# Fit a model with the optimal parameters and the random variable
# Fit a model that we can feed to variable importance function w best tune values
set.seed(121299)
xgb3 <- xgboost(data = train_x, label = train_y_nonfactor, subsample = 0.5, 
                eta = 0.1, nrounds = 50, max_depth = 5, gamma = 0, 
                colsample_bytree = 1, min_child_weight = 1, 
                objective = "binary:logistic")

# Check variable importance for the model
xgb.importance(feature_names = colnames(train_x), model = xgb3)
xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb3))
# SAVBAL, DDABAL, CDBAL, DDA, MMBAL, and ACCTAGE were the only ones better

# Refit a model with only these variables and see how it predicts
train_new <- train %>%
  select(INS, SAVBAL, DDABAL, CDBAL, DDA, MMBAL, ACCTAGE)
train_x_new <- model.matrix(INS ~ ., data = train_new)[, -1]

set.seed(121299)
xgb.caret3 <- caret::train(x = train_x_new, y = train_y,
                           method = "xgbTree",
                           tuneGrid = tune_grid,
                           trControl = trainControl(method = 'cv', # Using 10-fold cross-validation
                                                    number = 10))
xgb.caret3$bestTune

# See how well this model predicts
xgb.caret_pred3 <- predict(xgb.caret3, type = "prob")
plotROC(train$INS, xgb.caret_pred3[,2])
# AUROC = 0.8100

# Partial dependence plot using best model
partial(xgb3, pred.var = "SAVBAL", plot = TRUE, rug = TRUE, alpha = 0.1, 
        plot.engine = "lattice", train = train_x)
# Not impactful or worth including






