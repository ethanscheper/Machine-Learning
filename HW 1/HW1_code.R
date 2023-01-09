#### Load in necessary libraries ####
library(tidyverse)
library(caret)
library(leaps)
library(glmnet)
library(ggplot2)
library(earth)
library(mgcv)
library(InformationValue)

#### Read in the data ####
setwd("/Users/ethanscheper/Documents/AA 502/Machine Learning/HW 1")
train <- read.csv("insurance_t.csv")

#### Create a function for finding the mode ####
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

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

# Look at number of unique values for all of the variables with missing values
# Determine if variables are continuous or categorical
for (i in 1:nrow(missing_vars)) {
  var <- missing_vars$name[i]
  print(var)
  print(length(unique(train[[var]])))
}

# If a variable has >10 distinct values (continuous), use median for imputation
# If a variable has <=10 distinct values (categorical), use mode for imputation
# Except PHONE and POS - use mode for those as well
for (i in 1:nrow(missing_vars)) {
  var <- missing_vars$name[i]
  if (length(unique(train[[var]])) <= 10) {
    train[[var]] <- train[[var]] %>%
      replace_na(getmode(train[[var]]))
  }
  else {
    train[[var]] <- train[[var]] %>%
      replace_na(median(train[[var]], na.rm = T))
  }
}

# Check to make sure it worked
sum(is.na(train))

#### Build a model with the MARS (EARTH) algorithm ####
# Build the model
mars <- earth(INS ~ ., data = train, glm = list(family = binomial))
summary(mars)

# Report variable importance
evimp(mars)

# Report area under ROC curve and plot of ROC curve
mars_pred <- predict(mars, type = "response")
plotROC(train$INS, mars_pred)

#### Build a GAM model with splines on continuous variables ####
# Check which variables are continuous and which are categorical
continuous <- c()
categorical <- c()
for (col in colnames(train)) {
  if (length(unique(train[[col]])) <= 10) {
    categorical <- c(categorical, col)
  }
  else {
    continuous <- c(continuous, col)
  }
}

# Build the model
gam1 <- mgcv::gam(INS ~ s(ACCTAGE) + 
                   s(DDABAL) + 
                   s(DEP) + 
                   s(DEPAMT) + 
                   s(CHECKS) + 
                   s(NSFAMT) + 
                   s(PHONE) + 
                   s(TELLER) + 
                   s(SAVBAL) + 
                   s(ATMAMT) + 
                   s(POS) + 
                   s(POSAMT) + 
                   s(CDBAL) + 
                   s(IRABAL) + 
                   s(INVBAL) + 
                   s(MMBAL) + 
                   s(CCBAL) + 
                   s(INCOME) + 
                   s(LORES) + 
                   s(HMVAL) + 
                   s(AGE) + 
                   s(CRSCORE) + 
                   factor(BRANCH) + 
                   factor(DDA) + 
                   factor(DIRDEP) + 
                   factor(NSF) + 
                   factor(SAV) + 
                   factor(ATM) + 
                   factor(CD) + 
                   factor(IRA) + 
                   factor(INV) + 
                   factor(MM) + 
                   factor(MMCRED) + 
                   factor(CC) + 
                   factor(CCPURC) + 
                   factor(SDB) + 
                   factor(INAREA), 
                 family = 'binomial', method = 'REML', select = TRUE, data = train)
summary(gam1)
# Selected DDA, NSF, ATM, CD, IRA, INV, MM, CC | ACCTAGE, DDABAL, DEP, CHECKS, 
# TELLER, SAVBAL, ATMAMT, CDBAL

# Build a new model with the selected variables
gam2 <- mgcv::gam(INS ~ s(ACCTAGE) + 
                    s(DDABAL) + 
                    s(DEP) + 
                    s(CHECKS) + 
                    s(TELLER) + 
                    s(SAVBAL) + 
                    s(ATMAMT) + 
                    s(CDBAL) + 
                    factor(BRANCH) + 
                    factor(DDA) + 
                    factor(NSF) + 
                    factor(ATM) + 
                    factor(CD) + 
                    factor(IRA) + 
                    factor(INV) + 
                    factor(MM) + 
                    factor(CC),
                  family = 'binomial', method = 'REML', data = train)
summary(gam2)

# Report area under ROC curve and plot of ROC curve
gam_pred <- predict(gam2, type = "response")
plotROC(train$INS, gam_pred)












