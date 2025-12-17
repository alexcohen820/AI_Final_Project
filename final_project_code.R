library(randomForest)
library(caret)
library(yardstick)
library(dplyr)
library(xgboost)
library(broom)
library(knitr)
library(kableExtra)


#############################
#DATA IS NOT PUBLIC AND CANNOT BE SUPPLIED FOR REPRODUCTION

df <- read_excel("gambling_demo_data.xlsx")

set.seed(123)
split <- initial_split(df, prop = 0.8)
train_data <- training(split)
test_data  <- testing(split)

#######################################################################
#BASELINE MODELS

# Baseline predictions = mean of training outcome
baseline_pred_train <- rep(mean(train_data$All_Players_Percent), nrow(train_data))
baseline_pred_test  <- rep(mean(train_data$All_Players_Percent), nrow(test_data))

baseline_metrics <- data.frame(
  metric = c("RMSE", "MAE", "R2"),
  train = c(
    rmse_vec(train_data$All_Players_Percent, baseline_pred_train),
    mae_vec(train_data$All_Players_Percent, baseline_pred_train),
    rsq_vec(train_data$All_Players_Percent, baseline_pred_train)
  ),
  test = c(
    rmse_vec(test_data$All_Players_Percent, baseline_pred_test),
    mae_vec(test_data$All_Players_Percent, baseline_pred_test),
    rsq_vec(test_data$All_Players_Percent, baseline_pred_test)
  )
)

baseline_metrics



# Fit linear model
lm_model <- lm(All_Players_Percent ~ ., data = train_data)

# Predictions
lm_pred_train <- predict(lm_model, newdata = train_data)
lm_pred_test  <- predict(lm_model, newdata = test_data)

# Metrics
lm_metrics <- data.frame(
  metric = c("RMSE", "MAE", "R2"),
  train = c(
    rmse_vec(train_data$All_Players_Percent, lm_pred_train),
    mae_vec(train_data$All_Players_Percent, lm_pred_train),
    rsq_vec(train_data$All_Players_Percent, lm_pred_train)
  ),
  test = c(
    rmse_vec(test_data$All_Players_Percent, lm_pred_test),
    mae_vec(test_data$All_Players_Percent, lm_pred_test),
    rsq_vec(test_data$All_Players_Percent, lm_pred_test)
  )
)

lm_metrics




########################################################################
#LASSO
#########################################################################

x_train <- model.matrix(All_Players_Percent ~ ., train_data)[, -1]
y_train <- train_data$All_Players_Percent
x_test <- model.matrix(All_Players_Percent ~ ., test_data)[, -1]
y_test <- test_data$All_Players_Percent

set.seed(123)

cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 5,      
  standardize = TRUE
)
cv_lasso$lambda.min

lasso_final <- glmnet(
  x_train,
  y_train,
  alpha = 1,
  lambda = cv_lasso$lambda.min,
  standardize = TRUE
)

# Predictions (convert matrix -> numeric vector)
lasso_pred_train <- as.numeric(predict(lasso_final, x_train))
lasso_pred_test  <- as.numeric(predict(lasso_final, x_test))

# Metrics
lasso_metrics <- data.frame(
  metric = c("RMSE", "MAE", "R2"),
  train = c(
    rmse_vec(y_train, lasso_pred_train),
    mae_vec(y_train, lasso_pred_train),
    rsq_vec(y_train, lasso_pred_train)
  ),
  test = c(
    rmse_vec(test_data$All_Players_Percent, lasso_pred_test),
    mae_vec(test_data$All_Players_Percent, lasso_pred_test),
    rsq_vec(test_data$All_Players_Percent, lasso_pred_test)
  )
)

lasso_metrics

######IMPORTANCE

coef_df <- as.data.frame(as.matrix(coef(lasso_final)))
coef_df$Feature <- rownames(coef_df)
colnames(coef_df)[1] <- "Coefficient"

coef_df <- coef_df %>% 
  filter(Feature != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

coef_df <- coef_df %>%
  mutate(Coefficient = round(Coefficient, 3))

coef_df %>%
  kbl(caption = "LASSO Coefficients Sorted by Absolute Effect Size") %>%
  kable_styling(full_width = FALSE)



######################################################
#####RANDOM FOREST
#######################################################

#################CROSS VALIDATION

# Hyperparameters to test
hyper_grid <- list(
  ntree = c(100, 200, 500, 1000, 1500),
  mtry = c(2, 5, 10),
  nodesize = c(1, 2, 3, 5, 10),
  maxnodes = list(NULL, 5, 10) 
)

# Initialize results table
results_rf_cv <- data.frame(
  hyperparam = character(),
  value = character(),
  cv_rmse = numeric(),
  cv_mae = numeric(),
  cv_r2 = numeric(),
  stringsAsFactors = FALSE
)

# Base parameters
base_params <- list(
  ntree = 500,
  mtry = 5,
  nodesize = 5,
  maxnodes = NULL
)

# Define 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Loop over each hyperparameter
for (param_name in names(hyper_grid)) {
  for (value in hyper_grid[[param_name]]) {
    
    # Set parameters for this run
    params <- base_params
    params[[param_name]] <- value
    
    rf_cv <- train(
      All_Players_Percent ~ .,
      data = train_data,
      method = "rf",
      trControl = train_control,
      ntree = params$ntree,
      mtry = params$mtry,
      nodesize = params$nodesize,
      maxnodes = params$maxnodes,
      importance = FALSE
    )
    
    # Extract cross-validated results
    cv_metrics <- rf_cv$results[1, ]
    
    # Save results
    results_rf_cv <- rbind(
      results_rf_cv,
      data.frame(
        hyperparam = param_name,
        value = ifelse(is.null(value), "NULL", as.character(value)),
        cv_rmse = cv_metrics$RMSE,
        cv_r2   = cv_metrics$Rsquared
      )
    )
  }
}


# View results

cv_results_rf <- results_rf_cv %>% arrange(hyperparam, cv_rmse)
cv_results_rf


#####
##FINAL HYPERPARAMETERS

rf_model <- randomForest(
  All_Players_Percent ~ .,
  data = train_data,
  ntree = 200,        
  mtry = 2,            
  nodesize = 10,        
  maxnodes = NULL      
)
# Predictions
train_pred_rf <- predict(rf_model, newdata = train_data)
test_pred_rf  <- predict(rf_model, newdata = test_data)

# RMSE
train_rmse_rf <- rmse_vec(truth = train_data$All_Players_Percent, estimate = train_pred_rf)
test_rmse_rf  <- rmse_vec(truth = test_data$All_Players_Percent, estimate = test_pred_rf)

# MAE
train_mae_rf <- mae_vec(truth = train_data$All_Players_Percent, estimate = train_pred_rf)
test_mae_rf  <- mae_vec(truth = test_data$All_Players_Percent, estimate = test_pred_rf)

# R^2
train_r2_rf <- rsq_vec(truth = train_data$All_Players_Percent, estimate = train_pred_rf)
test_r2_rf  <- rsq_vec(truth = test_data$All_Players_Percent, estimate = test_pred_rf)

rf_metrics <- data.frame(
  metric = c("RMSE", "MAE", "R2"),
  train  = c(train_rmse_rf, train_mae_rf, train_r2_rf),
  test   = c(test_rmse_rf,  test_mae_rf,  test_r2_rf)
)

rf_metrics



#########################
#  Importance
rf_importance <- importance(rf_model)

rf_importance_df <- data.frame(
  Feature = rownames(rf_importance),
  Importance = rf_importance[, "IncNodePurity"]  
)

rf_importance_df <- rf_importance_df %>% arrange(desc(Importance))
rf_importance_df

ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Random Forest Feature Importance",
       x = "Feature",
       y = "Importance")


###################################################################
#####XGBOOST
######################################################################

train_matrix <- as.matrix(train_data[, setdiff(names(train_data), "All_Players_Percent")])
test_matrix <- as.matrix(test_data[, setdiff(names(test_data), "All_Players_Percent")])
train_label <- train_data$All_Players_Percent  
test_label <- test_data$All_Players_Percent



#########################



# Define hyperparameter grid
hyper_grid <- list(
  max.depth = c(3, 6, 10, 15),
  eta = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
  subsample = c(0.5, 0.7, 0.9, 1),
  colsample_bytree = c(0.5, 0.7, 0.9, 1),
  gamma = c(0, 0.0001, 0.001, 0.01, 0.1),
  min_child_weight = c(1, 3, 5, 7, 10),
  nrounds = c(200, 500, 1000, 1500),
  lambda = c(1, 3, 5, 10, 20)
)


# Initialize results table
cv_results <- data.frame(
  hyperparam = character(),
  value = numeric(),
  cv_rmse = numeric(),
  cv_r2 = numeric(),
  stringsAsFactors = FALSE
)


# Base parameters
base_params <- list(
  max.depth = 6,
  eta = 0.3,
  subsample = 1,
  colsample_bytree = 1,
  gamma = 0,
  min_child_weight = 1,
  lambda = 1,
  nrounds = 500
)

# Loop over each hyperparameter
for (param_name in names(hyper_grid)) {
  for (value in hyper_grid[[param_name]]) {
    
    # Set parameters for this run
    params <- base_params
    params[[param_name]] <- value
    
    # Cross-validation without early stopping
    cv <- xgb.cv(
      seed = 123,
      data = train_matrix,
      label = train_label,
      nfold = 5,
      nrounds = params$nrounds,
      max_depth = params$max.depth,
      eta = params$eta,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      gamma = params$gamma,
      min_child_weight = params$min_child_weight,
      lambda = params$lambda,
      objective = "reg:squarederror",
      metrics = list("rmse"),
      verbose = 0
    )
    
    # RMSE at final boosting round
    mean_rmse <- cv$evaluation_log$test_rmse_mean[params$nrounds]
    
    # Approximate R^2
    target_var <- var(train_label)
    cv_r2 <- 1 - (mean_rmse^2 / target_var)
    
    # Save results
    cv_results <- rbind(
      cv_results,
      data.frame(
        hyperparam = param_name,
        value = value,
        cv_rmse = mean_rmse,
        cv_r2 = cv_r2
      )
    )
  }
}


cv_results_xg <- cv_results %>% arrange(hyperparam, cv_rmse)
cv_results_xg


####################
##FINAL SET
#insert best hyperparameters
###################

bst <- xgboost(
  data = train_matrix,
  label = train_label,
  max.depth = 10,       
  eta = 0.2,          
  nrounds = 500,       
  gamma = 0,       
  subsample = 1,     
  min_child_weight = 1,   
  colsample_bytree = 1, 
  lambda = 1,             
  
  objective = "reg:squarederror",
  nthread = 2,                   
  verbose = 0                     
)

# Training predictions
train_pred <- predict(bst, train_matrix)

# Test predictions
test_pred <- predict(bst, test_matrix)

train_rmse <- rmse_vec(truth = train_label, estimate = train_pred)
test_rmse  <- rmse_vec(truth = test_label, estimate = test_pred)


train_r2 <- rsq_vec(truth = train_label, estimate = train_pred)
test_r2  <- rsq_vec(truth = test_label, estimate = test_pred)

train_mae <- mae_vec(truth = train_label, estimate = train_pred)
test_mae  <- mae_vec(truth = test_label,  estimate = test_pred)

xg_metrics <- data.frame(
  metric = c("RMSE", "R2", "MAE"),
  train  = c(train_rmse, train_r2, train_mae),
  test   = c(test_rmse, test_r2, test_mae)
)
xg_metrics



##########################
#WITH EARLY STOPPING
###########################
hyper_grid <- list(
  max.depth = c(3, 6, 10, 15),
  eta = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5),
  subsample = c(0.5, 0.7, 0.9, 1),
  colsample_bytree = c(0.5, 0.7, 0.9, 1),
  gamma = c(0, 0.0001, 0.001, 0.01, 0.1),
  min_child_weight = c(1, 3, 5, 7, 10),
  lambda = c(1, 3, 5, 10, 20)
)

early_cv_results <- data.frame(
  hyperparam = character(),
  value = numeric(),
  cv_rmse = numeric(),
  cv_r2 = numeric(),
  best_nrounds = integer(),
  stringsAsFactors = FALSE
)

base_params <- list(
  max.depth = 6,
  eta = 0.3,
  subsample = 1,
  colsample_bytree = 1,
  gamma = 0,
  min_child_weight = 1,
  lambda = 1
)


for (param_name in names(hyper_grid)) {
  for (value in hyper_grid[[param_name]]) {
    
    params <- base_params
    params[[param_name]] <- value
    
    # Cross-validation with a high nrounds and early stopping
    cv <- xgb.cv(
      data = train_matrix,
      label = train_label,
      nfold = 5,
      nrounds = 2000,
      max_depth = params$max.depth,
      eta = params$eta,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      gamma = params$gamma,
      min_child_weight = params$min_child_weight,
      lambda = params$lambda,
      objective = "reg:squarederror",
      metrics = "rmse",
      early_stopping_rounds = 10,
      verbose = 0
    )
    
    mean_rmse <- cv$evaluation_log$test_rmse_mean[cv$best_iteration]
    target_var <- var(train_label)
    cv_r2 <- 1 - (mean_rmse^2 / target_var)
    
    early_cv_results <- rbind(early_cv_results, data.frame(
      hyperparam = param_name,
      value = value,
      cv_rmse = mean_rmse,
      cv_r2 = cv_r2,
      best_nrounds = cv$best_iteration
    ))
  }
}

# Examine results and pick best hyperparameters
cv_results_xg_early <- early_cv_results %>% arrange(hyperparam, cv_rmse)
cv_results_xg_early


# 3. Run CV again with best hyperparameters
cv_final <- xgb.cv(
  data = train_matrix,
  label = train_label,
  nfold = 5,
  nrounds = 2000,
  max_depth = 6,
  eta = 0.3,
  subsample = 0.7,
  colsample_bytree = 0.5,
  gamma = 0,
  min_child_weight = 10,
  lambda = 3,
  objective = "reg:squarederror",
  metrics = "rmse",
  early_stopping_rounds = 10,
  verbose = 0
)
best_iteration <- cv_final$best_iteration
best_iteration


####################
##FINAL SET
#insert best hyperparameters
###################

early_bst <- xgboost(
  data = train_matrix,
  label = train_label,
  max.depth = 6,       
  eta = 0.3,          
  nrounds = 21,       
  gamma = 0,       
  subsample = 0.7,     
  min_child_weight = 10,   
  colsample_bytree = 0.5, 
  lambda = 3,             
  
  objective = "reg:squarederror",
  nthread = 2,                   
  verbose = 0                     
)

# Training predictions
train_pred <- predict(early_bst, train_matrix)

# Test predictions
test_pred <- predict(early_bst, test_matrix)

train_rmse <- rmse_vec(truth = train_label, estimate = train_pred)
test_rmse  <- rmse_vec(truth = test_label, estimate = test_pred)


train_r2 <- rsq_vec(truth = train_label, estimate = train_pred)
test_r2  <- rsq_vec(truth = test_label, estimate = test_pred)

train_mae <- mae_vec(truth = train_label, estimate = train_pred)
test_mae  <- mae_vec(truth = test_label,  estimate = test_pred)

early_xg_metrics <- data.frame(
  metric = c("RMSE", "R2", "MAE"),
  train  = c(train_rmse, train_r2, train_mae),
  test   = c(test_rmse, test_r2, test_mae)
)
early_xg_metrics


##################################
###IMPORTANCE
#####################


xgb_importance <- xgb.importance(
  feature_names = colnames(train_matrix),
  model = bst 
)
top_features <- xgb_importance[order(-Gain)]

ggplot(top_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  theme_minimal() +
  labs(title = "XGBoost Feature Importance",
       x = "Feature",
       y = "Gain")


xgb_importance <- xgb.importance(
  feature_names = colnames(train_matrix),
  model = early_bst 
)
top_features <- xgb_importance[order(-Gain)]

ggplot(top_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "darkred") +
  coord_flip() +
  theme_minimal() +
  labs(title = "XGBoost-ES Feature Importance",
       x = "Feature",
       y = "Gain")


