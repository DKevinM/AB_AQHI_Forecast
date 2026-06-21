library(data.table)
library(Cubist)

dir.create("models", showWarnings = FALSE)

cat("Loading training dataset...\n")

data <- fread("/tmp/training_dataset.csv.gz")

feature_cols <- c(
  "AQHI",
  "AQHI_lag1",
  "AQHI_lag2",
  "AQHI_lag3",
  "AQHI_lag6",
  "AQHI_lag12",
  "AQHI_lag24",
  "AQHI_change_1h",
  "AQHI_change_3h",
  "PM25",
  "NO2",
  "O3",
  "WS",
  "WD",
  "U",
  "V",
  "TEMP",
  "RH_model",
  "sin_hour",
  "cos_hour",
  "sin_doy",
  "cos_doy",
  "lat_norm",
  "lon_norm",
  "dist_center",
  "PM25_filled",
  "NO2_filled",
  "O3_filled",
  "TEMP_filled",
  "RH_was_missing",
  "WS_filled",
  "WD_filled",
  "WS_future_1h",
  "WD_future_1h",
  "TEMP_future_1h",
  "RH_future_1h",
  "U_future_1h",
  "V_future_1h"
)

target <- "AQHI_future_1h"

keep <- c(feature_cols, target)

data <- data[complete.cases(data[, ..keep])]

cat("Rows:", nrow(data), "\n")

split_index <- floor(nrow(data) * 0.80)

train <- data[1:split_index]
test  <- data[(split_index+1):nrow(data)]

x_train <- train[, ..feature_cols]
y_train <- train[[target]]

x_test <- test[, ..feature_cols]
y_test <- test[[target]]

cat("Training Cubist...\n")

model <- cubist(
  x = x_train,
  y = y_train,
  committees = 50,
  neighbors = 9
)

pred <- predict(model, x_test)

rmse <- sqrt(mean((pred - y_test)^2))
mae  <- mean(abs(pred - y_test))
r2   <- cor(pred, y_test)^2

saveRDS(
  model,
  "models/aqhi_1h_cubist.rds"
)

writeLines(
  c(
    paste("RMSE:", rmse),
    paste("MAE:", mae),
    paste("R2:", r2)
  ),
  "models/aqhi_1h_cubist_metrics.txt"
)

cat("Finished\n")
