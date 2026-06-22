library(data.table)
library(Cubist)

dir.create("models", showWarnings = FALSE)

cat("Loading training dataset...\n")

data <- fread("/tmp/training_dataset.csv.gz")





# -----------------------------------
# RH MODEL FIELD
# -----------------------------------

if (!"RH_model" %in% names(data)) {

  cat("Creating RH_model from RH...\n")

  data[, RH_model := RH]

  if ("RH_filled" %in% names(data)) {

    rh_filled_max <- suppressWarnings(
      max(data$RH_filled, na.rm = TRUE)
    )

    if (!is.na(rh_filled_max) && rh_filled_max > 1) {

      cat("Using RH_filled as filled RH values.\n")

      data[is.na(RH_model), RH_model := RH_filled]

    } else {

      cat("RH_filled appears to be a flag.\n")

    }
  }
}

if (!"RH_was_missing" %in% names(data)) {

  data[, RH_was_missing := 0]

}

cat("RH missing:", sum(is.na(data$RH)), "\n")
cat("RH_model missing:", sum(is.na(data$RH_model)), "\n")





cat("Columns loaded:", ncol(data), "\n")

required_cols <- c(
  "U",
  "V",

  "AQHI_future_1h",
#  "AQHI_future_2h",
#  "AQHI_future_3h",

  "WS_future_1h",
  "WD_future_1h",
  "TEMP_future_1h",
  "RH_future_1h",
  "U_future_1h",
  "V_future_1h",

  "WS_future_2h",
  "WD_future_2h",
  "TEMP_future_2h",
  "RH_future_2h",
  "U_future_2h",
  "V_future_2h",

  "WS_future_3h",
  "WD_future_3h",
  "TEMP_future_3h",
  "RH_future_3h",
  "U_future_3h",
  "V_future_3h"
)

missing_cols <- setdiff(
  required_cols,
  names(data)
)


if(length(missing_cols) > 0){
  stop(
    paste(
      "Missing columns:",
      paste(missing_cols, collapse=", ")
    )
  )
}





results <- list()

train_cubist_model <- function(target, name, future_suffix){

cat("\n===================================\n")
cat("Training:", name, "\n")
cat("Target:", target, "\n")
cat("===================================\n")

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

paste0("WS_future_", future_suffix),
paste0("WD_future_", future_suffix),
paste0("TEMP_future_", future_suffix),
paste0("RH_future_", future_suffix),
paste0("U_future_", future_suffix),
paste0("V_future_", future_suffix)


)

keep <- c(feature_cols, target)

d <- data[complete.cases(data[, ..keep])]

cat("Rows:", nrow(d), "\n")

split_index <- floor(nrow(d) * 0.80)

train <- d[1:split_index]
test  <- d[(split_index + 1):nrow(d)]

x_train <- train[, ..feature_cols]
y_train <- train[[target]]

x_test <- test[, ..feature_cols]
y_test <- test[[target]]

cat("Training rows:", nrow(x_train), "\n")
cat("Testing rows :", nrow(x_test), "\n")

model <- cubist(
x = x_train,
y = y_train,
committees = 30,
neighbors = 5
)

pred <- predict(model, x_test)

rmse <- sqrt(mean((pred - y_test)^2))
mae  <- mean(abs(pred - y_test))
r2   <- cor(pred, y_test)^2

high4 <- y_test >= 4
high6 <- y_test >= 6

high4_rmse <- NA
high4_mae  <- NA
high6_rmse <- NA
high6_mae  <- NA

if(sum(high4) > 0){
high4_rmse <- sqrt(mean((pred[high4] - y_test[high4])^2))
high4_mae  <- mean(abs(pred[high4] - y_test[high4]))
}

if(sum(high6) > 0){
high6_rmse <- sqrt(mean((pred[high6] - y_test[high6])^2))
high6_mae  <- mean(abs(pred[high6] - y_test[high6]))
}

cat("\nResults\n")
cat("RMSE:", round(rmse,3), "\n")
cat("MAE :", round(mae,3), "\n")
cat("R²  :", round(r2,3), "\n")

saveRDS(
model,
paste0("models/", name, "_cubist.rds")
)

writeLines(
c(
paste("Model:", name),
paste("Rows:", nrow(d)),
paste("Training Rows:", nrow(x_train)),
paste("Testing Rows:", nrow(x_test)),
paste("RMSE:", rmse),
paste("MAE:", mae),
paste("R2:", r2),
paste("AQHI_GE_4_Count:", sum(high4)),
paste("AQHI_GE_4_RMSE:", high4_rmse),
paste("AQHI_GE_4_MAE:", high4_mae),
paste("AQHI_GE_6_Count:", sum(high6)),
paste("AQHI_GE_6_RMSE:", high6_rmse),
paste("AQHI_GE_6_MAE:", high6_mae)
),
paste0("models/", name, "_cubist_metrics.txt")
)

results[[name]] <<- data.frame(
Model = name,
RMSE = rmse,
MAE = mae,
R2 = r2,
AQHI_GE_4_RMSE = high4_rmse,
AQHI_GE_4_MAE = high4_mae,
AQHI_GE_6_RMSE = high6_rmse,
AQHI_GE_6_MAE = high6_mae
)
}

train_cubist_model(
"AQHI_future_1h",
"aqhi_1h",
"1h"
)

# train_cubist_model(
# "AQHI_future_2h",
# "aqhi_2h",
# "2h"
# )

# train_cubist_model(
# "AQHI_future_3h",
# "aqhi_3h",
# "3h"
# )

summary_df <- do.call(rbind, results)

write.csv(
summary_df,
"models/cubist_model_summary.csv",
row.names = FALSE
)

cat("\n===================================\n")
cat("CUBIST MODEL SUMMARY\n")
cat("===================================\n")

print(summary_df)

cat("\nFinished.\n")
