# Breast Cancer Diagnosis: Data Analysis and Prediction
# Author: Alexis

# Set working directory
setwd("C:/Users/uzoeg/Downloads/archive (3)/")

# Load necessary libraries
library(tidyverse)
library(caret)
library(e1071)    # For SVM
library(pROC)     # For ROC Curve
library(ggpubr)   # For ANOVA visualization
library(corrplot)

# Load the Data
data <- read.csv("Breast_cancer_data.csv")  

# View first few rows
head(data)

# Structure of the dataset
str(data)

# Check missing values
sum(is.na(data))

# Data Preprocessing
data$diagnosis <- as.factor(data$diagnosis)
table(data$diagnosis)

# Normalize the Data
preProc <- preProcess(data[, 1:(ncol(data)-1)], method = c("center", "scale"))
data_norm <- predict(preProc, data)
data_norm$diagnosis <- data$diagnosis
head(data_norm)

# ANOVA Test
feature_means_by_diagnosis <- data %>%
  group_by(diagnosis) %>%
  summarise(
    mean_radius = mean(mean_radius),
    mean_texture = mean(mean_texture),
    mean_perimeter = mean(mean_perimeter),
    mean_area = mean(mean_area),
    mean_smoothness = mean(mean_smoothness)
  )

print(feature_means_by_diagnosis)

anova_result <- aov(mean_radius ~ diagnosis, data = data_norm)
print(summary(anova_result))

anova_result <- aov(mean_texture ~ diagnosis, data = data_norm)
print(summary(anova_result))

anova_result <- aov(mean_perimeter ~ diagnosis, data = data_norm)
print(summary(anova_result))

anova_result <- aov(mean_area ~ diagnosis, data = data_norm)
print(summary(anova_result))

anova_result <- aov(mean_smoothness ~ diagnosis, data = data_norm)
print(summary(anova_result))

# Data Visualization

# Histograms BEFORE normalization
data %>%
  select(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness) %>%
  gather(key = "Feature", value = "Value") %>%
  ggplot(aes(x = Value, fill = Feature)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(~Feature, scales = "free") +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Histograms of Features Before Normalization")

# Histograms AFTER normalization
data_norm %>%
  select(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness) %>%
  gather(key = "Feature", value = "Value") %>%
  ggplot(aes(x = Value, fill = Feature)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(~Feature, scales = "free") +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Histograms of Features After Normalization")

# Histogram of mean_radius by diagnosis
ggplot(data_norm, aes(x = mean_radius, fill = diagnosis)) +
  geom_histogram(alpha = 0.7, position = "identity", bins = 30) +
  labs(title = "Distribution of Mean Radius by Diagnosis", x = "Mean Radius", y = "Count")

# Histogram of mean_perimeter by diagnosis
ggplot(data_norm, aes(x = mean_perimeter, fill = diagnosis)) +
  geom_histogram(alpha = 0.7, position = "identity", bins = 30) +
  labs(title = "Distribution of Mean Perimeter by Diagnosis", x = "Mean Perimeter", y = "Count")

# Boxplots
ggplot(data_norm, aes(x = diagnosis, y = mean_area, fill = diagnosis)) +
  geom_boxplot() +
  labs(title = "Boxplot of Mean Area by Diagnosis", x = "Diagnosis", y = "Mean Area")

ggplot(data_norm, aes(x = diagnosis, y = mean_texture, fill = diagnosis)) +
  geom_boxplot() +
  labs(title = "Boxplot of Mean Texture by Diagnosis", x = "Diagnosis", y = "Mean Texture")

ggplot(data_norm, aes(x = diagnosis, y = mean_smoothness, fill = diagnosis)) +
  geom_boxplot() +
  labs(title = "Boxplot of Mean Smoothness by Diagnosis", x = "Diagnosis", y = "Mean Smoothness")

# Correlation heatmap
corr_data <- cor(data_norm[,1:(ncol(data_norm)-1)])
corrplot(corr_data, method = "color")

# Train-Test Split
set.seed(123)
splitIndex <- createDataPartition(data_norm$diagnosis, p = 0.7, list = FALSE)
train <- data_norm[splitIndex, ]
test <- data_norm[-splitIndex, ]

# Model 1 - Logistic Regression
log_model <- glm(diagnosis ~ ., data = train, family = binomial)
log_preds <- predict(log_model, newdata = test, type = "response")
log_preds_class <- ifelse(log_preds > 0.5, 1, 0)
log_preds_class <- as.factor(log_preds_class)

confusionMatrix(log_preds_class, test$diagnosis)

roc_obj_log <- roc(as.numeric(as.character(test$diagnosis)), as.numeric(log_preds))

# ROC Curve for Logistic Regression
roc_df <- data.frame(
  TPR = rev(roc_obj_log$sensitivities),
  FPR = rev(1 - roc_obj_log$specificities)
)

ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "blue", size = 1) +
  geom_abline(linetype = "dashed", color = "red") +
  labs(
    title = paste0("Logistic Regression ROC Curve (AUC = ", round(auc(roc_obj_log), 3), ")"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()

# Model 2 - SVM Classifier
svm_model <- svm(diagnosis ~ ., data = train, kernel = "radial", type = "C-classification")
svm_preds <- predict(svm_model, newdata = test)

confusionMatrix(svm_preds, test$diagnosis)

svm_probs <- attr(predict(svm_model, test, decision.values = TRUE), "decision.values")
roc_obj_svm <- roc(as.numeric(as.character(test$diagnosis)), svm_probs)

# ROC Curve for SVM
roc_svm_df <- data.frame(
  TPR = rev(roc_obj_svm$sensitivities),
  FPR = rev(1 - roc_obj_svm$specificities)
)

ggplot(roc_svm_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "darkgreen", size = 1.2) +
  geom_abline(linetype = "dashed", color = "red") +
  labs(
    title = paste0("SVM ROC Curve (AUC = ", round(auc(roc_obj_svm), 3), ")"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()
