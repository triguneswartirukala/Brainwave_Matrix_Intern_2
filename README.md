# Brainwave_Matrix_Intern_2

title: "Credit Card Fraud Detection"author: "Data Analyst"date: "2025-07-28"output: html_document
Credit Card Fraud Detection Analysis
This R Markdown document outlines a workflow for detecting credit card fraud using machine learning in R. We use a dataset with transaction features and a binary class label (0 = non-fraud, 1 = fraud). The dataset is assumed to be highly imbalanced, with far fewer fraud cases than non-fraud cases.
Setup
Load required libraries for data manipulation, visualization, modeling, and evaluation.
library(tidyverse)    # Data manipulation and visualization
library(caret)        # Machine learning workflow
library(randomForest)  # Random forest model
library(ROSE)         # Handling imbalanced data
library(skimr)        # Data summary
library(ggplot2)      # Plotting
set.seed(123)         # Reproducibility

Data Loading
Load the credit card fraud dataset. Replace "creditcard.csv" with the path to your dataset. The dataset should have columns like Time, Amount, V1 to V28, and Class.
# Load dataset (example: Kaggle Credit Card Fraud Detection dataset)
data <- read.csv("creditcard.csv")
# Quick overview
head(data)

Exploratory Data Analysis (EDA)
Summarize the dataset to understand its structure and check for missing values.
# Summary statistics
skim(data)

# Class distribution (imbalanced dataset)
table(data$Class)
prop.table(table(data$Class)) * 100  # Percentage of fraud vs. non-fraud

# Plot class distribution
ggplot(data, aes(x = factor(Class))) +
  geom_bar() +
  labs(title = "Class Distribution", x = "Class (0 = Non-Fraud, 1 = Fraud)", y = "Count") +
  theme_minimal()

# Distribution of Amount and Time
ggplot(data, aes(x = Amount)) +
  geom_histogram(bins = 50) +
  labs(title = "Transaction Amount Distribution", x = "Amount", y = "Frequency") +
  theme_minimal()

ggplot(data, aes(x = Time)) +
  geom_histogram(bins = 50) +
  labs(title = "Transaction Time Distribution", x = "Time (seconds)", y = "Frequency") +
  theme_minimal()

Data Preprocessing
Scale numerical features (Time and Amount) to have mean = 0 and standard deviation = 1, as PCA-derived features (V1 to V28) are already scaled. Split the data into training and testing sets.
# Scale Time and Amount
data$Time <- scale(data$Time)
data$Amount <- scale(data$Amount)

# Convert Class to factor
data$Class <- as.factor(data$Class)

# Split into training (70%) and testing (30%)
set.seed(123)
trainIndex <- createDataPartition(data$Class, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

Handling Imbalanced Data
Since fraud cases are rare, we use oversampling with ROSE to balance the training data.
# Apply ROSE oversampling to training data
train_balanced <- ROSE(Class ~ ., data = trainData, seed = 123)$data
table(train_balanced$Class)  # Check new class distribution

Model Training
Train two models: logistic regression and random forest.
Logistic Regression
# Train logistic regression
log_model <- train(Class ~ ., data = train_balanced, 
                   method = "glm", 
                   family = "binomial",
                   trControl = trainControl(method = "cv", number = 5))

# Predict on test set
log_pred <- predict(log_model, testData)
log_conf <- confusionMatrix(log_pred, testData$Class, positive = "1")
log_conf

Random Forest
# Train random forest
rf_model <- train(Class ~ ., data = train_balanced, 
                  method = "rf", 
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = data.frame(mtry = c(2, 4, 6)))

# Predict on test set
rf_pred <- predict(rf_model, testData)
rf_conf <- confusionMatrix(rf_pred, testData$Class, positive = "1")
rf_conf

Model Evaluation
Compare model performance using metrics like precision, recall, F1-score, and AUC.
# Print confusion matrices
print("Logistic Regression Confusion Matrix:")
log_conf

print("Random Forest Confusion Matrix:")
rf_conf

# Calculate ROC and AUC (requires pROC package)
library(pROC)
log_prob <- predict(log_model, testData, type = "prob")[,2]
rf_prob <- predict(rf_model, testData, type = "prob")[,2]

log_roc <- roc(testData$Class, log_prob)
rf_roc <- roc(testData$Class, rf_prob)

print(paste("Logistic Regression AUC:", auc(log_roc)))
print(paste("Random Forest AUC:", auc(rf_roc)))

# Plot ROC curves
plot(log_roc, col = "blue", main = "ROC Curves")
plot(rf_roc, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic", "Random Forest"), col = c("blue", "red"), lwd = 2)

Conclusion
The random forest model typically outperforms logistic regression for fraud detection due to its ability to capture non-linear relationships. However, both models should be tuned further and validated on new data. Key metrics like recall (detecting fraud cases) and AUC are critical for imbalanced datasets.

To run this analysis, ensure you have the dataset (e.g., creditcard.csv) and install the required R packages (tidyverse, caret, randomForest, ROSE, skimr, ggplot2, pROC).
