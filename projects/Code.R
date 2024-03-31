### Loading the dataset and simplified preprocessing ###
# Load necessary library
library(tidyverse)
# Load the dataset
dataset <- read.csv("C:/Users/sgayc/Desktop/Uni/7.Semester/Data Analytics/Sem 8/Portfolio/Projects/r_project/twitter_dataset.csv")
# Rename columns
colnames(dataset) <- c("target", "ids", "date", "flag", "user", "text")
# Check for missing values
summary(dataset)
# Alternatively, for a more detailed approach:
sapply(dataset, function(x) sum(is.na(x)))
# Check for duplicates
sum(duplicated(dataset)
# -> No duplicates or missing values


### Randomisation for the sakes of computational power ###
# Sample 1000 random rows from the dataset
sampled_dataset <- dataset %>% sample_n(5000)


### Checking for the possible values for 
# List unique values in the 'target' column
unique_values_in_target <- unique(sampled_dataset$target)
# Print the unique values
print(unique_values_in_target)
# -> Tweets are either negative (0) or positive (4)
# Transform the 'target' column: 0 remains 0 (negative sentiment), and 4 becomes 1 (positive sentiment)
sampled_dataset <- sampled_dataset %>%
  mutate(target = ifelse(target == 4, 1, target))
# Check the transformation by viewing unique values again
unique(sampled_dataset$target)


### Tokenisation of the text variable
# Installing libraries
install.packages("tm")
install.packages("caret")
install.packages("e1071")
library(tm)
library(caret)
library(e1071) # For Machine Learning
library(NLP)
# Convert the text data to UTF-8 encoding
sampled_dataset$text <- iconv(sampled_dataset$text, to = "UTF-8")
# Create a corpus from the text column
corpus <- VCorpus(VectorSource(sampled_dataset$text))
# Preprocess the corpus: remove punctuation, numbers, and stopwords, and lowercase the text
corpus_clean <- tm_map(corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords("english"))
# Tokenize the cleaned corpus
dtm <- DocumentTermMatrix(corpus_clean)
# Convert document-term matrix to a data frame for modeling
dtm_df <- as.data.frame(as.matrix(dtm))
colnames(dtm_df) <- make.names(colnames(dtm_df))


### Natural Language Processing using Machine Learning
# Bind the target variable to the DTM data frame
dtm_df$target <- sampled_dataset$target
# Split data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(dtm_df$target, p = .8, list = FALSE, times = 1)
dtm_train <- dtm_df[trainIndex, ]
dtm_test <- dtm_df[-trainIndex, ]

#Naive Bayes Model
# Train a Naive Bayes model
bayes_model <- naiveBayes(target ~ ., data = dtm_train)
# Predict on test set
bayes_predictions <- predict(bayes_model, dtm_test)
# Calculate accuracy
bayes_accuracy <- sum(bayes_predictions == dtm_test$target) / nrow(dtm_test)
print(paste("Bayes Accuracy:", bayes_accuracy))
# Create a new data frame with actual and predicted values for Bayes
bayes_results_table <- data.frame(Actual = dtm_test$target, Predicted = bayes_predictions)
head(bayes_results_table)



#SVM Model
# Train the SVM model
svm_model <- svm(target ~ ., data = dtm_train, kernel = "linear", type = "C-classification")
# Predict using the SVM model
svm_predictions <- predict(svm_model, dtm_test)
# Calculate accuracy
svm_accuracy <- sum(svm_predictions == dtm_test$target) / nrow(dtm_test)
print(paste("SVM Accuracy:", svm_accuracy))
# Create a new data frame with actual and predicted values for SVM
svm_results_table <- data.frame(Actual = dtm_test$target, Predicted = svm_predictions)
head(svm_results_table)




