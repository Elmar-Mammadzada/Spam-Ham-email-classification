library(readr)
library(dplyr)
library(tm)
library(Matrix)
library(glmnet)
library(caret)

emails <- read_csv("C:/Users/elmar_e6gaevu/OneDrive/Desktop/emails.csv")

glimpse(emails)
table(emails$spam) 

emails <- emails %>%
  mutate(id = row_number()) %>%
  select(id, everything())

corpus <- VCorpus(VectorSource(emails$text))

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

dtm <- DocumentTermMatrix(corpus)

dtm <- removeSparseTerms(dtm, 0.99)

X <- sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v,
                  dims = c(dtm$nrow, dtm$ncol), dimnames = dimnames(dtm))
y <- emails$spam

set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[train_index, ]
y_train <- y[train_index]
X_test  <- X[-train_index, ]
y_test  <- y[-train_index]

set.seed(123)
cv_model <- cv.glmnet(x = X_train, y = as.factor(y_train),
                      family = "binomial",
                      alpha = 1,
                      type.measure = "class")
plot(cv_model)

train_preds <- predict(cv_model, newx = X_train, s = "lambda.min", type = "class")
test_preds  <- predict(cv_model, newx = X_test,  s = "lambda.min", type = "class")

train_acc <- mean(train_preds == y_train)
test_acc  <- mean(test_preds == y_test)

cat("Training Accuracy:", round(train_acc, 4), "\n")
cat("Testing Accuracy:", round(test_acc, 4), "\n")

conf_matrix <- confusionMatrix(as.factor(test_preds), as.factor(y_test), positive = "1")
print(conf_matrix)