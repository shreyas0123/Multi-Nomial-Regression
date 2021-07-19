################################# problem1 ####################################################
#load the dataset
mdata <- read.csv("C:\\Users\\DELL\\Downloads\\mdata.csv")
colnames(mdata)

#EDA
mdata <- mdata[-c(1,2)]

###################
# Data Partitioning
n <-  nrow(mdata)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- mdata[train_index, ]
test <-  mdata[-train_index, ]

install.packages("nnet")
library(nnet)
commute <- multinom(prog ~ female + ses + schtyp + read + write + math + science + honors, data = train)
summary(commute)


##### Significance of Regression Coefficients###
z <- summary(commute)$coefficients / summary(commute)$standard.errors

p_value <- (1 - pnorm(abs(z), 0, 1)) * 2

summary(commute)$coefficients
p_value

# odds ratio 
exp(coef(commute))

# check for fitted values on training data
prob <- fitted(commute)


# Predicted on test data
pred_test <- predict(commute, newdata =  test, type = "probs") # type="probs" is to calculate probabilities
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

predtest_name <- apply(pred_test, 1, get_names)
?apply
pred_test$prediction <- predtest_name
View(pred_test)

# Confusion matrix
table(predtest_name, test$prog)

# confusion matrix visualization
# barplot(table(predtest_name, test$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), legend = c("bus", "car", "carpool", "rail"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtest_name, test$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy on test data
mean(predtest_name == test$prog)

################
# Training Data
pred_train <- predict(commute, newdata =  train, type="probs") # type="probs" is to calculate probabilities
pred_train

# Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL


predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)

# Confusion matrix
table(predtrain_name, train$prog)

# confusion matrix visualization
# barplot(table(predtrain_name, train$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), legend = c("bus", "car", "carpool", "rail"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtrain_name, train$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy 
mean(predtrain_name == train$prog)

################################ problem2 ######################################################3
#load the dataset
loan_data <- read.csv("C:\\Users\\DELL\\Downloads\\loan.csv")
colnames(loan_data)

#EDA
#select required 
loan_data <- loan_data[,c(3,4,5,6,7,8,9,10,13,17)]
summary(loan_data)
str(loan_data)

#Label Encoding
factors <- factor(loan_data$term)
loan_data$term <- as.numeric(factors)

factors <- factor(loan_data$int_rate)
loan_data$int_rate <- as.numeric(factors)

factors <- factor(loan_data$grade)
loan_data$grade <- as.numeric(factors)

factors <- factor(loan_data$sub_grade)
loan_data$sub_grade <- as.numeric(factors)

factors <- factor(loan_data$home_ownership)
loan_data$home_ownership  <- as.numeric(factors)

###################
# Data Partitioning
n <-  nrow(loan_data)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- loan_data[train_index, ]
test <-  loan_data[-train_index, ]

install.packages("nnet")
library(nnet)
str(loan_data)

loan_data$term <- as.numeric(loan_data$term)
loan_data$int_rate <- as.numeric(loan_data$int_rate)
loan_data$grade <- as.numeric(loan_data$grade)
loan_data$sub_grade <- as.numeric(loan_data$sub_grade)
loan_data$home_ownership <- as.numeric(loan_data$home_ownership)
loan_data$loan_status <- as.numeric(loan_data$loan_status)


commute <- multinom(loan_status ~ loan_amnt + funded_amnt + funded_amnt_inv + term + int_rate + installment + grade + sub_grade,home_ownership, data = train)
summary(commute)

##### Significance of Regression Coefficients###
z <- summary(commute)$coefficients / summary(commute)$standard.errors

p_value <- (1 - pnorm(abs(z), 0, 1)) * 2

summary(commute)$coefficients
p_value

# odds ratio 
exp(coef(commute))

# check for fitted values on training data
prob <- fitted(commute)


# Predicted on test data
pred_test <- predict(commute, newdata =  test, type = "probs") # type="probs" is to calculate probabilities
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

predtest_name <- apply(pred_test, 1, get_names)
?apply
pred_test$prediction <- predtest_name
View(pred_test)

# Confusion matrix
table(predtest_name, test$loan_status)

# confusion matrix visualization
# barplot(table(predtest_name, test$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), legend = c("bus", "car", "carpool", "rail"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtest_name, test$loan_status), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy on test data
mean(predtest_name == test$loan_status)

################
# Training Data
pred_train <- predict(commute, newdata =  train, type="probs") # type="probs" is to calculate probabilities
pred_train

# Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL


predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)

# Confusion matrix
table(predtrain_name, train$loan_status)

# confusion matrix visualization
# barplot(table(predtrain_name, train$prog), beside = T, col =c("red", "lightgreen", "blue", "orange"), legend = c("bus", "car", "carpool", "rail"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
barplot(table(predtrain_name, train$loan_status), beside = T, col =c("red", "lightgreen", "blue", "orange"), main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

# Accuracy 
mean(predtrain_name == train$loan_status)




