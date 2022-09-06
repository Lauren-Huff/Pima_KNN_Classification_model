#Load Libraries
library(ROCR)
library(caret)
library(class)
library(ggplot2)

#Read CSV
PI_Diabetes <- read.csv("pima-indians-diabetes.csv")

#Set target attribute as a factor 
PI_Diabetes$class <- as.factor(PI_Diabetes$class)

#Create a function to scale numeric attributes
dar2ed.scale.many <- function (dat, column_nos) 
{
  nms = names(dat)
  for (col in column_nos) {
    name = paste(nms[col], "_z", sep = "")
    dat[name] = scale(dat[, col])
  }
  cat(paste("Scaled", length(column_nos), "attribute(s)\n"))
  dat
}
#Scale numeric attributes
PI_Diabetes <- dar2ed.scale.many(PI_Diabetes, c(1:8))

#Create a vector for predictor attributes
names(PI_Diabetes)
predictors <- c(10:17)

#Partition training and test sets
library(caret)
set.seed(2015)
sam <- createDataPartition(PI_Diabetes$class, p = 0.5, list = FALSE)
train.a <- PI_Diabetes[sam,]
rest <- PI_Diabetes[-sam,]
sam <- createDataPartition(rest$class, p=0.5, list = FALSE)
train.b <- rest[sam,]
test <- rest[-sam,]

#Check proportions of partitioned data
table(PI_Diabetes$class)/nrow(PI_Diabetes)
table(train.a$class)/nrow(train.a)
table(train.b$class)/nrow(train.b)

#Build KNN classification models with different k values

## K=3
pred_trg_3 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 3)
cat("Training, k= 3/n")
table(Actual = train.b$class, Predicted = pred_trg_3)

## K=5
pred_trg_5 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 5)
cat("Training, k= 5/n")
table(Actual = train.b$class, Predicted = pred_trg_5)

# K=7
pred_trg_7 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 7)
cat("Training, k= 7/n")
table(Actual = train.b$class, Predicted = pred_trg_7)

# K=9
pred_trg_9 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 9)
cat("Training, k= 9/n")
table(Actual = train.b$class, Predicted = pred_trg_9)

#Create ROC Chart for each K Value to visualize trade off between false positive rate and true positive rate

## K = 3 ROC Chart
pred_trg_3 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 3, prob = TRUE)
p3 <- attr(pred_trg_3, "prob")
prob_plus_trg_3 <- ifelse(pred_trg_3 == "1", p3, 1-p3)
pred3<- prediction(prob_plus_trg_3, train.b$class, label.ordering = c("0", "1"))
perf3 <- performance(pred3, "tpr", "fpr")
plot(perf3)
lines(par()$usr[1:2], par()$usr[3:4])

## K = 5 ROC Chart
pred_trg_5 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 5, prob = TRUE)
p5 <- attr(pred_trg_5, "prob")
prob_plus_trg_5 <- ifelse(pred_trg_5 == "1", p5, 1-p5)
pred5<- prediction(prob_plus_trg_5, train.b$class, label.ordering = c("0", "1"))
perf5 <- performance(pred5, "tpr", "fpr")
plot(perf5)
lines(par()$usr[1:2], par()$usr[3:4])

##K = 7 ROC Chart
pred_trg_7 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 7, prob = TRUE)
p7 <- attr(pred_trg_7, "prob")
prob_plus_trg_7 <- ifelse(pred_trg_7 == "1", p7, 1-p7)
pred7<- prediction(prob_plus_trg_7, train.b$class, label.ordering = c("0", "1"))
perf7 <- performance(pred7, "tpr", "fpr")
plot(perf7)
lines(par()$usr[1:2], par()$usr[3:4])

## K = 9 ROC Chart
pred_trg_9 <- knn(train.a[,predictors], train.b[,predictors], train.a$class, 9, prob = TRUE)
p9 <- attr(pred_trg_9, "prob")
prob_plus_trg_9 <- ifelse(pred_trg_9 == "1", p9, 1-p9)
pred9<- prediction(prob_plus_trg_9, train.b$class, label.ordering = c("0", "1"))
perf9 <- performance(pred9, "tpr", "fpr")
plot(perf9)
lines(par()$usr[1:2], par()$usr[3:4])

#Displaying the cutoff probabilities and corresponding true positive and false positive rates

##Cutoff Probability for K = 3
prob.cuts3 <- data.frame(cut = perf3@alpha.values[[1]], fpr = perf3@x.values[[1]], trp = 
                           perf3@y.values[[1]])
##Cutoff Probability for K = 5
prob.cuts5 <- data.frame(cut = perf5@alpha.values[[1]], fpr = perf5@x.values[[1]], trp = 
                           perf5@y.values[[1]])
##Cutoff Probability for K = 7
prob.cuts7 <- data.frame(cut = perf7@alpha.values[[1]], fpr = perf7@x.values[[1]], trp = 
                           perf7@y.values[[1]])
##Cutoff Probability for K = 9
prob.cuts9 <- data.frame(cut = perf9@alpha.values[[1]], fpr = perf9@x.values[[1]], trp = 
                           perf9@y.values[[1]])

#Selecting a cutoff probability for each K Value

##Cutoff probability for K = 3
cutoff3 <- 0.333333

##Cutoff probability for K = 5
cutoff5 <- 0.4

##Cutoff probability for K = 7
cutoff7 <- 0.2857143

##Cutoff probability for K = 9
cutoff9 <- 0.333

# K = 7 had the best accuracy for predictions in the training partition. I will use K = 7 KNN model for the final model

## K = 7 Performance on the test partition 
pred_test_7 <- knn(train.a[ , predictors], test[ , predictors], train.a$class, 7, prob = TRUE)
p7 <- attr(pred_test_7,"prob")
prob_plus_test_7 <- ifelse(pred_test_7 == "1", p7, 1 - p7)
pred_test_7 <- ifelse(prob_plus_test_7 < cutoff7, "0", "1")
cat("Test, k = 7\n")
table(Actual = test$class, Predicted = pred_test_7)

#Model Accuracy Rate
(93+47)/(93+47+32+20)