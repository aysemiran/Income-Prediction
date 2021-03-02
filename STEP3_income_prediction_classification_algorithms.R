library(e1071)
library(caTools)
library(tree) # Contains the "tree" function
library(caret) # Contains classification and regression training

trainDataset <- read.csv("path/to/income_prediction/datasets/underSampledData.csv", sep=",",dec = ",")
trainDataset$income_.50K<- as.factor(trainDataset$income_.50K) #convert target feature to a factor

trainDataset[c(1,3,8,9,10)]<- as.numeric(unlist(trainDataset[c(1,3,8,9,10)]))
##### Machine Learning Models, their runtimes and accuracy plots ##### 

set.seed(80654)
# k-fold parameters
tControl <- trainControl(method="repeatedcv", repeats = 5, number=4, verboseIter = TRUE)

# ML models create, tune parameters plot, run time calculation and accuracy plot

# Naive Bayes Model
start_time <- Sys.time()
naive_model <- train(income_.50K ~ age + workclass + educational.num + marital.status +
                       occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week, data=trainDataset,
                     na.action=na.pass, method="naive_bayes", trControl=tControl,
                     tuneLength=13)
finish_time <- Sys.time()
naive_time <- finish_time - start_time

plot(naive_model)
plot(row.names(naive_model$resample),naive_model$resample[,1], main = "Bayes
Accuracy Plot",type="b" , lwd=2 ,col= "blue" , ylab="Accuracy" , xlab="Repeat Number" ,
     bty="l" , pch=20 , cex=2, xlim = c(1,21), ylim = c(0.73,0.85))

# Decision Tree Model
start_time <- Sys.time()
dtModel <- train(income_.50K ~ age + workclass + educational.num +
                   marital.status + occupation + relationship + gender + capital.loss+ capital.gain +
                   hours.per.week, data=trainDataset,
                 na.action=na.pass, method="rpart2", trControl=tControl, tuneLength=15)
finish_time <- Sys.time()
decision_tree_time <- finish_time - start_time
plot(dtModel)
fancyRpartPlot(dtModel$finalModel)
plot(row.names(dtModel$resample),dtModel$resample[,1], main = "Decision Tree Accuracy
Plot",type="b" , lwd=2 ,col= "blue" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" ,
     pch=20, cex=2, xlim = c(1,21), ylim = c(0.73,0.85))

# K-Nearest Neighbor Model 
start_time <- Sys.time()
knnModel <- train(income_.50K ~ age + workclass + educational.num + marital.status +
                    occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week, data=trainDataset,
                  na.action=na.pass, method="knn", trControl=tControl, tuneLength=10)
finish_time <- Sys.time()
kNN_time <- finish_time - start_time
plot(knnModel ,cex=2)
plot(row.names(knnModel$resample),knnModel$resample[,1], main = "Knn Accuracy
Plot",type="b" , lwd=2 ,col= "green" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" ,
     pch=20, cex=2, xlim = c(1,21), ylim = c(0.73,0.85))

# Logistic Regression Model 
start_time <- Sys.time()
logisticModel <-  train(income_.50K ~ age + workclass + educational.num + marital.status +
                          occupation + relationship + gender + capital.loss+ capital.gain + hours.per.week , data=trainDataset,
                        na.action=na.pass, method="regLogistic", trControl=tControl)
finish_time <- Sys.time()
logistic_time <- finish_time - start_time

plot(logisticModel)
plot(row.names(logisticModel$resample),logisticModel$resample[,1], main = "Logistic
Accuracy Plot",type="b" , lwd=2 ,col= "yellow" , ylab="Accuracy" , xlab="Repeat Number"
     , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.7,0.85))

# Support Vector Machine Model 
start_time <- Sys.time()
svmModel <- train(income_.50K ~ age + workclass + educational.num + marital.status +
                    occupation + relationship + gender + capital.loss+ capital.gain+ hours.per.week , data=trainDataset,
                  na.action=na.pass, method="svmLinearWeights2", trControl=tControl)
finish_time <- Sys.time()
svm_time <- finish_time - start_time

plot(row.names(svmModel$resample),svmModel$resample[,1], main = "SVM Accuracy
Plot",type="b" , lwd=2 ,col= "purple" , ylab="Accuracy" , xlab="Repeat Number" , bty="l" ,
     pch=20, cex=2, xlim = c(1,21), ylim = c(0.75,0.85))
plot(svmModel)

# Neural Network Model 
start_time <- Sys.time()
nnModel <- train(income_.50K ~ age + workclass + educational.num + marital.status +
          occupation + relationship + gender + capital.loss + capital.gain + hours.per.week , data=trainDataset,
        na.action=na.pass, method="nnet", trControl=tControl)
finish_time <- Sys.time()
nn_time <- finish_time - start_time

plot(row.names(nnModel$resample),nnModel$resample[,1], main = "Neural Network
Accuracy Plot",type="b" , lwd=2 ,col= "orange" , ylab="Accuracy" , xlab="Repeat Number"
     , bty="l" , pch=20, cex=2, xlim = c(1,21), ylim = c(0.75,0.85))
plot(nnModel)

# Accuracy Metrics Calculation
accuracy_metrics_of_model <- function(model){
  result <- confusionMatrix(model)
  precision <- result$table[1, 1] / (result$table[1, 1] + result$table[1, 2])
  recall <- result$table[1, 1] / (result$table[1, 1] + result$table[2, 1])
  f1_score <- 2 * ((precision*recall) / (precision + recall))
  #Print the table
  print("Confusion Matrix: ")
  print(confusionMatrix(model)$table)
  print(paste0("Precision: ", precision))
  print(paste0("Recall: ", recall))
  print(paste0("F-score: : ", f1_score))
  print(paste0("Accuracy: : ", max(model$results$Accuracy)))
  print(paste0("Error Rate: : ", 1-max(model$results$Accuracy)))
}

accuracy_metrics_of_model(naive_model)
statistics_of_model(dtModel)
statistics_of_model(knnModel)
statistics_of_model(logisticModel)
statistics_of_model(svmModel)
statistics_of_model(nnModel)

# Accuracy Comparison plot
plot( naive_model$resample[,1] ~row.names(naive_model$resample) , type="b" , bty="l" ,
      xlab="Repeat Number" , ylab="Accuracy" , col= "red" , lwd=2 , pch=20 , ylim=c(0.7,0.93) ,
      bty="l" , cex=2, xlim =c(1,21))
lines(dtModel$resample[,1] ~row.names(dtModel$resample) ,type="b", lwd=2 ,col= "blue",
      bty="l" , pch=20, cex=2)
lines(knnModel$resample[,1] ~row.names(knnModel$resample) ,type="b", lwd=2 ,col=
        "green", bty="l" , pch=20, cex=2)
lines(logisticModel$resample[,1] ~row.names(logisticModel$resample) ,type="b", lwd=2
      ,col= "yellow", bty="l" , pch=20, cex=2)
lines(svmModel$resample[,1] ~row.names(svmModel$resample) ,type="b", lwd=2 ,col=
        "purple", bty="l" , pch=20, cex=2)
lines(nnModel$resample[,1] ~row.names(nnModel$resample) ,type="b", lwd=2 ,col=
        "orange", bty="l" , pch=20, cex=2)
legend("topright",legend = c("Naive Bayes", "Decision Tree", "k-Nearest Neighbor", 
    "Logistic Regression", "Support Vector Machine", "Neural Network"),
       col = c("red", "blue", "green", "yellow","purple", "orange"),
       pch = c(20,20,20,20,20),
       bty = "n",
       text.col = "black",
       horiz = F,
    pt.cex = 2,
       cex = 0.7,y.intersp=0.5)
