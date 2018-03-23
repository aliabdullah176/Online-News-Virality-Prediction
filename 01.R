#Workspace prep
rm(list=ls())
setwd("D:/Study/OneDrive - The University of Texas at Dallas/Semester 2/BUAN 6341 Applied Machine Learning/HW/02")

# Loading libraries
library(data.table); library(caret); library(ROCR); library(ggplot2); library(car); library(reshape) 
library(e1071); library(kernlab); library(parallelSVM); library(evtree); library(parallel); library(doParallel)
rawdata <- fread('data.csv')
head(rawdata)
#removing variables that cause perfect linear dependence
rawdata <- rawdata[,c(-1,-2,-14,-38,-44)]
switch_print <- TRUE; switch_scale <- TRUE  

rawdata1 <- as.data.frame(rawdata)
if (switch_print) {quantile(rawdata1$shares, 0.5)}
rawdata1$shares <- as.factor( as.numeric(rawdata1$shares>1400) )
if (switch_scale) {   for (colName in names(rawdata1)) {
    if(class(rawdata1[,colName]) == 'integer' | class(rawdata1[,colName]) == 'numeric') {
      rawdata1[,colName] <- scale(rawdata1[,colName])
    } } }
if (switch_print) {summary(rawdata1$shares)}

#creating test and train sets
set.seed(176)
indx <- createDataPartition(rawdata1$shares, p = .7, list = F)
train <- ( rawdata1[indx, ] )
test  <- ( rawdata1[-indx, ] )

set.seed(176)
indx2 <- createDataPartition(rawdata1$shares, p = .5, list = F)
train2 <- ( rawdata1[indx2, ] )
test2  <- ( rawdata1[-indx2, ] )

set.seed(176)
indx3 <- createDataPartition(rawdata1$shares, p = .3, list = F)
train3 <- ( rawdata1[indx3, ] )
test3  <- ( rawdata1[-indx3, ] )

#creating svm model with linear kernel
set.seed(176)
t0 <- Sys.time()
svm01 <- parallelSVM(shares ~. , cross=10, scale=FALSE, kernel="linear", data = train, probability = TRUE)
time1 <- difftime(Sys.time(),t0, units="secs")
err_tr1 <- 1-mean(predict(svm01, train) == train$shares)
err_te1 <- 1-mean(predict(svm01, test) == test$shares)

pred1tr <- attr(predict(svm01,train, probability = TRUE), "probabilities")
pred1tra <- pred1tr[,1]
ROCRpred1 <- ROCR::prediction(pred1tra, train$shares)
ROCRperf1 <- performance(ROCRpred1, 'tpr','fpr')
png(filename="SVM_linear_TrainROC")
plot(ROCRperf1, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

pred1te <- attr(predict(svm01,test, probability = TRUE), "probabilities")
pred1tea <- pred1te[,1]
ROCRpredt1 <- ROCR::prediction(pred1tea, test$shares)
ROCRperft1 <- performance(ROCRpredt1, 'tpr','fpr')
png(filename="SVM_linear_TestROC")
plot(ROCRperft1, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

#svm with polynomial kernel
set.seed(176)
t0 <- Sys.time()
#svm02 <- svm(shares ~. , kernel="polynomial", data = train, probability=TRUE)
svm02 <- parallelSVM(shares ~. , cross=10, scale=FALSE, kernel="polynomial", degree=6, data = train, probability = TRUE)
time2 <- difftime(Sys.time(),t0, units="secs")
err_tr2 <- 1-mean(predict(svm02, train) == train$shares)
err_te2 <- 1-mean(predict(svm02, test) == test$shares)

pred2tr <- attr(predict(svm02, train, probability = TRUE), "probabilities")
pred2tra <- pred2tr[,1]
ROCRpred2 <- ROCR::prediction(pred2tra, train$shares)
ROCRperf2 <- performance(ROCRpred2, 'tpr','fpr')
png(filename="SVM_poly_TrainROC")
plot(ROCRperf2, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

pred2te <- attr(predict(svm02,test, probability = TRUE), "probabilities")
pred2tea <- pred2te[,1]
ROCRpredt2 <- ROCR::prediction(pred2tea, test$shares)
ROCRperft2 <- performance(ROCRpredt2, 'tpr','fpr')
png(filename="SVM_poly_TestROC")
plot(ROCRperft2, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

#svm with radial kernel
set.seed(176)
t0 <- Sys.time()
#svm03 <- svm(shares ~. , kernel="radial", data = train, probability = TRUE)
svm03 <- parallelSVM(shares ~. , cross=10, scale=FALSE, kernel="radial", data = train, probability = TRUE)
time3 <- difftime(Sys.time(),t0, units="secs")
err_tr3 <- 1-mean(predict(svm03, train) == train$shares)
err_te3 <- 1-mean(predict(svm03, test) == test$shares)

pred3tr <- attr(predict(svm03, train, probability = TRUE), "probabilities")
pred3tra <- pred3tr[,1]
png(filename="SVM_radial_TrainROC")
ROCRpred3 <- ROCR::prediction(pred3tra, train$shares)
ROCRperf3 <- performance(ROCRpred3, 'tpr','fpr')
plot(ROCRperf3, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

pred3te <- attr(predict(svm03,test, probability = TRUE), "probabilities")
pred3tea <- pred3te[,1]
ROCRpredt3 <- ROCR::prediction(pred3tea, test$shares)
ROCRperft3 <- performance(ROCRpredt3, 'tpr','fpr')
png(filename="SVM_radial_TestROC")
plot(ROCRperft3, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

acc <- as.data.frame(matrix(ncol=4,nrow=3))
names(acc) <- c("Kernel","trainError", "testError","seconds")
acc[,1] <- c("linear","Polynomial","Radial")
acc[1,2] <- err_tr1
acc[2,2] <- err_tr2
acc[3,2] <- err_tr3
acc[1,3] <- err_te1
acc[2,3] <- err_te2
acc[3,3] <- err_te3
acc[1,4] <- time1
acc[2,4] <- time2
acc[3,4] <- time3
acc

ggplot(data = acc, aes(x=Kernel, y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("Kernel")
ggplot(data = acc, aes(x=Kernel, y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("Kernel")
ggplot(data = acc, aes(x=Kernel, y=seconds)) + coord_cartesian(ylim = c(0, 1000)) + 
  geom_col(fill="darkred") + ylab("time taken in seconds to converge") + xlab("Kernel")

set.seed(176)
t0 <- Sys.time()
svm04 <- parallelSVM(shares ~. , cross=10, scale=FALSE, kernel="radial", data = train2, probability = TRUE)
time4 <- difftime(Sys.time(),t0, units="secs")
err_tr4 <- 1-mean(predict(svm04, train2) == train2$shares)
err_te4 <- 1-mean(predict(svm04, test2) == test2$shares)

set.seed(176)
t0 <- Sys.time()
svm05 <- parallelSVM(shares ~. , cross=10, scale=FALSE, kernel="radial", data = train3, probability = TRUE)
time5 <- difftime(Sys.time(),t0, units="secs")
err_tr5 <- 1-mean(predict(svm05, train3) == train3$shares)
err_te5 <- 1-mean(predict(svm05, test3) == test3$shares)

acc_svm <- as.data.frame(matrix(ncol=4,nrow=3))
names(acc_svm) <- c("TrainSetPct","trainError", "testError","seconds")
acc_svm[,1] <- c("70%","50%","30%")
acc_svm[1,2] <- err_tr3
acc_svm[2,2] <- err_tr4
acc_svm[3,2] <- err_tr5
acc_svm[1,3] <- err_te3
acc_svm[2,3] <- err_te4
acc_svm[3,3] <- err_te5
acc_svm[1,4] <- time3
acc_svm[2,4] <- time4
acc_svm[3,4] <- time5
acc_svm

#confusionMatrix(predict(svm05, test3), test3$shares)

ggplot(data = acc_svm, aes(x=TrainSetPct, y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("TrainSetPct")
ggplot(data = acc_svm, aes(x=TrainSetPct, y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("TrainSetPct")
ggplot(data = acc_svm, aes(x=TrainSetPct, y=seconds)) + coord_cartesian(ylim = c(0, 650)) + 
  geom_col(fill="darkred") + ylab("time taken in seconds to converge") + xlab("TrainSetPct")

##########################
# Classification Trees

#Multicore processing for Caret
cl <- makePSOCKcluster(4)
doParallel::registerDoParallel(cl, 4)

t0 <- Sys.time()
tree01 <- caret::train(shares~. , data=train, method = "rpart",
                       control = rpart.control(maxdepth = 10))
tree01_time <- Sys.time() - t0
plot(tree01)

tree01_error_train <- 1-mean(predict(tree01, train) == train$shares)
tree01_error_test <- 1-mean(predict(tree01, test) == test$shares)

tree01_prob_train <-  caret::predict.train(tree01, train, type= "prob")
tree01_prob_train <-  tree01_prob_train[,2]
tree01_ROCRpred_train <- ROCR::prediction(tree01_prob_train, train$shares)
tree01_ROCRperf_train <- performance(tree01_ROCRpred_train, 'tpr','fpr')
png(filename="tree01_TrainROC")
plot(tree01_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

tree01_prob_test <-  caret::predict.train(tree01, test, type= "prob")
tree01_prob_test <-  tree01_prob_test[,2]
tree01_ROCRpred_test <- ROCR::prediction(tree01_prob_test, test$shares)
tree01_ROCRperf_test <- performance(tree01_ROCRpred_test, 'tpr','fpr')
png(filename="tree01_TestROC")
plot(tree01_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

t0 <- Sys.time()
tree02 <- caret::train(shares~. , data=train, method = "rpart",
                       control = rpart.control(maxdepth = 7))
tree02_time <- Sys.time() - t0

tree02_error_train <- 1-mean(predict(tree02, train) == train$shares)
tree02_error_test <- 1-mean(predict(tree02, test) == test$shares)

tree02_prob_train <-  caret::predict.train(tree02, train, type= "prob")
tree02_prob_train <-  tree02_prob_train[,2]
tree02_ROCRpred_train <- ROCR::prediction(tree02_prob_train, train$shares)
tree02_ROCRperf_train <- performance(tree02_ROCRpred_train, 'tpr','fpr')
png(filename="tree02_TrainROC")
plot(tree02_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

tree02_prob_test <-  caret::predict.train(tree02, test, type= "prob")
tree02_prob_test <-  tree02_prob_test[,2]
tree02_ROCRpred_test <- ROCR::prediction(tree02_prob_test, test$shares)
tree02_ROCRperf_test <- performance(tree02_ROCRpred_test, 'tpr','fpr')
png(filename="tree02_TestROC")
plot(tree02_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

t0 <- Sys.time()
tree03 <- caret::train(shares~. , data=train, method = "rpart",
                       control = rpart.control(maxdepth = 5))
tree03_time <- Sys.time() - t0

tree03_error_train <- 1-mean(predict(tree03, train) == train$shares)
tree03_error_test <- 1-mean(predict(tree03, test) == test$shares)

tree03_prob_train <-  caret::predict.train(tree03, train, type= "prob")
tree03_prob_train <-  tree03_prob_train[,2]
tree03_ROCRpred_train <- ROCR::prediction(tree03_prob_train, train$shares)
tree03_ROCRperf_train <- performance(tree03_ROCRpred_train, 'tpr','fpr')
png(filename="tree03_TrainROC")
plot(tree03_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

tree03_prob_test <-  caret::predict.train(tree03, test, type= "prob")
tree03_prob_test <-  tree03_prob_test[,2]
tree03_ROCRpred_test <- ROCR::prediction(tree03_prob_test, test$shares)
tree03_ROCRperf_test <- performance(tree03_ROCRpred_test, 'tpr','fpr')
png(filename="tree03_TestROC")
plot(tree03_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

t0 <- Sys.time()
tree04 <- caret::train(shares~. , data=train, method = "rpart",
                       control = rpart.control(maxdepth = 3))
tree04_time <- Sys.time() - t0

tree04_error_train <- 1-mean(predict(tree04, train) == train$shares)
tree04_error_test <- 1-mean(predict(tree04, test) == test$shares)

tree04_prob_train <-  caret::predict.train(tree04, train, type= "prob")
tree04_prob_train <-  tree04_prob_train[,2]
tree04_ROCRpred_train <- ROCR::prediction(tree04_prob_train, train$shares)
tree04_ROCRperf_train <- performance(tree04_ROCRpred_train, 'tpr','fpr')
png(filename="tree04_TrainROC")
plot(tree04_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

tree04_prob_test <-  caret::predict.train(tree04, test, type= "prob")
tree04_prob_test <-  tree04_prob_test[,2]
tree04_ROCRpred_test <- ROCR::prediction(tree04_prob_test, test$shares)
tree04_ROCRperf_test <- performance(tree04_ROCRpred_test, 'tpr','fpr')
png(filename="tree04_TestROC")
plot(tree04_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

tree_acc <- as.data.frame(matrix(ncol=4,nrow=4))
names(tree_acc) <- c("Depth","trainError", "testError","Seconds")
tree_acc[,1] <- c(10,7,5,3)
tree_acc[1,2] <- tree01_error_train
tree_acc[2,2] <- tree02_error_train
tree_acc[3,2] <- tree03_error_train
tree_acc[4,2] <- tree04_error_train
tree_acc[1,3] <- tree01_error_test
tree_acc[2,3] <- tree02_error_test
tree_acc[3,3] <- tree03_error_test
tree_acc[4,3] <- tree04_error_test
tree_acc[1,4] <- tree01_time
tree_acc[2,4] <- tree02_time
tree_acc[3,4] <- tree03_time
tree_acc[4,4] <- tree04_time
tree_acc

ggplot(data = tree_acc, aes(x=as.factor(Depth), y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("Tree Depth")
ggplot(data = tree_acc, aes(x=as.factor(Depth), y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("Tree Depth")
ggplot(data = tree_acc, aes(x=as.factor(Depth), y=Seconds)) + coord_cartesian(ylim = c(0, 30)) + 
  geom_col(fill="darkred") + ylab("time taken to converge") + xlab("Tree Depth")

t0 <- Sys.time()
tree05 <- caret::train(shares~. , data=train2, method = "rpart",
                       control = rpart.control(maxdepth = 5))
tree05_time <- Sys.time() - t0

tree05_error_train <- 1-mean(predict(tree05, train2) == train2$shares)
tree05_error_test <- 1-mean(predict(tree05, test2) == test2$shares)

t0 <- Sys.time()
tree06 <- caret::train(shares~. , data=train3, method = "rpart",
                       control = rpart.control(maxdepth = 5))
tree06_time <- Sys.time() - t0

tree06_error_train <- 1-mean(predict(tree06, train3) == train3$shares)
tree06_error_test <- 1-mean(predict(tree06, test3) == test3$shares)

tree_acc2 <- as.data.frame(matrix(ncol=4,nrow=3))
names(tree_acc2) <- c("TrainSetPct","trainError", "testError","Seconds")
tree_acc2[,1] <- c("70%","50%","30%")
tree_acc2[1,2] <- tree03_error_train
tree_acc2[2,2] <- tree05_error_train
tree_acc2[3,2] <- tree06_error_train
tree_acc2[1,3] <- tree03_error_test
tree_acc2[2,3] <- tree05_error_test
tree_acc2[3,3] <- tree06_error_test
tree_acc2[1,4] <- tree03_time
tree_acc2[2,4] <- tree05_time
tree_acc2[3,4] <- tree06_time
tree_acc2

ggplot(data = tree_acc2, aes(x=as.factor(TrainSetPct), y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("Percent data used in train set")
ggplot(data = tree_acc2, aes(x=as.factor(TrainSetPct), y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("Percent data used in train set")
ggplot(data = tree_acc2, aes(x=as.factor(TrainSetPct), y=Seconds)) + coord_cartesian(ylim = c(0, 20)) + 
  geom_col(fill="darkred") + ylab("time taken to converge") + xlab("Percent data used in train set")

tree00 <- caret::train(shares~. , data=train, method = "rpart")
tree00

tree00 <- caret::train(shares~. , data=train, method = "rpart", tuneGrid= expand.grid (cp = c(0.001,0.0035,0.2,0.5,1)))
tree00
ggplot(data=tree00$results, aes(x=as.factor(tree00$results$cp), y=tree00$results$Accuracy)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("cross validation error") + xlab("TrainSetPct")

##########################
# BOOSTED TREES

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst01 <- train(shares~. , data=train,  
                method = 'AdaBoost.M1', 
                tuneGrid = expand.grid(maxdepth=10,mfinal=10, coeflearn = "Breiman"),
                trCtrl=treecontrol)
bst01_time <- difftime(Sys.time(),t0, units="secs")

(bst01_error_train <- 1-mean(predict(bst01, train) == train$shares))
(bst01_error_test <- 1-mean(predict(bst01, test) == test$shares))

bst01_prob_train <-  caret::predict.train(bst01, train, type= "prob")
bst01_prob_train <-  bst01_prob_train[,2]
bst01_ROCRpred_train <- ROCR::prediction(bst01_prob_train, train$shares)
bst01_ROCRperf_train <- performance(bst01_ROCRpred_train, 'tpr','fpr')
png(filename="tree01_TrainROC")
plot(bst01_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

bst01_prob_test <-  caret::predict.train(bst01, test, type= "prob")
bst01_prob_test <-  bst01_prob_test[,2]
bst01_ROCRpred_test <- ROCR::prediction(bst01_prob_test, test$shares)
bst01_ROCRperf_test <- performance(bst01_ROCRpred_test, 'tpr','fpr')
png(filename="tree01_TestROC")
plot(bst01_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst02 <- train(shares~. , data=train,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=7, mfinal=10, coeflearn = "Breiman"),
               trCtrl=treecontrol)
bst02_time <- difftime(Sys.time(),t0, units="secs")

(bst02_error_train <- 1-mean(predict(bst02, train) == train$shares))
(bst02_error_test <- 1-mean(predict(bst02, test) == test$shares))

bst02_prob_train <-  caret::predict.train(bst02, train, type= "prob")
bst02_prob_train <-  bst02_prob_train[,2]
bst02_ROCRpred_train <- ROCR::prediction(bst02_prob_train, train$shares)
bst02_ROCRperf_train <- performance(bst02_ROCRpred_train, 'tpr','fpr')
png(filename="tree01_TrainROC")
plot(bst02_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

bst02_prob_test <-  caret::predict.train(bst02, test, type= "prob")
bst02_prob_test <-  bst02_prob_test[,2]
bst02_ROCRpred_test <- ROCR::prediction(bst02_prob_test, test$shares)
bst02_ROCRperf_test <- performance(bst02_ROCRpred_test, 'tpr','fpr')
png(filename="tree01_TestROC")
plot(bst02_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst03 <- train(shares~. , data=train,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=5, mfinal=10, coeflearn = "Breiman"),
               trCtrl=treecontrol)
bst03_time <- difftime(Sys.time(),t0, units="secs")

(bst03_error_train <- 1-mean(predict(bst03, train) == train$shares))
(bst03_error_test <- 1-mean(predict(bst03, test) == test$shares))

bst03_prob_train <-  caret::predict.train(bst03, train, type= "prob")
bst03_prob_train <-  bst03_prob_train[,2]
bst03_ROCRpred_train <- ROCR::prediction(bst03_prob_train, train$shares)
bst03_ROCRperf_train <- performance(bst03_ROCRpred_train, 'tpr','fpr')
png(filename="bst03_TrainROC")
plot(bst03_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

bst03_prob_test <-  caret::predict.train(bst03, test, type= "prob")
bst03_prob_test <-  bst03_prob_test[,2]
bst03_ROCRpred_test <- ROCR::prediction(bst03_prob_test, test$shares)
bst03_ROCRperf_test <- performance(bst03_ROCRpred_test, 'tpr','fpr')
png(filename="bst03_TestROC")
plot(bst03_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst04 <- train(shares~. , data=train,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=3, mfinal=10, coeflearn = "Breiman"),
               trCtrl=treecontrol)
bst04_time <- difftime(Sys.time(),t0, units="secs")

(bst04_error_train <- 1-mean(predict(bst04, train) == train$shares))
(bst04_error_test <- 1-mean(predict(bst04, test) == test$shares))

bst04_prob_train <-  caret::predict.train(bst04, train, type= "prob")
bst04_prob_train <-  bst04_prob_train[,2]
bst04_ROCRpred_train <- ROCR::prediction(bst04_prob_train, train$shares)
bst04_ROCRperf_train <- performance(bst04_ROCRpred_train, 'tpr','fpr')
png(filename="bst04_TrainROC")
plot(bst04_ROCRperf_train, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Train ROC")
dev.off()

bst04_prob_test <-  caret::predict.train(bst04, test, type= "prob")
bst04_prob_test <-  bst04_prob_test[,2]
bst04_ROCRpred_test <- ROCR::prediction(bst04_prob_test, test$shares)
bst04_ROCRperf_test <- performance(bst04_ROCRpred_test, 'tpr','fpr')
png(filename="bst04_TestROC")
plot(bst04_ROCRperf_test, colorize = TRUE, text.adj = c(-0.2,1.7)) + title("Test ROC")
dev.off()

bst_acc <- as.data.frame(matrix(ncol=4,nrow=4))
names(bst_acc) <- c("Depth","trainError", "testError","Seconds")
bst_acc[,1] <- c(10,7,5,3)
bst_acc[1,2] <- bst01_error_train
bst_acc[2,2] <- bst02_error_train
bst_acc[3,2] <- bst03_error_train
bst_acc[4,2] <- bst04_error_train
bst_acc[1,3] <- bst01_error_test
bst_acc[2,3] <- bst02_error_test
bst_acc[3,3] <- bst03_error_test
bst_acc[4,3] <- bst04_error_test
bst_acc[1,4] <- bst01_time
bst_acc[2,4] <- bst02_time
bst_acc[3,4] <- bst03_time
bst_acc[4,4] <- bst04_time
bst_acc

ggplot(data = bst_acc, aes(x=as.factor(Depth), y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("Tree Depth")
ggplot(data = bst_acc, aes(x=as.factor(Depth), y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("Tree Depth")
ggplot(data = bst_acc, aes(x=as.factor(Depth), y=Seconds)) + coord_cartesian(ylim = c(0, 600)) + 
  geom_col(fill="darkred") + ylab("time taken to converge") + xlab("Tree Depth")

comp_acc <- as.data.frame(matrix(ncol=4,nrow=3))
names(comp_acc) <- c("Model","trainError", "testError","Seconds")
comp_acc[1,] <- acc[3,]
comp_acc[2,] <- tree_acc[3,]
comp_acc[3,] <- bst_acc[3,]
comp_acc[,1] <- c("RadialSVM","Tree","BoostedTree")
comp_acc

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst05 <- train(shares~. , data=train,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=5, mfinal=5, coeflearn = "Breiman"),
               trCtrl=treecontrol)
(bst05_time <- difftime(Sys.time(),t0, units="secs"))

(bst05_error_train <- 1-mean(predict(bst05, train) == train$shares))
(bst05_error_test <- 1-mean(predict(bst05, test) == test$shares))

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst06 <- train(shares~. , data=train,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=5, mfinal=15, coeflearn = "Breiman"),
               trCtrl=treecontrol)
(bst06_time <- difftime(Sys.time(),t0, units="secs"))

(bst06_error_train <- 1-mean(predict(bst06, train) == train$shares))
(bst06_error_test <- 1-mean(predict(bst06, test) == test$shares))

bst_acc2 <- as.data.frame(matrix(ncol=4,nrow=3))
names(bst_acc2) <- c("Iterations","trainError", "testError","Seconds")
bst_acc2[,1] <- c(10,5,15)
bst_acc2[1,2] <- bst03_error_train
bst_acc2[2,2] <- bst05_error_train
bst_acc2[3,2] <- bst06_error_train
bst_acc2[1,3] <- bst03_error_test
bst_acc2[2,3] <- bst05_error_test
bst_acc2[3,3] <- bst06_error_test
bst_acc2[1,4] <- bst03_time
bst_acc2[2,4] <- bst05_time
bst_acc2[3,4] <- bst06_time
bst_acc2

ggplot(data = bst_acc2, aes(x=as.factor(Iterations), y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("Iterations")
ggplot(data = bst_acc2, aes(x=as.factor(Iterations), y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("Iterations")
ggplot(data = bst_acc2, aes(x=as.factor(Iterations), y=Seconds)) + coord_cartesian(ylim = c(0, 600)) + 
  geom_col(fill="darkred") + ylab("time taken to converge") + xlab("Iterations")

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst07 <- train(shares~. , data=train2,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=5, mfinal=10, coeflearn = "Breiman"),
               trCtrl=treecontrol)
bst07_time <- difftime(Sys.time(),t0, units="secs")

(bst07_error_train <- 1-mean(predict(bst07, train2) == train2$shares))
(bst07_error_test <- 1-mean(predict(bst07, test2) == test2$shares))

set.seed(176)
treecontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
t0 <- Sys.time()
bst08 <- train(shares~. , data=train3,  
               method = 'AdaBoost.M1', 
               tuneGrid = expand.grid(maxdepth=5, mfinal=10, coeflearn = "Breiman"),
               trCtrl=treecontrol)
bst08_time <- difftime(Sys.time(),t0, units="secs")

(bst08_error_train <- 1-mean(predict(bst08, train3) == train3$shares))
(bst08_error_test <- 1-mean(predict(bst08, test3) == test3$shares))

bst_acc3 <- as.data.frame(matrix(ncol=4,nrow=3))
names(bst_acc3) <- c("TrainSetPct","trainError", "testError","Seconds")
bst_acc3[,1] <- c("70%","50%","30%")
bst_acc3[1,2] <- bst03_error_train
bst_acc3[2,2] <- bst07_error_train
bst_acc3[3,2] <- bst08_error_train
bst_acc3[1,3] <- bst03_error_test
bst_acc3[2,3] <- bst07_error_test
bst_acc3[3,3] <- bst08_error_test
bst_acc3[1,4] <- bst03_time
bst_acc3[2,4] <- bst07_time
bst_acc3[3,4] <- bst08_time
bst_acc3

ggplot(data = bst_acc3, aes(x=as.factor(TrainSetPct), y=trainError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("train set Classification error") + xlab("Percent data used in train set")
ggplot(data = bst_acc3, aes(x=as.factor(TrainSetPct), y=testError)) + coord_cartesian(ylim = c(0, 1)) + 
  geom_col(fill="darkred") + ylab("test set Classification error") + xlab("Percent data used in train set")
ggplot(data = bst_acc3, aes(x=as.factor(TrainSetPct), y=Seconds)) + coord_cartesian(ylim = c(0, 600)) + 
  geom_col(fill="darkred") + ylab("time taken to converge") + xlab("Percent data used in train set")

save.image("D:/Study/OneDrive - The University of Texas at Dallas/Semester 2/BUAN 6341 Applied Machine Learning/HW/02/03.RData")
