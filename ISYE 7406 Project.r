setwd("~/Documents/ISYE 7406/Project")
train_clean = read.csv("train_sample_big.csv", header=TRUE)
## Split to training and testing subset 
set.seed(123)
#use 80% of dataset as training set and 20% as test set
sample <- sample(c(TRUE, FALSE), nrow(train_clean), replace=TRUE, prob=c(0.8,0.2))
train  <- train_clean[sample, ]
test   <- train_clean[!sample, ]
## Extract the true response value for training and testing data
y1    <- train$country_destination
y2    <- test$country_destination

# AIC
reg0=glm(as.factor(country_destination)~1,data=train,family=binomial)
reg1=glm(as.factor(country_destination)~.,data=train,family=binomial)
step(reg0,scope=formula(reg1),
       direction="forward",k=2) 
# BIC - signup_method, first_affiliate_tracked, gender, language
step(reg0,scope=formula(reg1),
     direction="forward",k=log(41675))  
# leave one out CV
name_var=names(train)
 pred_i=function(i,k){
   fml = paste(name_var[8],"~",name_var[k],sep="")
   reg=glm(fml,data=train[-i,],family=binomial)
   predict(reg,newdata=train[i,],
             type="response")
 }
# ROC Curve
library(AUC)
ROC=function(k){
   Y=train[,8]=="Country"
   S=Vectorize(function(i) pred_i(i,k))
   (1:length(Y))
    R=roc(S,as.factor(Y))
    return(list(roc=cbind(R$fpr,R$tpr),
                              auc=AUC::auc(R)))
  }
# AUC
AUC=rep(NA,7)
for(k in 1:7){
    AUC[k]=ROC(k)$auc
    cat("Variable ",k,"(",name_var[k],") :",
            AUC[k],"\n") }

plot(0:1,0:1,col="white",xlab="",ylab="")
for(k in 1:7) 
 lines(ROC(k)$roc,type="s",col=CL[k])
legend(.8,.45,name_var,col=CL,lty=1,cex=.8)


library(randomForest)

## Build Random Forest with the default parameters
## It can be 'classification', 'regression', or 'unsupervised'
rf1 <- randomForest(as.factor(country_destination) ~., data=train, 
                    importance=TRUE)

## Check Important variables
importance(rf1)
## There are two types of importance measure 
##  (1=mean decrease in accuracy, 
##   2= mean decrease in node impurity)
importance(rf1, type=2)
varImpPlot(rf1)

## The plots show that first_affiliate_tracked, signup_flow, and 
## affiliate_channel are among the most important features when 
## predicting country_destination. 

## Prediction on the testing data set
rf.pred = predict(rf1, test, type='response')
table(rf.pred, y2)


##In practice, You can fine-tune parameters in Random Forest such as 
#ntree = number of tress to grow, and the default is 500. 
#mtry = number of variables randomly sampled as candidates at each split. 
#The default is sqrt(p) for classification and p/3 for regression
#nodesize = minimum size of terminal nodes. 
#The default value is 1 for classification and 5 for regression

rf2 <- randomForest(as.factor(country_destination) ~., data=train, 
                    ntree= 400, mtry=3, nodesize =2, importance=TRUE)

## The testing error of this new randomForest 
rf.pred2 = predict(rf2, test, type='response')
table(rf.pred2, y2)

## In general, we need to use a loop to try different parameter
## values (of ntree, mtry, etc.) to identify the right parameters 
## that minimize cross-validation errors.
library(caret)

# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- tuneRF(train[,1:11], train[,12], stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)

## (B) Boosting 
library(gbm)

gbm1 <- gbm(country_destination ~ .,data=train,
                 n.trees = 5000, 
                 shrinkage = 0.01, 
                 cv.folds = 10)

## Model Inspection 
## Find the estimated optimal number of iterations
perf_gbm1 = gbm.perf(gbm1, method="cv") 
perf_gbm1 # 4986

## summary model
## Which variances are important
summary(gbm1)

#Tuning
##############
library(caret)
trainControl <- trainControl(method = "cv",
                             number = 10,
                             returnResamp="all", ### use "all" to return all cross-validated metrics
                             search = "grid")

tuneGrid <- expand.grid(
  n.trees = c(5000, 10000),
  interaction.depth = c( 6, 13),
  shrinkage = c(0.01, 0.001),
  n.minobsinnode=c(5, 10)
)
gbm_op <- train(as.factor(country_destination) ~.,
                data = train,
                method = "gbm",
                tuneGrid = tuneGrid,
                trControl = trainControl,
                verbose=FALSE)
gbm_op
###########


## Make Prediction
## use "predict" to find the training or testing error

## Training error
pred1gbm <- predict(gbm1,newdata = train, n.trees=perf_gbm1, type="response")
pred1gbm[1:10]
y1hat <- ifelse(pred1gbm < y1 + 0.5 & pred1gbm > y1 - 0.5, 0, 1)
y1hat
sum(y1hat != 1)/length(y1hat)  ##Training error = 0.119198

## Testing Error
pred2gbm <- predict(gbm1,newdata = test, n.trees=perf_gbm1, type="response")
y2hat <- ifelse(pred2gbm < y1 + 0.5 & pred2gbm > y1 - 0.5, 0, 1)
y2hat
sum(y2hat != 1)/length(y2hat)
## Testing error = 0.1165995

## A comparison with other methods
## Testing errors of several algorithms on the auto dataset:
#A. Logistic regression: 1
modA <- step(glm(country_destination ~ ., data = train));
y2hatA <- predict(modA, test, type="response" )
sum(y2hatA != y2)/length(y2) 

library(VGAM)
mlr <- vglm(country_destination ~ ., multinomial(refLevel = 1), data = train)
summary(mlr)

# MLR
library(nnet)
mod1 <- multinom(country_destination ~ signup_method + first_affiliate_tracked + gender + language, data = train)
trainingerr <- mean(predict(mod1, train[, 1:11]) != train$country_destination)
testingerr <- mean(predict(mod1, test[,1:11]) != test$country_destination)
testingerr # 0.2884727

#B.Linear Discriminant Analysis : 0.2930046
library(MASS)
modB <- lda(country_destination ~ signup_method + first_affiliate_tracked + 
              gender + language, train)
y2hatB <- predict(modB, test)$class
mean( y2hatB  != y2)

## C. Naive Bayes (with full X). Testing error = 0.3126084
library(e1071)
modC <- naiveBayes(as.factor(country_destination) ~ signup_method + first_affiliate_tracked + 
                     gender + language, data = train)
y2hatC <- predict(modC, newdata = test)
mean( y2hatC != y2) 


#E: a single Tree: 0.2930046
library(rpart)
modE0 <- rpart(country_destination ~ .,data=train, method="class", 
               parms=list(split="gini"))
opt <- which.min(modE0$cptable[, "xerror"]); 
cp1 <- modE0$cptable[opt, "CP"];
modE <- prune(modE0,cp=cp1);
y2hatE <-  predict(modE, test,type="class")
mean(y2hatE != y2)

#F: Random Forest: 0.2932913
library(randomForest)
modF <- randomForest(as.factor(country_destination) ~., data=train, 
                     importance=TRUE)
y2hatF = predict(modF, test, type='response')
mean(y2hatF != y2)


## Testing error of KNN, and you can change the k values.
train <- subset(train, select = c("signup_method", "first_affiliate_tracked", "gender", "language", "country_destination"))
test <- subset(test, select = c("signup_method", "first_affiliate_tracked", "gender", "language", "country_destination"))
  
library(class)
xnew2 <- test[1: ncol(test)-1 ]     
xnkk <- 1;  
ypred2.test  <- knn(train[1: ncol(test)-1 ], xnew2, y1, k=kk);
mean( ypred2.test != y2) 
# 0.1019857

xnew2 <- test[,-1];      
kk <- 3;  
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2) 
# 0.1281087

xnew2 <- test[,-1];         
kk <- 5;   
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2)
# 0.1403509

xnew2 <- test[,-1];        
kk <- 7;     
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2)
# 0.1540389

xnew2 <- test[,-1];       
kk <- 9;  
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2) 
# 0.1652207

xnew2 <- test[,-1];         
kk <- 11;     
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2) 
# 0.170908

xnew2 <- test[,-1];        
kk <- 13;    
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2) 
# 0.1791016

xnew2 <- test[,-1];       
kk <- 15;    
ypred2.test  <- knn(train[,-1], xnew2, y1, k=kk);
mean( ypred2.test != y2) 
# 0.1845961

### Cross-Validation
dataset =  rbind(train,  test)      ### combine to a full data set
n1 = 10464;   # testing set sample size
#n2= 10464;     # testing set sample size
n = dim(dataset)[1];    ## the total sample size
set.seed(7406);   ### set the seed for randomization
###    Initialize the TE values for all models in all $B=100$ loops
B= 100;            ### number of loops
TEALL = NULL;      ### Final TE values
for (b in 1:B){
  ### randomly select n1 observations as a new training  subset in each loop
  flag <- sort(sample(1:n, n1));
  traintemp <- dataset[flag,];  ## temp training set for CV
  testtemp  <- dataset[-flag,]; ## temp testing set for CV
  
  # knn
  xnew3 <- testtemp[,-1];
  kk <- 1; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te1 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 3; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te2 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 5; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te3 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 7; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te4 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 9; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te5 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 11; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te6 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 13; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te7 <- mean( ypred3.test != testtemp[,12])
  
  xnew3 <- testtemp[,-1];
  kk <- 15; 
  ypred3.test  <- knn(traintemp[,-1], xnew3, traintemp[,12], k=kk);
  te8 <- mean( ypred3.test != testtemp[,12])
  
  TEALL = rbind( TEALL, cbind(te1, te2, te3, te4, te5, te6, te7, te8) );
}

dim(TEALL);  ### Bx9 matrices

colnames(TEALL) <- c("KNN1", "KNN3", "KNN5", "KNN7",
                     "KNN9", "KNN11", "KNN13", "KNN15");

## sample mean/variances of the testing errors so as to compare these models
apply(TEALL, 2, mean); #     KNN1      KNN3      KNN5      KNN7      KNN9     KNN11     KNN13     KNN15 
                      #     0.1565600 0.1854141 0.1981461 0.2053740 0.2107791 0.2150503 0.2184641 0.2215207 
apply(TEALL, 2, var);

### CV
set.seed(1);
B = 100;
TEALL=NULL;
for (b in 1:B){
  #sample <- sample(c(TRUE, FALSE), nrow(Auto), replace=TRUE, prob=c(0.8,0.2))
  #train  <- Auto[sample, ]
  #test   <- Auto[!sample, ]
  flag <- sort(sample(1:n, n1));
  train <- train_clean[-flag,];
  test <- train_clean[flag,];
  ytrue <- test$country_destination
  ###### Boosting
  #library(class)
  #gbm1 <- gbm(country_destination ~ .,data=train, n.trees = 5000, 
  #            shrinkage = 0.01, cv.folds = 10)
  #perf_gbm1 = gbm.perf(gbm1, method="cv")
  #pred2gbm <- predict(gbm1,newdata = test, n.trees=perf_gbm1, type="response")
  #y2hat <- ifelse(pred2gbm < y1 + 0.5 & pred2gbm > y1 - 0.5, 0, 1)
  #gbmTestErr <- sum(y2hat != 1)/length(y2hat)
  ###### LDA
  library(MASS)
  lda <- lda(country_destination ~ signup_method + first_affiliate_tracked + gender + language, train)
  pred1test <- predict(lda,test[,1:11])$class; 
  ldaTestErr <- mean(pred1test != test$country_destination)
  ####### QDA
  #qda <- qda(country_destination~., train)
  #pred2test <- predict(qda,test[,1:11])$class; 
  #qdaTestErr <- mean(pred2test != test$country_destination)
  ####### Naive Bayes
  library(naivebayes)
  nb <- naive_bayes(as.character(country_destination) ~ signup_method + first_affiliate_tracked + gender + language, train) 
  nbTestErr <- mean( predict(nb,test[,1:11]) != test$country_destination)
  ####### Logistic Regression
  #lr <- glm(country_destination~., binomial("logit"), train)
  #pred4test = ifelse(predict(lr, test[,1:11], type="response") >= 0.5, 1, 0)
  #lrTestErr <- mean( pred4test  != test$country_destination)
  ####### KNN
  #library(class)
  #kk <- 1
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn1TestErr <- mean(ypredtest != test[,1]) 
  
  #kk <- 3
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn3TestErr <- mean(ypredtest != test[,1]) 
  
  #kk <- 5
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn5TestErr <- mean(ypredtest != test[,1]) 
  
  #kk <- 7
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn7TestErr <- mean(ypredtest != test[,1]) 
  
  #kk <- 9
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn9TestErr <- mean(ypredtest != test[,1])
  
  #kk <- 11
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn11TestErr <- mean(ypredtest != test[,1])
  
  #kk <- 13
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn13TestErr <- mean(ypredtest != test[,1])
  
  #kk <- 15
  #xnew <- test[,-1]
  #ypredtest  <- knn(train[,-1], xnew, train[,1], k=kk);
  #knn15TestErr <- mean(ypredtest != test[,1])
  
  ### PCA KNN ###
  #library(caret)
  #responseY <- train[,1]
  #predictorX <- train[,2:8]
  # PCs from PCA
  #pca <- princomp(predictorX, cor=T) # principal components analysis using correlation matrix
  #pc.comp <- pca$scores
  #pc.comp1 <- -1*pc.comp[,1]
  #pc.comp2 <- -1*pc.comp[,2]
  
  # data partition for train/test sets.
  #trainIndex <- createDataPartition(responseY, times=1, p = 0.8, list = F)
  #X = cbind(pc.comp1, pc.comp2)
  
  # fitting models for 30 different k-values (one for test and one for train set for each K)
  #test.error = rep(0,15)
  #for(k in 1:15){
  #  model.knn.test <- knn(train=X[trainIndex,], test=X[-trainIndex,], cl=responseY[trainIndex], k=k, prob=F)
  #  test.error[k] <- sum(model.knn.test!=responseY[-trainIndex])/length(responseY[-trainIndex])
  #}
}
TEALL = rbind(TEALL, cbind(ldaTestErr, nbTestErr));
dim(TEALL);
round(apply(TEALL, 2, mean),4);
# gbmTestErr ldaTestErr qdaTestErr  nbTestErr 
# 0.1162     0.2916     0.3206     0.3147
# apply(TEALL, 2, mean)
# gbmTestErr ldaTestErr qdaTestErr  nbTestErr 
# 0.1161942  0.2915711  0.3206231  0.3146980 
round(apply(TEALL, 2, var),4);


