
# Business Understanding: 
# Here the problem is to correctly recognize handwritten digits. 
# Data  for hand written imagesis given in terms of pixel for both test and train data.
# Both data set has first column as the target variable &  rest are the pixel data attributes
# It has 1 categorical variable and 784 numerical attributes


############################ Steps carried out in code ##################
# Data Understanding and EDA
# Data Preparation
# Model Building 
#    Linear SVM Model at C=1
#    Linear SVM Model at C=10
#    Non Linear model - SVM
#   Cross validation 
#     Tuning linear SVM model 
#    Tuning Non-linear SVM model 
#  Conclusion/Summary 
#####################################################################################

##Loading Neccessary libraries

install.packages("kernlab")
install.packages("readr")
install.packages("caret")
install.packages("caTools")
install.packages("e1071")
install.packages("ggplot2")
install.packages("gridExtra")

library(kernlab)
library(readr)
library(caret)
library(caTools)
library(e1071)
library(ggplot2)
library(gridExtra)

#Loading Data
mnist_train <- read.csv("mnist_train.csv", header = F, stringsAsFactors = F)
mnist_test <- read.csv("mnist_test.csv", header = F, stringsAsFactors = F)

#Understanding Dimensions
dim(mnist_train) #60000 observations and 784 variables

#Structure of the dataset
str(mnist_train) #10000 observations and 784 variables

#printing first few rows
head(mnist_train)

#Exploring the data
summary(mnist_train)

#Check for duplicate records
nrow(unique(mnist_train))  #  no duplicate records in training data
nrow(unique(mnist_test))  #  no duplicate observations in test data 

#NA values in the train dataset
sum(is.na(mnist_train)) # Zero null values

#NA values in test dataset
sum(is.na(mnist_test))  # Zero null values

#EDA - Exploratory Data analysis

# Check the training data and digits available
names(mnist_train)[1] <- "label"
names(mnist_test)[1] <- "label"
plottrain <- ggplot(data = mnist_train, aes(x=mnist_train$label, fill = mnist_train$label)) + geom_bar()
# from plot- number of observations in each category is almost same
print(plottrain)


# Check the test data and digits available
plottest <- ggplot(data = mnist_test, aes(x=label, fill = label)) + geom_bar()
print(plottest)

#checking the intensity of each digit to cross check exact recognition

# plotting the distribution of each digits if it is normal distribution or not
number1 <- qplot(subset(check_mnist_train, label ==1)$intensity, binwidth = .75, 
                 xlab = "Intensity of 1")
number2 <- qplot(subset(check_mnist_train, label ==2)$intensity, binwidth = .75, 
                 xlab = "Intensity of 2")
number3 <- qplot(subset(check_mnist_train, label ==3)$intensity, binwidth = .75, 
                 xlab = "Intensity of 3")
number4 <- qplot(subset(check_mnist_train, label ==4)$intensity, binwidth = .75, 
                 xlab = "Intensity of 4")
number5 <- qplot(subset(check_mnist_train, label ==5)$intensity, binwidth = .75, 
                 xlab = "Intensity of 5")
number6 <- qplot(subset(check_mnist_train, label ==6)$intensity, binwidth = .75,
                 xlab = "Intensity of 6")
number7 <- qplot(subset(check_mnist_train, label ==7)$intensity, binwidth = .75,
                 xlab = "Intensity of 7")
number8 <- qplot(subset(check_mnist_train, label ==8)$intensity, binwidth = .75,
                 xlab = "Intensity of 8")
number9 <- qplot(subset(check_mnist_train, label ==9)$intensity, binwidth = .75,
                 xlab = "Intensity of 9")

grid.arrange(number1, number2, number3,number4,number5,number6,number7,number8,number9, ncol = 3)

#distributions for 4 and 7 are less "normal" than the distrubution for 1 
#indicating  that two people write 4 in two ways

check_mnist_train <- mnist_train
check_mnist_train$intensity <- apply(check_mnist_train[,-1], 1, mean) #takes the mean of each row in train
intlabel <- aggregate (check_mnist_train$intensity, by = list(check_mnist_train$label), FUN = mean)
plot <- ggplot(data=intlabel, aes(x=Group.1, y = x)) +
  geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + 
  ylab("average intensity")

#digit "1" is less intense while digit "0" is the most intense.

#####################################################################################
# Data Preparation

# Changing output variable "label" to factor type 
label <- as.factor(mnist_train[[1]]) 
test_label <- as.factor(mnist_test[[1]]) 
dftrain.pixel <- mnist_train[,2:ncol(mnist_train)]
dftest.pixel <- mnist_test[,2:ncol(mnist_test)]

# Attributes/variables are 784 in number, so applying the dimensionality/attributes reduction 
# Reduce the data set using PCA - principal component analysis of factor rotations in factor analysis.
#the possible range of pixel is  0 to 255, so dividing the data set by 255
# brings the value of each pixel value to the range of [0-1] which is normalized scale.

scaletrain_reduced <- dftrain.pixel/255
scaletrain_cov <- cov(scaletrain_reduced)
scaletrain_pca<-prcomp(scaletrain_cov)

# tranforming the dataset/ applying PCA to normalized-raw data
#  number of optimum attribtes to be considered will come from plot of
# standard deviations data in two dimensions
#selecting 60 because there is a sharp bend at this point 
# indicating beyond this points less significant attributes is present

plot1 <- plot(scaletrain_pca$sdev)
print(plot1)
scaletrain_final <-as.matrix(scaletrain_reduced) %*% scaletrain_pca$rotation[,1:60]


# Applying PCA to test data
scaletest_reduced <- dftest.pixel/255
scaletest_final <-as.matrix(scaletest_reduced) %*% scaletrain_pca$rotation[,1:60]

#plots on the reduced data after principal component analysis
plot1 <- plot(scaletrain_pca$sdev)
print(plot1)
plot2 <- plot(scaletrain_pca$x)
print(plot2)
plot3 <- plot(scale_pca$rotation)
print(plot3)
# From  plots, the distribution of the data set is ellipsoid in nature 

#####################################################################################
#  Model Building 
trainFinal <- cbind.data.frame(label, scaletrain_final)
names(trainFinal)[1] <- "label"
trainFinal$label <- as.factor(trainFinal$label)

testFinal <- cbind.data.frame(test_label, scaletest_final)
names(testFinal)[1] <- "label"
testFinal$label <- as.factor(testFinal$label)

# All the below models has been done with the entire data set and findings mentioned  in comments
#  same below model done with the sample data as well

# since the original train and test data contains 60000 and 10000 observations,
#for better performance of the model to run in min. time preparing the model with the sample data

#since the data set is huge, taking 30% of the data as sampling

indices_train = sample(1:nrow(trainFinal), 0.3*nrow(trainFinal))
indices_test = sample(1:nrow(testFinal), 0.3*nrow(testFinal))

trainFinal = trainFinal[indices_train,]
testFinal = testFinal[indices_test,]

#Note-
# Two  models created-
# 1.One with the entire data set with PCA 
# 2. Second with  the sampling data set with PCA 

#  Linear model - SVM  at Cost(C) = 1

# Model with C = 1
model_1<- ksvm(label ~ ., data = trainFinal,scale = FALSE, C=1)

# Predicting the model results 
result_1<- predict(model_1, testFinal[,-1])

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
conf_matrix_1 <- confusionMatrix(result_1, testFinal$label, positive = "Yes")

print(conf_matrix_1)



#print the output of confusion matrix of model_1
if(is.defined(conf_matrix_1)){
  printConfusionMatrixStats(conf_matrix_1)
}

#-----------------------------------------Linear model - SVM  at Cost(C) = 1-----------------------------------

# Note: Here Sensitivity and Specificity are average 


#*****Complete Dataset-(Train Data-60000, Test Data-10000)*****
# Accuracy                      - 0.979
# Sensitivity                   - 0.979 
# Specificity                   - 0.998

#*****Sample Dataset-(Train Data-18000, Test Data-3000)********
# Accuracy                      - 0.970
# Sensitivity                   - 0.970 
# Specificity                   - 0.996

#Linear model - SVM  at Cost(C) = 10

# Model with C = 10
model_2<- ksvm(label ~ ., data = trainFinal,scale = FALSE, C=10)

# Predicting the model results 
result_2<- predict(model_2, testFinal[,-1])

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
conf_matrix_2 <- confusionMatrix(result_2, testFinal$label, positive = "Yes")

#print the statistics of confusion matrix of model_2
print(conf_matrix_2)
if(is.defined(conf_matrix_2)){
  printConfusionMatrixStats(conf_matrix_2)
}



#-----------------------------------------Linear model - SVM  at Cost(C) = 10-----------------------------

#*****Complete Dataset-(Train Data-60000, Test Data-10000)*****
# Accuracy    - 0.984
# Sensitivity - 0.984
# Specificity - 0.998

#*****Sample Dataset-(Train Data-18000, Test Data-3000)********
# Accuracy                      - 0.974
# Sensitivity                   - 0.974 
# Specificity                   - 0.997

#  From this model with the 
#sample accuracy of 97% and C=10, the model is highly overfitting


##########################Non Linear model - SVM###########################################

# RBF kernel 
model_rbf <- ksvm(label ~ ., data =trainFinal,scale=FALSE, kernel = "rbfdot")


# Predicting the model results 
result_RBF<- predict(model_rbf, testFinal[,-1])

#confusion matrix - RBF Kernel
conf_matrix_model_rbf <- confusionMatrix(result_RBF,testFinal$label)

#print  confusion matrix
print(conf_matrix_model_rbf)
printConfusionMatrixStats(conf_matrix_model_rbf)



#-----------------------------------------Non Linear model - SVM -------------------------------------------------

#*****Complete Dataset-(Train Data-60000, Test Data-10000)*****
# Accuracy    - 0.979
# Sensitivity - 0.979
# Specificity - 0.997

#*****Sample Dataset-(Train Data-18000, Test Data-3000)********
# Accuracy                      - 0.970
# Sensitivity                   - 0.970 
# Specificity                   - 0.970

############################Cross validation#########################################################
  
##########################Hyperparameter tuning and Cross Validation - Linear - SVM ###########################################
# using the train function from caret package to perform crossvalidation

trainControl <- trainControl(method="cv", number=5)# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# Making grid of  C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

#  5-fold cross validation

cv.svm <- train(label~., data=trainFinal, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(cv.svm)

# Best tune  C=1, Accuracy - 0.929  (Complete Dataset)
# Best tune  C=1, Accuracy -  0.926 (sample Dataset) 

# Plotting model results
plot(cv.svm)

# checking the model after cross validation on test data

valid_linear_test<- predict(cv.svm, testFinal)
conf_matrix <- confusionMatrix(valid_linear_test, testFinal$label)


#print the o/p of confusion matrix 
print("Model using Hyperparameter tuning and Cross Validation - Linear - SVM ")
print(conf_matrix)
printConfusionMatrixStats(conf_matrix)


#-----------------------------------------Observation- Cross Validation - Linear - SVM ----------------------------------------


#*****Complete Dataset-(Train Data-60000, Test Data-10000)*****
# Accuracy    - 0.9358
# Sensitivity - 0.934723
# Specificity - 0.992872


#*****Sample Dataset-(Train Data-18000, Test Data-3000)********
# Accuracy                      - 0.931
# Sensitivity                   - 0.947 
# Specificity                   - 0.992
#------------------------------------------------------------------------------------------------------------------------------

########################Hyperparameter tuning and Cross Validation - Non-Linear - SVM#############################################
  
trainControl <- trainControl(method="cv", number=5)# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# Making grid of "sigma" and C values. 
grid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))


# 5-fold cross validation
cv.svm_radial <- train(label~., data=trainFinal, method="svmRadial", metric=metric, 
                        tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(cv.svm_radial)

#####################################################################
# Best tune at sigma = 0.03 & C=2, Accuracy - 0.985 (Complete Dataset)
# Best tune at sigma = 0.02 & C=3, Accuracy - 0.976(Sample Dataset) 

# Plotting model results
plot(cv.svm_radial)


############################ Checking overfitting - Non-Linear - SVM###########################################
# checking the model results on test data
valid_non_linear<- predict(cv.svm_radial, testFinal)
conf_matrix_non_linear <- confusionMatrix(valid_non_linear, testFinal$label)

print(conf_matrix_non_linear)
printConfusionMatrixStats(conf_matrix_non_linear)


#----------------------------------------- Cross Validation - Non Linear - SVM -----------------------------


#*****Complete Dataset-(Train Data-60000, Test Data-10000)*****
# Accuracy    - 0.985
# Sensitivity - 0.984
# Specificity - 0.998

#*****Sample Dataset-(Train Data-18000, Test Data-3000)********
# Accuracy                      - 0.976
# Sensitivity                   - 0.975 
# Specificity                   - 0.997


#########################Conclusion/Summary#############################################

#Two models prepared 
#1.Complete Dataset-(Train Data-60000, Test Data-10000)
#2.Sample Dataset-(Train Data-18000, Test Data-3000)
#one with entire dataset & another with sample (30% of complete)dataset.


#*****Complete Dataset-(Train Data-60000, Test Data-10000)*****
#-----------------------------------------------------------------------------------------------
#"Evaluation"	                 "Accuracy"	"Average Sensitivity"	"Average Specificity"		"C value"	
#------------------------------------------------------------------------------------------------
#linear model  with C = 1         0.979	      0.979	                0.998	                 1	  	
#linear model  with C = 10 	      0.984	      0.984	                0.998	               	 10	      	    	
#Model Non-Linear - SVM	          0.979	      0.979	                0.997	       	      	NA     	  	     	        
#Cross Validation Linear SVM    	0.936	      0.931	                0.941	              1 to 5	  	     	          
#Cross Validation Non-Linear SVM  0.985	      0.984	                0.998	              1 to 5        
#Best tune  C=1, Accuracy - 0.9296334(Linear)
#Best tune at sigma = 0.03 & C=2, Accuracy - 0.985(Non-Linear)
#-----------------------------------------------------------------------------------------
#For the complete dataset  it is confirmed that 
#the accuracy of non-linear model is better than linear model i.e. 0.985 > 0.936.
#The best tune model is  'SVM Non-linear with sigma = 0.03 & C=2, Accuracy - 0.985', 
#and it is also equal to the accuracy of test data which is 0.985  
#------------------------------------------------------------------------------------------------



#*****Sample Dataset-(Train Data-18000, Test Data-3000)********
# --------------------------------------------------------------------------------------------------
#"Evaluation"	               "Accuracy"	"Average Sensitivity"	"Average Specificity"		"C value"
#---------------------------------------------------------------------------------------------------
#linear model  with C = 1      0.970	      0.970	                0.997	               	   1		
#linear model  with C = 10 	   0.974	      0.974	                0.997	               	   10
#Non-Linear - SVM	             0.972	      0.979	                0.997	              	   NA       
#Cross ValidationLinear - SVM  0.936	      0.931	                0.992	                   1 to 5         
#Cross ValidationNon-LinearSVM  0.976	      0.975	                0.975	                   1 to 5        
#Best tune  C=1, Accuracy - 0.9296334(Linear)
#Best tune at sigma = 0.02 & C=3, Accuracy - 0.971(Non-Linear)	  
#---------------------------------------------------------------------------------------
# For the sample dataset  it is confirmed that
# the accuracy of non-linear model is better than linear model i.e. 0.976 > 0.936.
# The best tune model is SVM Non-linear with sigma = 0.02 & C=3, Accuracy - 0.971',
#and it is also equal to the accuracy of test data which is 0.975
#------------------------------------------------------------------------------------------------------------

# For the sample dataset  it is confirmed that-

# the accuracy of non-linear model is better than linear model i.e. 0.976 > 0.936.
# The best tune model is SVM Non-linear with sigma = 0.02 & C=3, 
#Accuracy - 0.971', and it is also equal to the accuracy of test data which is 0.975

# From both the above observations it is confirmed that-: 

#SVM Non-linear model with accuracy of 0.975  is the best.

#Criterion chosen for selecting sigma and C-:

#Here,  sigma = 0.02 & C=3, reason is that -This model is neither biased nor overfitted.
#Accuracy, average sensitivity and average specificity all are  equal to 0.975
# So the most optimum model is SVM Non-linear 
#with sigma = 0.02 & C=3, Accuracy - 0.971 approximated to 0.97 i.e. 97% 
# This is also confirmed with 0.976 accuracy with the test data i.e. 98%.
#It also signifies, model is very stable 
# Also non-linearity is very less which can be concluded from the plot - plot(scaletrain_pca$x).
#The polynomial is ellipsoid is nature.



