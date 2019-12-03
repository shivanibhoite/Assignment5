
library(caret)
library(gbm)
library('RANN')
#install.packages("klaR")
library('klaR')
#install.packages("ggpubr")
library(ggpubr)
data(scat)
str(scat)


########## 1 Set the Species column as the target/outcome and convert it to numeric. (5 points)
scat_2<-scat
scat_2$Species<-as.factor(scat_2$Species)
target<-scat_2$Species
target_class<-factor(target)
scat_2$Species<-unclass(scat_2$Species)
#Converted to numeric - printing the data below
print(scat_2)

########## 2 Remove the Month, Year, Site, Location features. (5 points)
#Before removing
head(scat)
scat_subset <- subset(scat, select = - c(Month, Year, Site, Location))
#After removing
head(scat_subset)



########## 3 Check if any values are null. If there are, impute missing values using KNN. (10 points) 
sum(is.na(scat_subset))
preProcValues <- preProcess(scat_subset, method = c("knnImpute","center","scale"))
scat_processed <- predict(preProcValues, scat_subset)
sum(is.na(scat_processed))
str(scat_processed)


########## 4 Converting every categorical variable to numerical (if needed). (5 points)
#Not needed


########## 5 With a seed of 100, 75% training, 25% testing. 
########### Build the following models: randomforest, neural net, naive bayes and GBM.
scat_processed$Species<-as.factor(scat_processed$Species)
#print(scat_processed)
#Building Models
#Spliting training set into two parts based on outcome: 75% and 25%
set.seed(100)
index <- createDataPartition(scat_processed$Species, p=0.75, list=FALSE)
str(index)
trainSet <- scat_processed[ index,]
testSet <- scat_processed[-index,]

#feature selection
control <- rfeControl(functions = rfFuncs,method = "repeatedcv", repeats = 3,verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
str(predictors)

Species_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
Species_Pred_Profile
#names(getModelInfo())


#Making models

#1)GBM
#As there are more than 2 categories for prediction in GBM the distribution has to be changed from bernoulli to multinomial
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',distribution='multinomial')

#2)Random Forest
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',importance=T)

#3) Neural Network
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',importance=T)

#4) Naive Bayes
model_nbayes<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')


###Model Summarization 
#1)GBM
print(model_gbm)

#2)Random Forest
print(model_rf)

#3) Neural Network
print(model_nnet)

#4) Naive Bayes
print(model_nbayes)

###Plot Variable Importance
#GBM
plot(varImp(object=model_gbm),main="GBM - Variable Importance")

#RF
plot(varImp(object=model_rf),main="RF - Variable Importance")


#NNET
#for ploting the variable importance of 

df1<-as.data.frame(varImp(object=model_nnet)$importance)
print(df1)
df2 = data.frame(name = c("d15N ","d13C","Mass","CN","Length","ropey","flat","Diameter","Number","Age","TI","segmented","Taper","scrape"))
cbinded_df<-cbind(df1,df2)

p<-ggplot(data=cbinded_df, aes(x=name, y=Overall)) +
  geom_bar(stat="identity")+ggtitle('Neural Net - Variable Importance')
nnet_var_imp<-p + coord_flip()
nnet_var_imp

#Naive Bayes
plot(varImp(object=model_nbayes),main="Naive Bayes - Variable Importance")

###Confusion Matrix
#GBM
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])


#RF
predictions<-predict.train(object=model_rf,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])


#Neural Network
predictions<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])


#Naive Bayes
predictions<-predict.train(object=model_nbayes,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])

#######End of Question 5


######### 6 For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) 
#create and display a data frame that has the following columns: ExperimentName, accuracy, kappa. 
#Sort the data frame by accuracy. (15 points)


gbm_df <- data.frame("Experiment" = 'GBM', "Accuracy" = model_gbm$results$Accuracy, "Kappa" = model_gbm$results$Kappa)
gbm_df <-gbm_df[order(-gbm_df$Accuracy),]

rf_df <- data.frame("Experiment" = 'Random Forest', "Accuracy" = model_rf$results$Accuracy, "Kappa" = model_rf$results$Kappa)
rf_df <-rf_df[order(-rf_df$Accuracy),]

nnet_df <- data.frame("Experiment" = 'Neural Network', "Accuracy" = model_nnet$results$Accuracy, "Kappa" = model_nnet$results$Kappa)
nnet_df <-nnet_df[order(-nnet_df$Accuracy),]

nb_df <- data.frame("Experiment" = 'Naive Bayes', "Accuracy" = model_nbayes$results$Accuracy, "Kappa" = model_nbayes$results$Kappa)
nb_df <-nb_df[order(-nb_df$Accuracy),]

total <- rbind(gbm_df[1,], rf_df[1,],nnet_df[1,],nb_df[1,])
total <-total[order(-total$Accuracy),]
print(total)


########## 7 Tune the GBM model using tune length = 20 and: 
#a) print the model summary and b) plot the models. (20 points)
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
model_gbm_tune_7<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
print(model_gbm_tune_7)
plot(model_gbm_tune_7)


########## 8 Using GGplot and gridExtra to plot all variable of importance plots into one single plot. (10 points)
#GBM
gbm_var_imp<-ggplot(varImp(object=model_gbm))+ggtitle('GBM - Variable Importance')

#RF
rf_var_imp<-ggplot(varImp(object=model_rf))+ggtitle('RF - Variable Importance')

#NNET
df1<-as.data.frame(varImp(object=model_nnet)$importance)
#print(df1)
df2 = data.frame(name = c("d15N ","d13C","Mass","CN","Length","ropey","flat","Diameter","Number","Age","TI","segmented","Taper","scrape"))
cbinded_df<-cbind(df1,df2)

p<-ggplot(data=cbinded_df, aes(x=name, y=Overall)) +
  geom_bar(stat="identity")+ggtitle('Neural Net - Variable Importance')
nnet_var_imp<-p + coord_flip()

#NB
nb_var_imp<-ggplot(varImp(object=model_nbayes))+ggtitle('Naive Bayes - Variable Importance')

#Combining the data
grid.arrange(gbm_var_imp, rf_var_imp,nb_var_imp,nnet_var_imp)

######### 9 Which model performs the best? and why do you think this is the case? 
#Can we accurately predict species on this dataset? (10 points)

print(total)

#The Neural Network performs the best with an accuracy of 70%.
#     Neural networks model is the best as it shows the ability to learn on non-linear relationships and complex
#     relationships like seen in the dataset we have.
#     Neural network here takes into consideration all the other features and builds a weighted relationship in between them
#     Hence this relationship helps in acheieving the highest accuracy
#     YES we can predict the species with this model


#################### Graduate Question
#Using feature selection with rfe in caret and the repeatedcv method: Find the top 3
#predictors and build the same models as in 6 and 8 with the same parameters. (20 points)

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Species_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
Species_Pred_Profile

#Taking only the top 3 predictors
predictors_top3<-c("CN", "d13C", "d15N")

# For example, to apply, GBM, Random forest, Neural net:
model_gbm_10<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='gbm',distribution='multinomial')
model_rf_10<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='rf', importance=T)
model_nnet_10<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='nnet', importance=T)
model_nbayes_10<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='naive_bayes',importance=T)


# Create a dataframe that compares the non-feature selected models ( the same as on 7)
# and add the best BEST performing models of each (randomforest, neural net, naive bayes and gbm) 
#and display the data frame that has the following columns: ExperimentName, accuracy, kappa. Sort the data frame by accuracy. (40 points)
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
#Making the non-feature selected models using all predictors
model_gbm_10_tune<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',distribution='multinomial',trControl=fitControl,tuneLength=20)
model_rf_10_tune<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf', importance=T,trControl=fitControl,tuneLength=20)
model_nnet_10_tune<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet', importance=T,trControl=fitControl,tuneLength=20)
model_nbayes_10_tune<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes',importance=T,trControl=fitControl,tuneLength=20)


#Making the non-feature selected models using top 3 predictors
model_gbm_10_tune_top3<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='gbm',distribution='multinomial',trControl=fitControl,tuneLength=20)
model_rf_10_tune_top3<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='rf', importance=T,trControl=fitControl,tuneLength=20)
model_nnet_10_tune_top3<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='nnet', importance=T,trControl=fitControl,tuneLength=20)
model_nbayes_10_tune_top3<-train(trainSet[,predictors_top3],trainSet[,outcomeName],method='naive_bayes',importance=T,trControl=fitControl,tuneLength=20)


#Making a dataframe

#For models using top 3 predictors
gbm_df_10 <- data.frame("Experiment" = 'GBM with top 3 Features', "Accuracy" = model_gbm_10$results$Accuracy, "Kappa" = model_gbm_10$results$Kappa)
gbm_df_10 <-gbm_df_10[order(-gbm_df_10$Accuracy),]

rf_df_10 <- data.frame("Experiment" = 'Random Forest with top 3 Features', "Accuracy" = model_rf_10$results$Accuracy, "Kappa" = model_rf_10$results$Kappa)
rf_df_10 <-rf_df_10[order(-rf_df_10$Accuracy),]

nnet_df_10 <- data.frame("Experiment" = 'Neural Network with top 3 Features', "Accuracy" = model_nnet_10$results$Accuracy, "Kappa" = model_nnet_10$results$Kappa)
nnet_df_10 <-nnet_df_10[order(-nnet_df_10$Accuracy),]

nb_df_10 <- data.frame("Experiment" = 'Naive Bayes with top 3 Features', "Accuracy" = model_nbayes_10$results$Accuracy, "Kappa" = model_nbayes_10$results$Kappa)
nb_df_10 <-nb_df_10[order(-nb_df_10$Accuracy),]

#For models using tuning for all features
gbm_df_10_tune <- data.frame("Experiment" = 'GBM with Tune for all features', "Accuracy" = model_gbm_10_tune$results$Accuracy, "Kappa" = model_gbm_10_tune$results$Kappa)
gbm_df_10_tune <-gbm_df_10_tune[order(-gbm_df_10_tune$Accuracy),]

rf_df_10_tune <- data.frame("Experiment" = 'Random Forest with Tune for all features', "Accuracy" = model_rf_10_tune$results$Accuracy, "Kappa" = model_rf_10_tune$results$Kappa)
rf_df_10_tune <-rf_df_10_tune[order(-rf_df_10_tune$Accuracy),]

nnet_df_10_tune <- data.frame("Experiment" = 'Neural Network with Tune for all features', "Accuracy" = model_nnet_10_tune$results$Accuracy, "Kappa" = model_nnet_10_tune$results$Kappa)
nnet_df_10_tune <-nnet_df_10_tune[order(-nnet_df_10_tune$Accuracy),]

nb_df_10_tune <- data.frame("Experiment" = 'Naive Bayes with Tune for all features', "Accuracy" = model_nbayes_10_tune$results$Accuracy, "Kappa" = model_nbayes_10_tune$results$Kappa)
nb_df_10_tune <-nb_df_10_tune[order(-nb_df_10_tune$Accuracy),]

#For models using tuning with top 3 features
gbm_df_10_tune_top3 <- data.frame("Experiment" = 'GBM with Tune for top 3 Features', "Accuracy" = model_gbm_10_tune_top3$results$Accuracy, "Kappa" = model_gbm_10_tune_top3$results$Kappa)
gbm_df_10_tune_top3 <-gbm_df_10_tune_top3[order(-gbm_df_10_tune_top3$Accuracy),]

rf_df_10_tune_top3 <- data.frame("Experiment" = 'Random Forest with Tune for top 3 Features', "Accuracy" = model_rf_10_tune_top3$results$Accuracy, "Kappa" = model_rf_10_tune_top3$results$Kappa)
rf_df_10_tune_top3 <-rf_df_10_tune_top3[order(-rf_df_10_tune_top3$Accuracy),]

nnet_df_10_tune_top3 <- data.frame("Experiment" = 'Neural Network with Tune for top 3 Features', "Accuracy" = model_nnet_10_tune_top3$results$Accuracy, "Kappa" = model_nnet_10_tune_top3$results$Kappa)
nnet_df_10_tune_top3 <-nnet_df_10_tune_top3[order(-nnet_df_10_tune_top3$Accuracy),]

nb_df_10_tune_top3 <- data.frame("Experiment" = 'Naive Bayes with Tune for top 3 Features', "Accuracy" = model_nbayes_10_tune_top3$results$Accuracy, "Kappa" = model_nbayes_10_tune_top3$results$Kappa)
nb_df_10_tune_top3 <-nb_df_10_tune_top3[order(-nb_df_10_tune_top3$Accuracy),]


total_10 <- rbind(gbm_df_10[1,], rf_df_10[1,],nnet_df_10[1,],nb_df_10[1,],gbm_df_10_tune[1,], rf_df_10_tune[1,],nnet_df_10_tune[1,],nb_df_10_tune[1,],gbm_df_10_tune_top3[1,],rf_df_10_tune_top3[1,],nnet_df_10_tune_top3[1,],nb_df_10_tune_top3[1,])
total_10 <-total_10[order(-total_10$Accuracy),]
print(total_10)


#c. Which model performs the best? and why do you think this is the case? 
#Can we accurately predict species on this dataset? (10 points)
#Ans---The Neural Network model with top 3 parameters and parameter tunning works the best acheieving 
#     upto accuracy of 77%.
#     Neural networks model is the best as it shows the ability to learn on non-linear relationships and complex
#     relationships like seen in the dataset we have.
#     Plus the neural network here is using the best of 3 features and tuning them. Hence the accuracy is higher than the previous.
#     Yes, We can predict the species with using this model.


