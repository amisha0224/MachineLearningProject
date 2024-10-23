#DATA SCIENCE CEE ASSESSMENT - MODEL BUILDING
#AMISHA Y [MS212408]

#Classification
rm(list = ls(all = T))
setwd("C:\\Users\\amish\\Downloads\\data science")

mydata = read.csv("student.csv")
head(mydata)
summary(mydata)

#Coding the target to 0 and 1 for graduate and dropout respectively
mydata$Target <- ifelse(mydata$Target == "Dropout", 1, 0)

library(corrplot)
matrix<- cor(mydata, method = "pearson")
View(matrix)
corrplot(matrix, method = "number")
mydata$Target=as.factor(mydata$Target)
str(mydata)
table(mydata$Target)


#Exploratory Data Analysis
install.packages("summarytools")

mydata$Gender = as.factor(mydata$Gender)
mydata$Scholarship_holder = as.factor(mydata$Scholarship_holder)
mydata$Tuition_fees_up_to_date = as.factor(mydata$Tuition_fees_up_to_date)
mydata$Marital_status = as.factor(mydata$Marital_status)
mydata$Daytime_evening_attendance = as.factor(mydata$Daytime_evening_attendance )
mydata$Debtor = as.factor(mydata$Debtor)

View(mydata)
library(summarytools)
view(dfSummary(mydata))


## breaking into train and test
set.seed(123)
rows = sample(nrow(mydata)*0.7, replace = F)
train = mydata[rows,]
test = mydata[-rows,]

## model
logmodel = glm(train$Target ~ ., data = train, family = "binomial")
logmodel
summary(logmodel)
library(car)
car::vif(logmodel)
#train data
train_pred = predict(logmodel, train, type = "response")
train_pred = ifelse(train_pred >=0.5,1,0)
table(train_pred)
head(train$Target, 10)
train_pred[1:10]

#precision, recall, f1-score
table(train$Target, train_pred)
#precision = true ones/predicted ones
train_precision = 1531/(1531+158)
train_precision
# recall = true ones/actual ones
train_recall = 1531/(1531+54)
train_recall
#f1-score = 2pr/(p+r)
f1_train = 2*train_precision*train_recall/(train_precision+train_recall)
f1_train

install.packages("ROCR")
library(ROCR)
pe=prediction(train_pred,train$Target)
perf=performance(pe,measure = "tpr", x.measure = "fpr")
plot(perf,main="AUC under ROC")


#test data
test_pred = predict(logmodel, test, type = "response")
test_pred = ifelse(test_pred >=0.5,1,0)
table(test_pred)
head(test$Target, 10)
test_pred[1:10]
#precision, recall, f1-score
table(test$Target, test_pred)
#precision = true ones/predicted ones
test_precision = 596/(596+73)
test_precision
# recall = true ones/actual ones
test_recall = 596/(596+28)
test_recall
#f1-score = 2pr/(p+r)
f1_test = 2*test_precision*test_recall/(test_precision+test_recall)
f1_test

library(ROCR)
pe=prediction(test_pred,test$Target)
perf=performance(pe,measure = "tpr", x.measure = "fpr")
plot(perf,main="AUC under ROC")

#Additional analysis on entire data
pred<-predict(logmodel,mydata,type = 'response')
pred_scale<-ifelse(pred >0.5,1,0)
df<-data.frame(pred,pred_scale)
View(df)

conf_matrix<-table(Actual=mydata$Target,Predicted= pred_scale > 0.5)
print(conf_matrix)
library(Metrics)
accuracy(mydata$Target,pred_scale)

library(ROCR)

pr<-prediction(pred_scale,mydata$Target)
#creating a prediction class
pref<- performance(pr,measure = "tpr",x.measure = "fpr")
auc(mydata$Target,pred_scale)
plot(pref,main="AUC under ROC")


"""
CONCLUSIONS

Exploratory Data Analysis gives the variables in distict forms and the type of data is analysed and 
graphical representation is also depicted here. Categorical variables are described as factors.

Significant variables:
Course                                         0.001860 ** 
Nacionality                                    0.000963 ***
Mother_s_qualification                         0.047555 *  
Mother_s_occupation                            0.019027 *  
Displaced                                      0.048245 *  
Debtor1                                        0.000102 ***
Tuition_fees_up_to_date1                       5.98e-15 ***
Scholarship_holder1                            0.000367 ***
Age_at_enrollment                              0.024920 *  
International                                  0.000428 ***
Curricular_units_1st_sem__approved_            2.36e-09 ***
Curricular_units_2nd_sem__enrolled_            1.05e-06 ***
Curricular_units_2nd_sem__approved_             < 2e-16 ***
Curricular_units_2nd_sem_grade                 0.007086 ** 
Unemployment_rate                              0.014948 *  

Other variables are not significant so they can be removed


INFERENCE ON TRAIN DATA
we observe that f1 score is 93% ,precision is 90% and recall is 96%  for the train data
f1-score: F1 score above 90% indicates a good balance between precision and recall.
Precision: Precision above 90% means that the vast majority of the model's positive predictions are correct.  
The model is highly accurate and effecient at identifying positive instances while minimizing false positives and false negatives.
Recall: A recall of 95% indicates that the model is correctly capturing 90% 
of the actual positive instances present in the data

INFERENCE ON TEST DATA
 we observe that precision in test data is 89% ,recall is 95% and f1 score is 92%. 
 f1-score:the metrics have decreased by 1% from the train data ,but we can conclude that A high F1 score (92% in this case) indicates that the model has achieved a good balance between precision and recall. 
 It suggests that the model is accurate and effective at identifying positive instances while minimizing false positives and false negatives.
 Recall:recall of 95% indicates that the model is correctly capturing 89% of the actual positive instances present in the data.
 Precision: A precision of 89% indicates that out of all instances predicted as positive by the model, 95% of them are actually positive. This suggests that the model is making positive predictions with high accuracy.##



"""



