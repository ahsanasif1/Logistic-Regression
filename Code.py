#!/usr/bin/env python
# coding: utf-8

# # MUHAMMAD AHSAN ASIF
# # 218606833



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


# # Part 1



#1.1 Data Munging
train_data = pd.read_csv('train_wbcd.csv') 
test_data = pd.read_csv('test_wbcd.csv')





print("Shape of Training data : ",train_data.shape)
print("")
print(train_data.info())


# In[13]:


print("Shape of Testing data : ",train_data.shape)
print("")
print(test_data.info())


# In[19]:


#Printing the total number of B and M's in train and test data 
print('Training Data : ',train_data.Diagnosis.value_counts())
print("")
print('Testing Data : ',test_data.Diagnosis.value_counts())


# In[31]:


sns.countplot(train_data['Diagnosis'])
plt.title('Train data')


# In[32]:


sns.countplot(test_data['Diagnosis'])
plt.title("Testing Data")


# In[ ]:


#As seen in the above plots the class distribution of each Testing and Training data is not balanced.


# In[28]:


#Printing the number of features which have missing values
#For Training
print('Column having missing values : ',train_data.columns[train_data.isnull().any()])

#Also we can see the missing values through a plot, the yellow lines tell us where the missing values are in the data frame
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:


#For testing 
print('Column having missing values : ',test_data.columns[test_data.isnull().any()])

#Also we can see the missing values through a plot, the yellow lines tell us where the missing values are in the data frame
sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[44]:


#Filling the number of missing values with mean values

medianval = train_data['f21'].median()
train_data['f21'] = train_data['f21'].fillna(meanval)

medianval2 = test_data['f21'].median()
test_data['f21'] = test_data['f21'].fillna(meanval2)

#I chose the median method because 


# In[40]:


#Normalizing the data set

#Before normalizing the data i will change all the B and M into 0 
def mappingfunc(data,feature):
    Map=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        Map[i]=count
        count=count+1
    data[feature]=data[feature].map(Map)
    return data


# In[48]:


#Mapping the train data 
TR1 = mappingfunc(train_data,'Diagnosis')
#Mapping the test data
TE1 = mappingfunc(test_data,'Diagnosis')


# In[43]:


#Now creating a function to normalize the dataframe
# I am normalizing the entire dataframe except the Diagnosis and Patient ID columns as they aren't needed to be normalzied
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
    dataNorm["Diagnosis"]=dataset["Diagnosis"]
    dataNorm["Patient_ID"]=dataset["Patient_ID"]
    return dataNorm


# In[55]:


train_data = normalize(train_data)
test_data = normalize(test_data)


# In[ ]:





# In[ ]:


#1.2 Logistic Regression 


# In[61]:


#Before starting the Logistic Regression i will drop the Patient_ID column as i wont be needing it in my algorithms
train_data = train_data.drop('Patient_ID',axis = 1 )
test_data = test_data.drop('Patient_ID',axis = 1)


# In[ ]:


#After that selecting our predictors and response 


# In[65]:


#Getting all the columns 
print(train_data.columns)


# In[67]:


predictors = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
       'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',
       'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29',
       'f30']

response = ['Diagnosis']


# In[79]:


#Training l1 model with alpha

alpha_val = 0.1

#Initialize the Logitic regression model with l1 penalty
lr = LogisticRegression(C=1/alpha_val, penalty='l1',solver='liblinear')
lr.fit(train_data[predictors], train_data['Diagnosis'])
y_predict = lr.predict(test_data[predictors])

#Evaluate our model and checking the accuracy 

model_acc = accuracy_score(y_predict, test_data['Diagnosis'])
print("Model Accuracy is: {}".format(model_acc))
print("Model Coeff: {}".format(np.append(lr.intercept_, lr.coef_)))


# In[80]:


#Checking the precision,recall,f1-score for Benign and Malignant
print(classification_report(test_data['Diagnosis'],y_predict))


# In[81]:


#printing the confusion matrix for this model 
print(confusion_matrix(test_data["Diagnosis"],y_predict))


# In[77]:


#Now training the l2 model with lambda val


# In[82]:


lambda_val = 0.1
#Initialize the Logitic regression model with l2 penalty
lr = LogisticRegression(C=1/lambda_val, penalty='l2',solver='liblinear')
lr.fit(train_data[predictors], train_data['Diagnosis'])
y_predict2 = lr.predict(test_data[predictors])

#Evaluate our model and checking the accuracy for this model
model_acc = accuracy_score(y_predict2, test_data['Diagnosis'])
print("Model Accuracy is: {}".format(model_acc))
print("Model Coeff: {}".format(np.append(lr.intercept_, lr.coef_)))


# In[83]:


#Checking the precision,recall,f1-score for Benign and Malignant
print(classification_report(test_data['Diagnosis'],y_predict2))


# In[84]:


#printing the confusion matrix for this model 
print(confusion_matrix(test_data["Diagnosis"],y_predict2))


# In[ ]:


#1.3


# In[133]:


def runLRmodel(trials, data, predictors, label, penalty_type, penalty_score):

   model_acc     = 0
   model_weights = np.zeros([1,31])

   for i in range(0,trials):
      Dtrain, Dtest = train_test_split(data, test_size=0.3)
      lr = LogisticRegression(C=1/penalty_score, penalty=penalty_type,solver='liblinear')
      lr.fit(Dtrain[predictors], Dtrain[label])
      y_predict = lr.predict(Dtest[predictors])
    
#Since my fID = 1, i would be using f1_score
      model_acc += f1_score(y_predict, Dtest[label],average='micro')
      model_weights += np.append(lr.intercept_, lr.coef_)

   model_acc /= trials
   model_weights /= trials

   return np.round(model_acc, decimals=2), np.round(model_weights,decimals=2)


# In[134]:


#A part , for L1 model choosing the best alpha


# In[135]:


alpha_vals = [0.1,1,3,10,33,100,333,1000, 3333, 10000, 33333]
l1_acc = np.zeros(len(alpha_vals))
index = 0
#L1 regularization
for l in alpha_vals:
   l1_acc[index], w = runLRmodel(10,train_data, predictors, 'Diagnosis', 'l1', np.float(l))
   index += 1

print("Acc: {}".format(l1_acc))
# penalty at which validation accuracy is maximum
max_index_l1  = np.argmax(l1_acc)
best_alpha = alpha_vals[max_index_l1]
print("Best Alpha: {}".format(best_alpha))


# In[ ]:


#B part, for L2 model, selecting the best model 


# In[137]:


lambda_vals = [0.001, 0.003, 0.01, 0.03, 0.1,0.3,1,3,10,33]
l2_acc = np.zeros(len(lambda_vals))
index = 0
#L2 regularization
for l in lambda_vals:
   l2_acc[index], w = runLRmodel(10,train_data, predictors, 'Diagnosis', 'l2', np.float(l))
   index += 1

print("Acc: {}".format(l2_acc))
# penalty at which validation accuracy is maximum
max_index_l2  = np.argmax(l2_acc)
best_lambda = lambda_vals[max_index_l2]
print("Best Lambda: {}".format(best_lambda))


# In[ ]:


#C Part, Now I would be running again with the best alpha and best lamda to retrain the L1 and L2 model 


# In[139]:


#Retraining L1 model 
alphaValue = best_alpha
#Initialize the Logitic regression model with l1 penalty
lr = LogisticRegression(C=1/alpha_val, penalty='l1',solver='liblinear')
lr.fit(train_data[predictors], train_data['Diagnosis'])
y_predict3 = lr.predict(test_data[predictors])

#Evaluate our model and checking the accuracy for this model
model_acc = accuracy_score(y_predict3, test_data['Diagnosis'])
print("Model Accuracy is: {}".format(model_acc))
print("Model Coeff: {}".format(np.append(lr.intercept_, lr.coef_)))


# In[142]:


#Checking the precision,recall,f1-score for the test data
print(classification_report(test_data['Diagnosis'],y_predict3))
#printing the confusion matrix for this model 
print('Confusion Matrix : ')
print(confusion_matrix(test_data["Diagnosis"],y_predict3))


# In[ ]:


#printing the confusion matrix for this model 
print(confusion_matrix(test_data["Diagnosis"],y_predict3))


# In[143]:


#Retraining L2 model 
lamdaValue = best_lambda
#Initialize the Logitic regression model with l1 penalty
lr = LogisticRegression(C=1/lamdaValue, penalty='l2',solver='liblinear')
lr.fit(train_data[predictors], train_data['Diagnosis'])
y_predict4 = lr.predict(test_data[predictors])

#Evaluate our model and checking the accuracy for this model
model_acc = accuracy_score(y_predict4, test_data['Diagnosis'])
print("Model Accuracy is: {}".format(model_acc))
print("Model Coeff: {}".format(np.append(lr.intercept_, lr.coef_)))


# In[144]:


#Checking the precision,recall,f1-score for the test data
print(classification_report(test_data['Diagnosis'],y_predict4))
#printing the confusion matrix for this model 
print('Confusion Matrix : ')
print(confusion_matrix(test_data["Diagnosis"],y_predict4))


# In[210]:


#Top 5 features selected in decreasing order of feature weights 
Weights = w.T
Decreasing_order = np.sort(-Weights)
Decreasing_order[:5]


# # Part 2 (Multiclass Classification)

# In[145]:


#2.1
#1

question2data = pd.read_csv('reduced_mnist.csv')
#Total number of data points 
print(question2data.shape)
#Total number of features 
print(question2data.info())
#Total number of unique labels 
print(question2data.columns.unique)


# In[159]:


#2
#Splitting into training and testing data 

Dtrain,Dtest = train_test_split(question2data,test_size=0.3)

#Getting the labels from the columns 
predictors2 = Dtrain.columns[1:]
response2 = 'label'

alpha = 1
#Initialize the Logitic regression model with l1 penalty with a default one vs Rest Classifier
lr = LogisticRegression(C=1/alpha, penalty='l1',solver='liblinear',multi_class='ovr')
lr.fit(Dtrain[predictors2], Dtrain['label'])
y_predict5 = lr.predict(Dtest[predictors2])


# In[184]:



#Evaluate our model and checking the accuracy for this model
#Checking the precision, recall to their functions since classification report deosnt work with multiclass targets

model_acc_new = accuracy_score(y_predict5, Dtest['label'])
print("Model Accuracy is: {}".format(model_acc_new))

#Checking the precision
model_precision = precision_score(y_predict5,Dtest[response2],average='weighted')
print('Precision Score is : ',model_precision)

#Checking the recall
model_recall = recall_score(y_predict5,Dtest[response2],average='weighted')
print('Recall Score is : ',model_recall)

#Model Coefficients
print("Model Coeff: {}".format(np.append(lr.intercept_, lr.coef_)))


# In[174]:


#2.2
def runLRmodelMC(trials, data, predictors, label, penalty_type, penalty_score):

   model_acc     = 0
   model_train_acc     = 0
   model_weights = np.zeros([1,785])

   for i in range(0,trials):
      #print("test: ", penalty_score)
      Dtrain, Dtest = train_test_split(data, test_size=0.3)
      lr = LogisticRegression(C=1/penalty_score, penalty=penalty_type,multi_class='ovr',solver='liblinear')
      lr.fit(Dtrain[predictors], Dtrain[label])
      y_predict = lr.predict(Dtest[predictors])
      y_predict_train = lr.predict(Dtrain[predictors])
    
    #precision score 
      model_acc += precision_score(y_predict, Dtest[label], average='micro')
      model_train_acc += precision_score(y_predict_train, Dtrain[label], average='micro')  
      #print(model_train_acc)

   model_acc /= trials
   model_train_acc /= trials

   return np.round(model_acc, decimals=2), np.round(model_train_acc, decimals=2)


# In[175]:


lambda_vals1 = [0.1, 1, 3, 10, 33, 100, 333, 1000, 3333, 10000, 33333]
val_acc = np.zeros(len(lambda_vals1))
tr_acc = np.zeros(len(lambda_vals1))

index = 0
#L1 regularization
for l in lambda_vals1:
   val_acc[index], tr_acc[index] = runLRmodelMC(10,Dtrain, predictors2,'label', 'l1', np.float(l))
   index += 1


# In[177]:


print("Training Accuracy : {}".format(tr_acc))
print("Validation Accuracy: {}".format(val_acc))

# penalty at which validation accuracy is maximum
max_index_12  = np.argmax(val_acc)
max_index_13 = np.argmax(tr_acc)
best_lambda = lambda_vals[max_index_l2]
print("Best Lambda: {}".format(best_lambda))


# In[181]:


#plot the accuracy curve
plt.plot(range(0,len(lambda_vals1)), val_acc, color='b', label='Validation Accuracy')
plt.plot(range(0,len(lambda_vals1)), tr_acc, color='r', label='Training Accuracy')
#replace the x-axis labels with penalty values
plt.xticks(range(0,len(lambda_vals1)), lambda_vals1, rotation='vertical')
plt.plot((max_index_12, max_index_12), (0, val_acc[max_index_l2]), ls='dotted', color='b')
plt.plot((max_index_13, max_index_13), (0, tr_acc[max_index_13]), ls='dotted', color='b')
plt.xlabel("Alpha Value")
plt.ylabel('Average Model Accuracy ')

#Set the y-axis from 0 to 1.0
axes = plt.gca()
axes.set_ylim([0, 1.1])

plt.legend(loc="lower left")
plt.show()

#After looking at the graph we can that after the alpha value '333' the model was underfitting and before that the model was overfitting


# In[187]:


#2 Building the final model with the best alpha


alpha = best_alpha
#Initialize the Logitic regression model with l1 penalty with a default one vs Rest Classifier
lr = LogisticRegression(C=1/alpha, penalty='l1',solver='liblinear',multi_class='ovr')
lr.fit(Dtrain[predictors2], Dtrain['label'])
y_predict6 = lr.predict(Dtest[predictors2])


# In[190]:


print("Confusion Matrix : ")
print(confusion_matrix(Dtest['label'],y_predict6))


# In[196]:


#Precision score for the new model
print(precision_score(Dtest['label'],y_predict6,average=None))


# In[193]:


#Recall score for the new model
print(recall_score(Dtest['label'],y_predict6,average=None))


# In[195]:


#Accuracy score for the new model
print(accuracy_score(Dtest['label'],y_predict6))


# In[ ]:


#3 
#As seen in the graph that the training accuracy went more than 90% whereas the Validation accuracy stayed below 90% but more than 85% therefore there is overfitting seen in the graph

