#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd 
   
# Takes the file
filepath = r"diabetes.csv"
   
# read the CSV file 
df = pd.read_csv(filepath) 
   
# print the data set
print(df)


# In[7]:


print(df['Outcome'].value_counts())


# In[11]:


# The data set is in-balanced

df_suffle = df.sample(frac=1,random_state=4)

# Get all the Diabetic records
Diabetic_df = df_suffle.loc[df_suffle['Outcome'] == 1]

# Get 270 non-diabetic records
Nondiabetic_df = df_suffle.loc[df_suffle['Outcome'] == 0].sample(n=270,random_state=25)

# Concatenate both dataframes again
Balanced_df = pd.concat([Diabetic_df, Nondiabetic_df])

print(Balanced_df)


# In[16]:


#plot the dataset
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 8))
sns.countplot('Outcome', data=Balanced_df)
plt.title('Balanced Classes')
plt.show()


# In[17]:


# Now it is required to find the retationship between two attribute, if two attributes are highly related 
# we can drop one to reduce the computational complexity(but in this data set we have less records still checking)

#Plot heatmap and show case correlation of each features
correlation_matrix = Balanced_df.corr()
plt.figure(figsize=(15,12))
sns.heatmap(correlation_matrix , annot = True, cmap = "coolwarm")
plt.show()


# In[ ]:


# From the plot it is observerd that no attribute is highly related with any other attribute except dogonal value 
# no value is close to .3.Therefore, all attributes will contribute.  


# In[21]:


#Assigning training and test data
from sklearn.model_selection import train_test_split
X = Balanced_df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y = Balanced_df.Outcome


X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[24]:


#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)
Y_pred = rf_classifier.predict(X_test)
print("Accuracy of the model:",accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[25]:


from sklearn.svm import SVC
svm_classifier = SVC()
svm_classifier.fit(X_train, Y_train)
Y_pred = svm_classifier.predict(X_test)
print("Accuracy of the model:",accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[ ]:




