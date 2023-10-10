import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
data = pd.read_csv('creditcard.csv')

#Dataset informations
data.describe()
data.info()
data.head()
data.tail()

data.isnull().sum()

#Distribution of legit and fradulent transactions
data['Class'].value_counts()
#This dataseet is highly unbalanced
# 0 - Normal Transaction
# 1 - Fradulent transaction

# Separating the data for analysis
legit = data[data.Class==0]
fraud = data[data.Class==1]

print(legit.shape)
print(fraud.shape)

# Statistical measures of the data
legit.Amount.describe()
fraud.Amount.describe()

# Comparing the values of both the transactions
data.groupby('Class').mean()

# Building a sample dataset containing similar distribution of normal and fraudulent transaction
# Number of fraudulent transaction - 492 

legit.sample = legit.sample(492)

# Concatenating two dataframes
new_data = pd.concat([legit.sample,fraud],axis=0)

new_data.head()
new_data.tail()

new_data['Class'].value_counts()

new_data.groupby('Class').mean()

#  Splitting the dataset into Features and Targets
x = new_data.drop(['Class'],axis=1) 
y = new_data['Class']
y

# Training and Testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,stratify=y,random_state=0)

print(x.shape,x_train.shape,x_test.shape)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=500)
model.fit(x_train,y_train)
print('Accuracy:',model.score(x_test, y_test))

# Evaluation - Accuracy score
# Accuracy on training data
x_train_pred = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_pred,y_train)
print('Accuracy on training data:',training_data_accuracy)

# Accuracy on test data
x_test_pred = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_pred,y_test)
print('Accuracy on test data:',test_data_accuracy)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(x_train,y_train)
print('Accuracy:',dtc.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
rfc.fit(x_train,y_train)
print('Accuracy:',dtc.score(x_test,y_test))

from sklearn.svm import SVC
svm=SVC(kernel='linear',random_state=0)
svm.fit(x_train,y_train)
print('Accuracy:',svm.score(x_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print('Accuracy:',knn.score(x_test,y_test))