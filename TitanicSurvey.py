import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.describe()
train.head()
test.describe()
test.head()
train.info()
test.info()

# Combining two dataframes
data = pd.concat([train,test],axis=0)
data.describe()
data.info()
data.head()
data.tail()
data = data.reset_index()

data = data.drop(['Name','PassengerId','Cabin','Ticket'],axis=1)
data.info()
data.head()

data.isnull()
data.isnull().sum()

data['Age'].mean()
data['Fare'].mean()

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

data.isnull().sum()

data['Embarked'].mode()[0]
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data.isnull().sum()

data['Sex'].unique()
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Sex'].unique()

data['Embarked'].unique()
data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2})
data['Embarked'].unique()

#About Survived columns
data['Survived'].unique()
#0 Represents not survived and 1 represents survived
#Survived vs Not Survived
data['Survived'].value_counts() #Gets total values of columns
#Survived and not survived at Gender level
 #To compare two columns
#Filling missing values

data['Survived'].mode()[0]
data['Survived'] = data['Survived'].fillna(data['Survived'].mode()[0])

data.isnull().sum()

# x and y
x = data.drop(['Survived'],axis=1)
y = data['Survived']

y.head()
y.info()

data.head()

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=0)
x_train
x_test


from sklearn.model_selection import cross_val_score
score= cross_val_score(x_test,y_test)
print('CV Score:',np.mean(score))

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(max_iter=500)
logreg.fit(x_train,y_train)
print('Accuracy:',logreg.score(x_test, y_test))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
print('Accuracy:',dtc.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
print('Accuracy:',dtc.score(x_test,y_test))

from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
print('Accuracy:',svm.score(x_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print('Accuracy:',knn.score(x_test,y_test))