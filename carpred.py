import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset

carprice=pd.read_csv('carprice.csv')
carprice.describe()
carprice.head()
carprice.info()


carprice.isnull().sum()
carprice.columns
print(carprice['name'].value_counts()) #Values of categorical function
carprice.shape

carprice.head(1)
carprice['name'].unique()
carprice.dtypes
carprice['name'].value_counts()
pd.get_dummies(carprice, columns=["name","fueltypes","aspiration","doornumbers","carbody","drivewheels","enginelocation","enginetype","cylindernumber","fuelsystem"],prefix=["name1","fueltype1","aspiration1","doornumbers1","carbody1","drivewheels1","engineloc1","enginetype1","cylindernum1","fuelsys1"])
#Encoding for the categorical datas
x
y
x=carprice.drop(['name','price','fueltypes','aspiration','doornumbers','carbody','drivewheels','enginelocation','enginetype','cylindernumber','fuelsystem'],axis=1)
y=carprice['price']
x
y
#Splitting the dataset in to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Importing the models
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(x_train,y_train)

#predicting the test results
y_pred=linear.predict(x_test)

#To check accuracy
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)
accuracy
