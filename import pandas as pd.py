import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\SRINIVAS\\OneDrive\\Documents\\csv\\50_Startups.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,2]=labelencoder.fit_transform(x[:,2])
onehotencoder=OneHotEncoder()
x=onehotencoder.fit_transform(x).toarray()
print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(y_test,y_pred)
print("train score:",regressor.score(x_train,y_train))
print("test score:",regressor.score(x_test,y_test))
plt.title("actual vs predict")
plt.xlabel("position levels")
plt.ylabel("predicted values")
plt.plot([min(y_test),max(y_test),min(y_test),max(y_test)],linestyle='--',color='blue')
plt.show()
