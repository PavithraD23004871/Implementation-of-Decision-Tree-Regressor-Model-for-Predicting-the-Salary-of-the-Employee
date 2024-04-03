# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2.

## Program:
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Pavithra D 
RegisterNumber:  212223230146
*/
```
import pandas as pd
data=pd.read_csv('/content/Salary_EX7.csv')
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train , y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt, feature_names=x.columns,filled=True)
plt.show()
```
## Output:

HEAD:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138955967/5f90c73c-4c37-4d4c-b7d4-f426df909f2b)

MSE:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138955967/176f9622-b50f-470c-8fe2-2e30dd1fa158)

r2:

![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138955967/f0b60e74-7b1f-4cfa-beee-1e6436580f3a)


![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138955967/89fa5184-f31e-4a16-96e1-8092f2e78607)


![image](https://github.com/PavithraD23004871/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/138955967/f9fc75be-a125-44cf-afc9-e1426b27b856)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
