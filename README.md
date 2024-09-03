# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: pavithran S
RegisterNumber: 212223240113 
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![image](https://github.com/user-attachments/assets/46eda9e4-7a78-4373-be9f-7901c188d5ea)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/da5fe063-a634-404f-86f7-161d1c1333af)

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![image](https://github.com/user-attachments/assets/0fcd5cf9-f355-4ae9-8b52-4aa86420d022)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape
```
![image](https://github.com/user-attachments/assets/b88df7d6-09ed-44eb-ab60-3634cea3a06e)
```
X_test.shape
```
![image](https://github.com/user-attachments/assets/83a87fcb-6f22-434e-b77c-2269e6415c3a)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, Y_train)
```
![image](https://github.com/user-attachments/assets/8104bf0a-8d1a-40d5-bbfd-85f33447eaee)
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![image](https://github.com/user-attachments/assets/b0539479-4006-41f2-b6b0-658a2943d89f)
```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/aa7025ea-6074-4ce1-ad34-fea318104683)
![image](https://github.com/user-attachments/assets/ad0df20a-ff8b-411b-9367-ea50eb99d025)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
