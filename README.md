# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeClassifier from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Import metrics from sklearn and calculate the accuracy of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.



## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JaisonRaphael V
RegisterNumber:  212221230038
*/
```
~~~
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics   
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
~~~
## Output:
Data Head:

![1 (1)](https://user-images.githubusercontent.com/94165957/174470153-b79a3d0d-191b-4958-a7e8-2ee2e74addff.png)

Data Info:

![2 (1)](https://user-images.githubusercontent.com/94165957/174470168-6e9badd8-78b5-4ae4-97c9-1d51f3764e56.png)

Data Isnull:

![3 (1)](https://user-images.githubusercontent.com/94165957/174470173-87d62953-3b33-40be-9877-29679ffa0007.png)

Data Left:

![4 (1)](https://user-images.githubusercontent.com/94165957/174470175-7c5b28a5-5b20-4f49-be59-3e4d2c41f305.png)

Data Head:

![5 (1)](https://user-images.githubusercontent.com/94165957/174470178-62d05368-cf04-41d3-9bc5-9c37b491f899.png)

X.Head:

![6 (1)](https://user-images.githubusercontent.com/94165957/174470237-eeb83bb5-c800-4f3b-bd03-9d19f3380b27.png)

Data Fit:

![7 (1)](https://user-images.githubusercontent.com/94165957/174470245-8bf31931-b9b5-4bb3-864a-aa97b46f2bbf.png)

Accuracy:

![8 (1)](https://user-images.githubusercontent.com/94165957/174470250-3b672f4e-ec1a-4cd2-a637-2750a89ecdce.png)

Predicted Values:

![9](https://user-images.githubusercontent.com/94165957/174470264-6cf90595-758d-4309-b6fe-7f85ee34cc16.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
