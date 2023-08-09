import pandas as pd
from sklearn.model_selection import train_test_split
#classification
from sklearn.neighbors import KNeighborsClassifier


data=pd.read_csv("iris.csv")
print(data.head())
print(data.isnull().sum())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.89)

model=KNeighborsClassifier()

model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))



