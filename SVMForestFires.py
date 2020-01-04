import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

forestData = pd.read_csv("C:/My Files/Excelr/16 - Support Vector Machines/Assignment/forestfires.csv")
forestData.head()
forestData.describe()

forestData.columns
forestData.drop('month',inplace=True,axis=1)
forestData.drop('day',inplace=True,axis=1)
forestData.isnull().sum()
forestData.corr()


forestData['size_category'].value_counts()
forestData['size_category'].value_counts().plot(kind="bar")

#convert size to dummy variable
forestData['size_category'] = np.where(forestData['size_category'] == "small",0,1)

from sklearn.model_selection import train_test_split
train,test = train_test_split(forestData,test_size=0.3)
trainX = train.drop('size_category',axis=1)
trainY = train['size_category']
testX = test.drop('size_category',axis=1)
testY = test['size_category']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(trainX,trainY)
pred_train_linear = model_linear.predict(trainX)
pred_test_linear = model_linear.predict(testX)

train_acc = np.mean(pred_train_linear == trainY)
train_acc
test_acc = np.mean(pred_test_linear == testY)
test_acc

kernal_methods = ['linear', 'poly', 'rbf', 'sigmoid']

acc=[];
for i in range(0,len(kernal_methods)):
    print(kernal_methods[i])
    model = SVC(kernel=kernal_methods[i])
    model.fit(trainX,trainY)
    pred_train = model.predict(trainX)
    print(pred_train)
    train_acc_all = np.mean(pred_train == trainY)
    pred_test = model.predict(testX)
    print(pred_test)
    test_acc_all = np.mean(pred_test == testY)
    acc.append([train_acc,test_acc])

plt.plot(kernal_methods,[i[0] for i in acc],"bo-")
plt.plot(kernal_methods,[i[1] for i in acc],"ro-")
plt.legend(["train","test"])
