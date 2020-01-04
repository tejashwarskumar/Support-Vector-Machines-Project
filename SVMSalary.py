import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

salaryTrainData = pd.read_csv("C:/My Files/Excelr/16 - Support Vector Machines/Assignment/SalaryData_Train(1).csv")
salaryTestData = pd.read_csv("C:/My Files/Excelr/16 - Support Vector Machines/Assignment/SalaryData_Test(1).csv")
salaryTrainData.columns
salaryTrainData.describe()
salaryTrainData.shape

salaryTrainData["Salary"].value_counts()
salaryTrainData["Salary"][0]
salaryTrainData["Salary"] = np.where(salaryTrainData["Salary"] == " <=50K",0,1)
salaryTrainData["Salary"].value_counts()

sns.boxplot(x="Salary",y="age",data=salaryTrainData,palette = "hls")
sns.boxplot(data=salaryTrainData,palette = "hls")

salaryTestData["Salary"] = np.where(salaryTestData["Salary"] == " <=50K",0,1)
salaryTestData["Salary"].value_counts()
salaryTrainData.isnull().sum()
salaryTrainData.corr()

salaryTrainData.columns
cat_arr = ['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
for i in cat_arr:
    salaryTrainData[i] = prepocess.fit_transform(salaryTrainData[i])
    salaryTestData[i] = prepocess.fit_transform(salaryTestData[i])

trainX = salaryTrainData.drop('Salary',axis=1)
trainY = salaryTrainData['Salary']
testX = salaryTestData.drop('Salary',axis=1)
testY = salaryTestData['Salary']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

len(trainX)

acc=[];
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
for i in range(0,len(kernal_methods)):
    model = SVC(kernel=kernal_methods[i])
    model.fit(trainX,trainY)
    pred_train = model.predict(trainX)
    train_acc = np.mean(pred_train == trainY)
    pred_test = model.predict(testX)
    test_acc = np.mean(pred_test == testY)
    acc.append([train_acc,test_acc])

plt.plot(kernal_methods,[i[0] for i in acc],"bo-")
plt.plot(kernal_methods,[i[1] for i in acc],"ro-")
plt.legend(["train","test"])
