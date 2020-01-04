salarytrain<-read.csv(file.choose())
salarytest<-read.csv(file.choose())

library(dummies)
salary_train<-dummy.data.frame(salarytrain, names = c("workclass","education","maritalstatus","occupation","relationship","race","sex","native") , sep = ".")
salary_test<-dummy.data.frame(salarytest, names = c("workclass","education","maritalstatus","occupation","relationship","race","sex","native") , sep = ".")

library(kernlab)

model_vanilla<-ksvm(Salary ~.,data = salary_train,kernel = "vanilladot")
pred_vanilla<-predict(model_vanilla,newdata=salary_test)
mean(pred_vanilla==salary_test$Salary)
table(pred_vanilla,salary_test$Salary)

model_rbf<-ksvm(Salary ~.,data = salary_train,kernel = "rbfdot")
pred_rbf<-predict(model_rbf,newdata=salary_test)
mean(pred_rbf==salary_test$Salary)
table(pred_rbf,salary_test$Salary)

model_bessel<-ksvm(Salary ~.,data = salary_train,kernel = "besseldot")
pred_bessel<-predict(model_bessel,newdata=salary_test)
mean(pred_bessel==salary_test$Salary)
table(pred_bessel,salary_test$Salary)

model_tanh<-ksvm(Salary ~.,data = salary_train,kernel = "tanhdot")
pred_tanh<-predict(model_tanh,newdata=salary_test)
mean(pred_tanh==salary_test$Salary)
table(pred_tanh,salary_test$Salary)

model_laplace<-ksvm(Salary ~.,data = salary_train,kernel = "laplacedot")
pred_laplace<-predict(model_laplace,newdata=salary_test)
mean(pred_laplace==salary_test$Salary)
table(pred_laplace,salary_test$Salary)

vanilla<-mean(pred_vanilla==salary_test$Salary)
rbf<-mean(pred_rbf==salary_test$Salary)
bessel<-mean(pred_bessel==salary_test$Salary)
tanh<-mean(pred_tanh==salary_test$Salary)
laplace<-mean(pred_laplace==salary_test$Salary)
table_mean<-data.frame(c("vanilla","rbf","bessel","tanh","laplace"),c(vanilla,rbf,bessel,tanh,laplace))
colnames(table_mean)<-c("model","mean")
View(table_mean)
