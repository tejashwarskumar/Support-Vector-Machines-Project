forestfire1<-read.csv(file.choose())
forestfire<-forestfire1[,-c(1:2)]
View(forestfire)

smp_size <- floor(0.9*nrow(forestfire))
set.seed(123)
train_ind <- sample(seq_len(nrow(forestfire)), size = smp_size)
forestfire_train <- forestfire[train_ind,]
forestfire_test <- forestfire[-train_ind,]

library(kernlab)

model_vanilla<-ksvm(size_category~.,data = forestfire_train,kernel = "vanilladot")
pred_vanilla<-predict(model_vanilla,newdata=forestfire_test)
vanilla<-mean(pred_vanilla==forestfire_test$size_category)
vanilla
table(pred_vanilla,forestfire_test$size_category)

model_rbfdot<-ksvm(size_category~.,data = forestfire_train,kernel = "rbfdot")
pred_rbfdot<-predict(model_rbfdot,newdata=forestfire_test)
rbf<-mean(pred_rbfdot==forestfire_test$size_category)
rbf
table(pred_rfdot,forestfire_test$size_category)

model_besseldot<-ksvm(size_category~.,data = forestfire_train,kernel = "besseldot")
pred_bessel<-predict(model_besseldot,newdata=forestfire_test)
bessel<-mean(pred_bessel==forestfire_test$size_category)
bessel
table(pred_bessel,forestfire_test$size_category)

model_tanh<-ksvm(size_category~.,data = forestfire_train,kernel = "tanhdot")
pred_tanh<-predict(model_tanh,newdata = forestfire_test)
tanh<-mean(pred_tanh==forestfire_test$size_category)
tanh
table(pred_tanh,forestfire_test$size_category)

model_laplace<-ksvm(size_category~.,data = forestfire_train,kernel = "laplacedot")
pred_laplace<-predict(model_laplace,newdata = forestfire_test)
laplace<-mean(pred_laplace==forestfire_test$size_category)
laplace
table(pred_laplace,forestfire_test$size_category)

table_mean<-data.frame(c("vanilla","rbf","bessel","tanh","laplace"),c(vanilla,rbf,bessel,tanh,laplace))
colnames(table_mean)<-c("model","mean")
View(table_mean)
