#####NEURAL NETWORK#####

#Build a Neural Network model for 50_startups data to predict profit 

#Loading the data
su<- read.csv(file.choose())
View(su)
str(su)

library(plyr)
su$State<- as.numeric(revalue(su$State ,c("New York"="0", "California"="1",
                                          "Florida"="2")))
str(su$State)
attach(su)

library(DataExplorer)
plot_str(su)
plot_missing(su)

#EDA and Visualizations
summary(su)
#R.D.Spend: Mean= 73722, Median= 73051; As Mean>Median,it is skewed to the left.
#Administration: Mean= 121345, Median= 122700; As Mean<Median,it is skewed to the right.
#Marketing.Spend: Mean= 211025, Median= 212716; As Mean>Median,it is skewed to the left.
#Profit: Mean= 112013, Median= 107978; As Mean>Median,it is skewed to the left.

plot_histogram(R.D.Spend)
plot_histogram(Administration)
plot_histogram(Marketing.Spend)
plot_histogram(Profit)

plot(R.D.Spend, Profit)
plot(Administration, Profit)
plot(Marketing.Spend, Profit)
plot(State, Profit)

#Applying normalization to dataset
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}

fifty_norm<-as.data.frame(lapply(su,FUN=normalize))
summary(fifty_norm$Profit)

summary(su$Profit)

#Paritioning data
set.seed(1234)
ind <- sample(2, nrow(fifty_norm), replace = TRUE, prob = c(0.7,0.3))
su_train <- fifty_norm[ind==1,]
su_test  <- fifty_norm[ind==2,]

#ANN (1 hidden neuron)
library(nnet)
library(neuralnet)
fifty_model <- neuralnet(Profit~R.D.Spend+Administration
                         +Marketing.Spend+State, data=su_train)
str(fifty_model)
summary(fifty_model)

#Visualizing the network topology
plot(fifty_model)

library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(fifty_model, alpha = 0.6)

#Evaluating model performance
set.seed(12345)
model_results <- compute(fifty_model,su_test[1:4])
predicted_profit <- model_results$net.result

cor(predicted_profit,su_test$Profit)
#0.9442474097
plot(predicted_profit,su_test$Profit)

#Unnormalizing prediction for actual results
str_max <- max(su$Profit)
str_min <- min(su$Profit)

unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}

ActualProfit_pred <- unnormalize(predicted_profit,str_min,str_max)
head(ActualProfit_pred)

#Improving model performance by increasing complexity 
#ANN (4 hidden neurons)
set.seed(12345)
fifty_model2 <- neuralnet(Profit~R.D.Spend+Administration
                          +Marketing.Spend+State,data = su_train,
                          hidden = 4)
str(fifty_model2)
summary(fifty_model2)

#Visualizing the network topology
plot(fifty_model2)

par(mar = numeric(4), family = 'serif')
plotnet(fifty_model2, alpha = 0.6)

#Evaluating model performance
model_results2 <- compute(fifty_model2,su_test[1:4])
predicted_profit2 <- model_results2$net.result

cor(predicted_profit2,su_test$Profit)
#0.95323592
plot(predicted_profit2,su_test$Profit)

par(mar = numeric(4), family = 'serif')
plotnet(fifty_model2, alpha = 0.6)
#Error has reduced and training steps have increased as the number of neurons under 
#hidden layer have increased