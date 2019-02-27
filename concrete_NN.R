#####NEURAL NETWORK#####

#Prepare a model for strength of concrete data using Neural Networks

#Loading the dataset
concrete<- read.csv(file.choose())
View(concrete)
str(concrete)
attach(concrete)

library(DataExplorer)
plot_str(concrete)
plot_missing(concrete)

#EDA and Visualizations
summary(concrete)
#cement: Mean= 281.2, Median= 272.9; As Mean>Median,it is skewed to the left.
#slag: Mean= 73.9, Median= 22.0; As Mean>Median,it is skewed to the left.
#ash: Mean= 54.19, Median= 0.00; As Mean>Median, it is skewed to the left. 
#water: Mean= 181.6, Median= 185.0; As Mean<Median, it is skewed to the right. 
#superplastic: Mean= 6.205, Median= 6.400;As Mean<Median, it is skewed to the right.
#coarseagg: Mean= 972.9, Median= 968.0;As Mean>Median, it is skewed to the left.
#fineagg: Mean= 773.6, Median= 779.5;As Mean<Median, it is skewed to the right. 
#age: Mean= 45.66, Median= 28.00;As Mean>Median, it is skewed to the left.
#strength: Mean= 35.82, Median= 34.45;As Mean>Median, it is skewed to the left.

plot_histogram(cement)
plot_histogram(slag)
plot_histogram(ash)
plot_histogram(water)
plot_histogram(superplastic)
plot_histogram(coarseagg)
plot_histogram(fineagg)
plot_histogram(age)
plot_histogram(strength)

#Applying normalization to dataset
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}

concrete_norm<-as.data.frame(lapply(concrete,FUN=normalize))
summary(concrete_norm$strength)

summary(concrete$strength)

#Paritioning data
set.seed(1234)
ind <- sample(2, nrow(concrete_norm), replace = TRUE, prob = c(0.7,0.3))
concrete_train <- concrete_norm[ind==1,]
concrete_test  <- concrete_norm[ind==2,]

#ANN (1 hidden neuron)
library(nnet)
library(neuralnet)
concrete_model <- neuralnet(strength~cement+slag+ash+water+superplastic+coarseagg+
                              fineagg+age,data = concrete_train)
str(concrete_model)
summary(concrete_model)

#Visualizing the network topology
plot(concrete_model)

library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(concrete_model, alpha = 0.6)

#Evaluating model performance
set.seed(12345)
concrete_results <- compute(concrete_model,concrete_test[1:8])
predicted_strength <- concrete_results$net.result

cor(predicted_strength,concrete_test$strength)
#0.8429302469
plot(predicted_strength,concrete_test$strength)

#Unnormalizing prediction for actual results
str_max <- max(concrete$strength)
str_min <- min(concrete$strength)

unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}

ActualStrength_pred <- unnormalize(predicted_strength,str_min,str_max)
head(ActualStrength_pred)

#Improving model performance by increasing complexity 
#ANN (5 hidden neurons)
set.seed(12345)
concrete_model2 <- neuralnet(strength~cement+slag+ash+water+superplastic+coarseagg+
                               fineagg+age,data = concrete_train, hidden = 5)
str(concrete_model2)
summary(concrete_model2)

#Visualizing the network topology
plot(concrete_model2)

par(mar = numeric(4), family = 'serif')
plotnet(concrete_model2, alpha = 0.6)

#Evaluating model performance
concrete_results2 <- compute(concrete_model2,concrete_test[1:8])
predicted_strength2 <- concrete_results2$net.result

cor(predicted_strength2,concrete_test$strength)
#0.9354678069
plot(predicted_strength2,concrete_test$strength)

par(mar = numeric(4), family = 'serif')
plotnet(concrete_model2, alpha = 0.6)
#Error has reduced and training steps have increased as the number of neurons under 
#hidden layer have increased