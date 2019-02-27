#####NEURAL NETWORK#####


#Predict the burned area of forest fires with neural networks

#Loading the dataset
ff<- read.csv(file.choose())
View(ff)
str(ff)
attach(ff)
library(DataExplorer)
plot_str(ff)
plot_missing(ff)

#EDA and Visualizations
summary(ff)
#FFMC: Mean= 90.64, Median= 91.60; As Mean<Median,it is skewed to the right.
#DMC: Mean= 110.9, Median= 108.3; As Mean>Median,it is skewed to the left.
#DC: Mean= 547.9, Median= 664.2; As Mean<Median,it is skewed to the right.
#ISI:Mean= 8.400, Median= 19.30; As Mean<Median,it is skewed to the right.
#temp: Mean= 18.89, Median= 19.30; As Mean<Median,it is skewed to the right.
#RH: Mean= 44.29, Median= 42.00; As Mean>Median,it is skewed to the left.
#wind: Mean= 4.018, Median= 4.000; As Mean>Median,it is skewed to the left.
#rain: Mean= 0.02166, Median= 0.0000; As Mean>Median,it is skewed to the left.
#area: Mean= 12.85, Median= 0.52; As Mean>Median,it is skewed to the left.

plot_histogram(FFMC)
plot_histogram(DMC)
plot_histogram(DC)
plot_histogram(ISI)
plot_histogram(temp)
plot_histogram(RH)
plot_histogram(wind)
plot_histogram(area)

#Applying normalization to dataset
ff_norm <-as.data.frame(scale(ff[,-c(1,2,31)]))
summary(ff_norm$area)
attach(ff_norm)

#Paritioning data
ff_train <- ff_norm[1:410,]
ff_test <- ff_norm[411:517,]

#ANN (6 hidden neuron)
library(nnet)
library(neuralnet)
ff_model1 <- neuralnet(area~dayfri+daymon+daysat+daysun+daythu+daytue+daywed+
                         FFMC+DMC+DC+ISI+temp+RH+wind+rain+
                         monthapr+monthaug+monthdec+monthfeb+monthjan+monthjul+
                         monthjun+monthmar+monthmay+monthnov+monthoct+monthsep
                       ,data = ff_train,hidden = 6,stepmax=1e6)
str(ff_model1)
summary(ff_model1)

#Visualizing the network topology
plot(ff_model1)

library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(ff_model1, alpha = 0.6)

#Evaluating model performance
set.seed(1000)
ff_results <- compute(ff_model1,ff_test[,-9])
str(ff_results)
predicted_area <- ff_results$net.result

cor(predicted_area,ff_test$area)
# -0.0328501274
plot(predicted_area,ff_test$area)

#Improving model performance by increasing complexity 
#ANN (10 hidden neurons)
ff_model2 <- neuralnet(area~dayfri+daymon+daysat+daysun+daythu+daytue+daywed+
                         FFMC+DMC+DC+ISI+temp+RH+wind+rain+
                         monthapr+monthaug+monthdec+monthfeb+monthjan+monthjul+
                         monthjun+monthmar+monthmay+monthnov+monthoct+monthsep
                       ,data = ff_train,hidden = 10,stepmax=1e6)
str(ff_model2)
summary(ff_model2)
plot(ff_model2)

#Visualizing the network topology
plot(ff_model2)

library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(ff_model2, alpha = 0.6)

#Evaluating model performance
set.seed(1000)
ff_results2 <- compute(ff_model2,ff_test[,-9])
str(ff_results2)
predicted_area2 <- ff_results2$net.result

cor(predicted_area2,ff_test$area)
# 0.003034345092
plot(predicted_area2,ff_test$area)

par(mar = numeric(4), family = 'serif')
plotnet(ff_model2, alpha = 0.6)
#Error has reduced and training steps have increased as the number of neurons under 
#hidden layer have increased