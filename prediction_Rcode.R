#import important libraries
library(randomForest)
library(corrplot)
library(psych)
library(e1071)
library(caret)
library(ggplot2)
library(gbm)
library(dplyr)
library(kernlab)
library(ROCR)

#import data
df.forest <- read.csv("/Users/adityakulai/Desktop/BA project/forestfires.csv") 
df.forest <- tbl_df(df.forest) 

set.seed(1234)

#check skew so use log
hist(df.forest$area)
rug(df.forest$area)
df.forest <- mutate(df.forest, y = log(area + 1))  
hist(df.forest$y)

# normalize
# subtract the min value in x and divide by the range of values in x.

normalise <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))  
}
df.forest$temp <- normalise(df.forest$temp)
df.forest$rain <- normalise(df.forest$rain)
df.forest$RH <- normalise(df.forest$RH)
df.forest$wind <- normalise(df.forest$wind)

#check the number of small and large fires
sum(df.forest$area < 5)
sum(df.forest$area >= 5)

#create a new column and add 'small' and 'large' labels for area<5 hectares and area>=5 hectares respectively
df.forest$size <- NULL
df.forest$size <- factor(ifelse(df.forest$area < 5, 1, 0),
                         labels = c("small", "large"))

#seperate into data into training and testing
intrain <- sample(x = nrow(df.forest), size = 400, replace = FALSE)

#training linear svm classifier
m.lin <- svm(size ~ temp + RH + wind + rain,
             data = df.forest[intrain, ],
             kernel = "linear", C = 1)
#check error rate of training model
m.lin
#predict and check accuracy
pred <- predict(m.lin, newdata = df.forest[-intrain, ], type = "response")
table(pred, df.forest[-intrain, "size"][[1]])  
confusionMatrix(table(pred, df.forest[-intrain, "size"][[1]]), positive = "small") 

#training polynomial svm classifier
m.poly <- ksvm(size ~ temp + RH + wind + rain,
               data = df.forest[intrain, ],
               kernel = "polydot", C = 1)
m.poly
#training radial svm classifier
m.rad <- ksvm(size ~ temp + RH + wind + rain,
              data = df.forest[intrain, ],
              kernel = "rbfdot", C = 1)
m.rad
#training tan svm classifier
m.tan <- ksvm(size ~ temp + RH + wind + rain,
              data = df.forest[intrain, ],
              kernel = "tanhdot", C = 1)
m.tan

# as lowest error rate amongst the 3 classifiers above is of radial so
# predict using radial and check accuracy

pred <- predict(m.rad, newdata = df.forest[-intrain, ], type = "response")
table(pred, df.forest[-intrain, "size"][[1]])  
confusionMatrix(table(pred, df.forest[-intrain, "size"][[1]]), positive = "small")  # from the caret package, also need e1071 package

#test ensemble classifiers - random forest and gradient boosting model
#first lets fit random forest classifier and check error rate

forest.rf <- randomForest(size ~ temp + RH + wind + rain, data = df.forest[intrain, ], importance=TRUE, ntree=300)
forest.rf
#predict and find accuracy
pred <- predict(forest.rf, newdata = df.forest[-intrain, ], type = "response")
confusionMatrix(table(pred, df.forest[-intrain, "size"][[1]]), positive = "small")  # from the caret package, also need e1071 package

#now gradient boosting classifier and check error rate
fitControl <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 1)
mod_BR <- train(size ~ temp + RH + wind + rain, df.forest[intrain, ], method="gbm", trControl=fitControl, verbose = FALSE)
#predict and find accuracy
pred <- predict(mod_BR, newdata = df.forest[-intrain, ])
plot(mod_BR, main = "Model 2")
confusionMatrix(table(pred, df.forest[-intrain, "size"][[1]]), positive = "small")  # from the caret package,

#best model from above is radial svm classifier which least error rate and highest accuracy