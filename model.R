library(ggplot2)
library(tidyverse)
library(glmnet)    # for ridge regression and lasso
library(tidyverse) # for data cleaning
library(psych)     # for function tr() to compute trace of a matrix


# importing the data
data = read.csv("./datasets/healthcare-dataset-stroke-data.csv")

# data cleaning
print(sum(data$bmi == "N/A"))  # 201 rows with no BMI
print(sum(data$gender == "Other"))  # only ONE row, so we can omit it to reduce the model's complexity
print(sum(data$smoking_status == "Unknown"))  # 1544 rows with unknown smoking status

# total rows removed
print(sum(data$bmi == "N/A" | data$gender == "Other" | data$smoking_status == "Unknown"))
# 1685 rows will be removed, leaving 3425 rows or around 67% of the data

data = data[!(data$bmi == "N/A"), ]  # remove bmi which is N/A
data = data[!(data$gender == "Other"), ]  # remove the only row with 'Other' gender
data = data[!(data$smoking_status == "Unknown"), ]  # remove unknown smoking status

# bmi is in a string format, we need to convert to double
data = transform(data, bmi = as.numeric(bmi))

#remove id column
sapply(data, class)
data = data[,-1]

#correlation and scatter plot
chart.Correlation(data[sapply(data, function(x) !is.character(x))], histogram = TRUE, method = "pearson", pch=19)

#plotting avgglucoselvl ~ bmi
cor.val = round(cor(data$bmi, data$avg_glucose_level), 2)
cor.label = paste0("Correlation: ", cor.val)
ggplot(data,
       aes(x = bmi,
           y = avg_glucose_level)) +
    geom_point() +
    annotate(x = 75, y = 300,  geom = "text", 
             label = cor.label, size = 5) +
    labs(x = "BMI", 
         y = "Average Glucose Level") +
    ggtitle("Scatterplot of Average Glucose Level VS BMI")
# Based on the scatterplot of BMI on average glucose level, 
# there seems to be no linear relationship between the two variables
# since their correlation is close to 0.


# creating training and test data
set.seed(123)
testIndices = sample(1:3425, nrow(data) * 0.25, replace=F)
trainData = data[-testIndices, ]
testData = data[testIndices, ]

# logistic regression - create model stroke ~ bmi
modelBMI = glm(stroke ~ bmi, trainData, family="binomial")

summary(modelBMI)
# Based on the P value produced from the model above, we can see that BMI actually 
# does not significantly affect stroke incidents.

# prediction for stroke ~ bmi
probStroke <- modelBMI %>% predict(trainData, type = "response")
predStroke <- ifelse(probStroke > 0.5, "yes", "no")
predStroke

# logistic function probability curve
trainData %>%
    mutate(prob = ifelse(stroke == "yes", 1, 0)) %>%
    ggplot(aes(bmi, prob)) +
    geom_point(alpha = 0.15) +
    geom_smooth(method = "glm", method.args = list(family = "binomial")) +
    labs(
        title = "Logistic Regression Model for Stroke VS BMI", 
        x = "BMI",
        y = "Probability of Being Stroke"
    )
# the model also shows that BMI does not have a significant impact in predicting stroke.
# and thus we would be looking at the
# other factors affecting stroke incidents.

modelStroke = glm(stroke ~ gender + age + hypertension + heart_disease + Residence_type + 
                       avg_glucose_level + smoking_status + bmi, trainData, family="binomial")
summary(modelStroke)

# since hypertension is significant in predicting stroke
# , we would like to see if bmi is significant in predicting hypertension
# logistic function probability curve predicting hypertension ~ bmi
trainData %>%
    mutate(prob = ifelse(hypertension == "yes", 1, 0)) %>%
    ggplot(aes(bmi, prob)) +
    geom_point(alpha = 0.15) +
    geom_smooth(method = "glm", method.args = list(family = "binomial")) +
    labs(
        title = "Logistic Regression Model for Hypertension VS BMI", 
        x = "BMI",
        y = "Probability of Having Hypertension"
    )
# Similar to our findings for BMI on stroke, does not seem to have any significance in predicting hypertension.


# To achieve a model with only significant variables that affect stroke, we can 
# use stepwise selection method, whilst trying both AIC and BIC.

step.fit.aic.stroke = step(modelStroke, direction = "both", criterion = AIC)
summary(step.fit.aic.stroke) #age,hypertension,heart disease, avg glucose lvl
step.fit.bic.stroke = step(modelStroke, k=log(nrow(trainData)), direction = "both", criterion = BIC)
summary(step.fit.bic.stroke) #age, avg glucose level

# We can really see that BMI does not significantly affect incidence of stroke in
# all possible models (AIC,BIC), but lets see if BMI affects hypertension, glucose level, heart disease
modelGlucoseBMI = lm(avg_glucose_level~ bmi, data = trainData)
summary(modelGlucoseBMI)  #IT DOES!!!!
resid_glucose_BMI <- residuals(modelGlucoseBMI)
plot(resid_glucose_BMI) #residuals seem to have neither patterns nor variations 

modelHypertensionBMI = glm(hypertension ~ bmi, trainData, family="binomial")
summary(modelHypertensionBMI)  #YES IT DOES

modelHeartdiseaseBMI = glm(heart_disease ~ bmi, trainData, family="binomial")
summary(modelHeartdiseaseBMI)  #Nope

# now we see other possible variables that may increase the likelihood of hypertension, heart disease or increase your glucose level

step.fit.aic.stroke = step(modelStroke, direction = "both")
summary(step.fit.aic.stroke)


step.fit.bic.stroke = step(modelOverall, k=log(nrow(trainData)), direction = "both")
summary(step.fit.bic.stroke)

# We can really see that BMI does not significantly affect incidence of stroke in
# all 3 models (at all), but lets see if BMI affects hypertension and glucose level
modelGlucoseBMI = lm(avg_glucose_level~ bmi, data = data)
summary(modelGlucoseBMI)  # IT DOES!!!!

modelHypertensionBMI = glm(hypertension ~ bmi, trainData, family="binomial")
summary(modelHypertensionBMI)  # YES IT DOESS~~~~~

# now we see other possible variables that may increase the likelihood of hypertension or increase your glucose level

modelHypertension = glm(hypertension ~ .-stroke, trainData, family="binomial")
step.fit.aic.hyper = step(modelHypertension, direction = "both")
summary(step.fit.aic.hyper)
step.fit.bic.hyper = step(modelHypertension, k=log(nrow(trainData)), direction = "both")

summary(step.fit.bic.hyper)  # bmi and age positively affect hypertension

summary(step.fit.bic)  # bmi and age affect positively hypertension


modelHeartdisease = glm(heart_disease ~ .-stroke, trainData, family="binomial")
step.fit.aic.heart = step(modelHeartdisease, direction = "both")
summary(step.fit.aic.heart)
step.fit.bic.heart = step(modelHeartdisease, k=log(nrow(trainData)), direction = "both")
summary(step.fit.bic.heart) #avg glucose level, gender=Male and age positively affect heartdisease

modelGlucose = lm(avg_glucose_level ~ .-stroke, trainData, family="binomial")
resid_glucose <- residuals(modelGlucose)
plot(resid_glucose)
step.fit.aic.gluc = step(modelGlucose, direction = "both")
summary(step.fit.aic.gluc)
step.fit.bic.gluc = step(modelGlucose, k=log(nrow(trainData)), direction = "both")

summary(step.fit.bic.gluc)  # bmi,age, heart disease, gender=Male positively affect glucose level

# hypertension, heart disease and glucose level themselves affect each other (crossly related)

summary(step.fit.bic)  # bmi,age, heart disease, gender:Male positively affect glucose level
# hypertension and glucose level itself is crossly related

# Now let us create a model to predict the probability of stroke occurrence
step.fit.aic.stroke = step(modelStroke, direction = "both", trace = 0)
summary(step.fit.aic.stroke)
step.fit.bic.stroke = step(modelOverall, k=log(nrow(trainData)), direction = "both", trace = 0)
summary(step.fit.bic.stroke)


# Now let us create a model to predict the probability of stroke occurrence
modelStrokeFinal = glm(stroke ~ age + hypertension + avg_glucose_level + hypertension*bmi + avg_glucose_level*bmi, data = trainData, family="binomial")
summary(modelStrokeFinal)

#Regularization

#Predictions
predictStroke = predict(modelStrokeFinal, data=testData, type="response") 
summary(predictStroke)


#Error Measurement


# Standardized
x <- model.matrix(stroke ~ gender + age + hypertension + heart_disease + Residence_type + 
                      avg_glucose_level + smoking_status + bmi, data)[ , -1]  # to discard the intercept
y <- data$stroke

dim(x)


# method 2: use a random set of numbers
set.seed(123)
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]

# lambda's to try out
grid <- 10^seq(10, -2, length = 100)

ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)  #coef(), predict() in glmnet package
# ridge.mod is the trained ridge regression model
# x and y must be pass separately, cannot do y ~ x.
# alpha=0: fit a ridge regression model
# alpha=1: fit a lasso model
# 0 < alpha < 1: fit an elastic net model (covered in this syllabus)

dim(coef(ridge.mod)) 
# coef(): returns a vector of ridge regression coefficients associated with each lambda value, stored in a matrix
# 20 rows: one for each predictors plus an intercept; 
# 100 columns: one for each lambda value in the range of grid

plot(ridge.mod, xvar = "lambda", main = "Ridge penalty")
# ridge regression coefficients approaching "close to zero" as lambda grows from 0 --> infinity

## Cross-validation (Scientific)

# Perform a 10-fold cross-validation to select lambda 

set.seed(123)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0) # 10-fold cv be default
cv.out
#type.measure="mse" if regression
#type.measure="deviance"
#family="gaussian" if linear regression
#family="binomial" if logistic regression

plot(cv.out, main = "MSE of a range of 10-fold cv-lambda") 
# choose (optimal) cv-lambda that has the lowest (minimum) MSE

bestlam <- cv.out$lambda.min # unstandardized best cv-lambda value
# cross-check: log(cv.out$lambda.min) - see if this value is within the range of min MSE on the plot

ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ]) # manually set s 
mean((ridge.pred - y.test)^2) # test MSE of best cv-lambda


# refit a final ridge regression model 
## using the best cv-lambda value with the smallest MSE

final.mod <- glmnet(x, y, alpha = 0)
predict(final.mod, type = "coefficients", s = bestlam)[1:10, ]  #check out the ridge coefficient estimates


# 6.6.2 The Lasso
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)    # alpha = 1: lasso
plot(lasso.mod)

# Select the best tuning paramater for lasso: cv-lambda with lowest MSE
set.seed(123)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
mean((lasso.pred - y.test)^2)  #MSE = 0.04894854

out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients", s = bestlam)[1:10, ]
lasso.coef
lasso.coef[lasso.coef != 0]  #check out variables remaining in the model (with non-zero estimated coefficients)


