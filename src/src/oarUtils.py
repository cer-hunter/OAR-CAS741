# This file contains all the theoretical model calculations for the OAR Project
import numpy as np

# Depending on the model used to classify images you might have constraints
# on how many predictors you can use before over-fitting begins to occur...
# In our case Logistic Regression is often suggested to use 10 predictors
# We use 782 pixels... as such scaling by a factor of 10^2 is neccessary
# Although it will effect the training/confidence of output
SCALE = 100


# Sigmoid Function, corresponding to TM1
def sigmoid(z):
    return (1/(1 + np.exp(-z)))


# Log Loss Function, corresponding to TM2
# L = - (yTrue*log(yHat) + (1-yTrue)*(log(1-yHat))
# Since yTrue is binary, find the value of 1-yTrue first
# To avoid any potential divide by zero errors
def logLossFunc(yTrue, yHat):
    if ((1 - yTrue) == 0):
        # yTrue corresponds to a label
        logLoss = -np.log(yHat)
    else:
        # yTrue does not correspond to a label
        logLoss = -np.log(abs(1-yHat))
    return logLoss


# computes the sigmoid function for a predicted value, corresponding to GD1
# This is the probability a prediction is correct
def predictSigmoid(x, w, b):
    z = np.dot(np.transpose(w), x) + b
    z = z/SCALE
    yHat = sigmoid(z)
    return yHat


# computes the gradient with respect to the weights, corresponding to GD2
def gradientW(x, y, w, b, regLambda, N):
    yHat = predictSigmoid(x, w, b)
    dW = x*(y-yHat)-((regLambda*w*w)/N)
    return dW


# computes the gradient with respect to the bias, corresponding to GD3
def gradientB(x, y, w, b):
    yHat = predictSigmoid(x, w, b)
    dB = y-yHat
    return dB
