# This file contains all the theoretical model calculations for the OAR Project
import numpy as np


# Sigmoid Function, corresponding to TM1
def sigmoid(z):
    return (1 / 1 + np.exp(-z))


# Log Loss Function, corresponding to TM2
def logLossFunc(yTrue, yHat):
    yTrue = np.array(yTrue)
    yHat = np.array(yHat)
    n = yHat.size
    logLoss = 0
    for i in range(n):
        cost = yTrue[i]*np.log10(yHat[i]) + (1-yTrue[i])*np.log10(1-yHat[i])
        logLoss = logLoss + (-1/n)*cost
    return logLoss


# computes the sigmoid function for a predicted value, corresponding to GD1
def predict(x, w, b):
    yHat = sigmoid(np.dot(np.transpose(w), x) + b)
    return yHat


# computes the gradient with respect to the weights, corresponding to GD2
def gradientW(x, y, w, b, reg_lambda, N):
    yHat = predict(x, w, b)
    dW = np.transpose(x) * (yHat - y) - (reg_lambda*w*w)/N
    return dW


# computes the gradient with respect to the bias, corresponding to GD3
def gradientB(x, y, w, b):
    yHat = predict(x, w, b)
    dB = yHat - y
    return dB


# computes the gradient descent for one epoch, corresponding to IM1
def gradientDescent(x, y, w, b, reg_lambda, alpha, N):
    dW = gradientW(x, y, w, b, reg_lambda, N)
    dB = gradientB(x, y, w, b)
    w = w + alpha*dW
    b = b + alpha*dB
    return w, b
