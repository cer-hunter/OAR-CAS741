# This file contains all the theoretical model calculations for the OAR Project
import numpy as np


# Sigmoid Function, corresponding to TM1
def sigmoid(z):
    return (1 / 1 + np.exp(-z))


# Log Loss Function, corresponding to TM2
def logLossFunc(yTrue, yHat):
    logLoss = 0
    cost = yTrue*np.log10(yHat) + (1-yTrue)*np.log10(1-yHat)
    logLoss = logLoss + (-1)*cost
    return logLoss


# computes the sigmoid function for a predicted value, corresponding to GD1
def predictSigmoid(x, w, b):
    yHat = sigmoid(np.dot(np.transpose(w), x) + b)
    return yHat


# computes the gradient with respect to the weights, corresponding to GD2
def gradientW(x, y, w, b, regLambda, N):
    yHat = predictSigmoid(x, w, b)
    dW = np.transpose(x) * (yHat - y) - (regLambda*w*w)/N
    return dW


# computes the gradient with respect to the bias, corresponding to GD3
def gradientB(x, y, w, b):
    yHat = predictSigmoid(x, w, b)
    dB = yHat - y
    return dB
