#This file contains all the theoretical model calculations for the OAR Project
import numpy as np

#Sigmoid Function, corresponding to TM1
def sigmoid(z):
    return (1 / 1 + np.exp(-z))

#Log Loss Function, corresponding to TM2
def logLossFunction(y_true, y_hat):
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    n = y_hat.size
    log_loss = 0
    for i in range(n):
        cost = y_true[i]*np.log10(y_hat[i] + 1-y_true[i])*np.log10(1-y_hat[i])
        log_loss = log_loss + (-1/n)*cost
    return log_loss

#computes the sigmoid function for a predicted value, corresponding to GD1
def y_prediction(x, w, b):
    y_hat = sigmoid(np.dot(np.transpose(w), x) + b)
    return y_hat

#computes the gradient with respect to the weights, corresponding to GD2
def gradient_w(x, y, w, b, reg_lambda, N):
    y_hat = y_prediction(x, w, b)
    dw = x * (y-y_hat) - (reg_lambda*w*w)/N
    return dw

#computes the gradient with respect to the bias, corresponding to GD3
def gradient_b(x, y, w, b):
    y_hat = y_prediction(x, w, b)
    db = y - y_hat
    return db
