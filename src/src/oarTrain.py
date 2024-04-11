import sys
sys.path.insert(0, '../src/')
from src.oarUtils import gradientW, gradientB


# computes the gradient descent for one epoch, corresponding to IM1
def train(x, yTrue, w, b, regLambda, alpha, N):

    dW = gradientW(x, yTrue, w, b, regLambda, N)
    dB = gradientB(x, yTrue, w, b)
    w = w + alpha*dW
    b = b + alpha*dB

    return w, b
