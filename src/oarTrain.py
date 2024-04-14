import sys
from oarUtils import gradientW, gradientB
sys.path.insert(0, "../OAR-CAS741/src/")


# computes the gradient descent for one epoch, corresponding to IM1
def train(x, yTrue, w, b, regLambda, alpha, N):
    if(isinstance(alpha, float)):
        dW = gradientW(x, yTrue, w, b, regLambda, N)
        dB = gradientB(x, yTrue, w, b)
        w = w + alpha*dW
        b = b + alpha*dB

        return w, b
    else:
        return ValueError, ValueError
