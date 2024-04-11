from sklearn import metrics as mtr
import numpy as np
import matplotlib.pyplot as plt


# Generate confusion matrix by overall samples
def confMatrix(yTrue, yPred):
    cm = mtr.confusion_matrix(yTrue/yTrue, yPred)
    cmPlot = mtr.ConfusionMatrixDisplay(cm)

    cmPlot.plot()
    plt.show()
    return cm


# Generate confusion matrix by label
def confMatrixLabels(yTrue, yPred, title):
    cm = mtr.confusion_matrix(yTrue, yPred)
    cmPlot = mtr.ConfusionMatrixDisplay(cm)

    cmPlot.plot()
    plt.suptitle(title)
    plt.show()
    return cm


def lossGraph(trainLoss, epochs):
    xAxis = np.arange(1, epochs+1)
    label = ["Train Loss"]
    plt.plot(xAxis, trainLoss, label="Train Loss")
    plt.legend(label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
