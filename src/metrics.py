from sklearn import metrics as mtr
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


def lossGraph(testLoss, trainLoss, epochs):
    plt.plot(epochs, testLoss, label="Test Loss")
    plt.plot(epochs, trainLoss, label="Train Loss")
    plt.xlabel("Epochs")
    plt.xlabel("Loss")
    plt.show()
