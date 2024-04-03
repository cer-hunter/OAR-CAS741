from sklearn import metrics as mtr
import matplotlib.pyplot as plt


def confMatrix(yTrue, yPred):
    # Generate confusion matrix by overall samples
    cm = mtr.confusion_matrix(yTrue/yTrue, yPred)
    cmPlot = mtr.ConfusionMatrixDisplay(cm)

    cmPlot.plot()
    plt.show()
    return cm
