import numpy as np
from oarUtils import predictSigmoid
import json

LABEL_NUM = 26  # Number of labels
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def classify(inputImg):

    # Retrieve OAR Model from record
    record = open('model.json', 'r')
    model = json.load(record)

    if (len(model) != 0):
        weights = np.asarray(model["weights"])
        bias = np.asarray(model["bias"])
    else:
        print("No classification model found")
        return "N/A", 0

    prediction = np.empty(LABEL_NUM)
    for i in range(LABEL_NUM):
        w = weights[0, i]
        b = bias[i]
        yHat = predictSigmoid(inputImg, w, b)
        prediction[i] = yHat
    bestLbl = (np.abs(prediction - 1)).argmin()  # Finds the best prediction
    probability = prediction[bestLbl]
    label = LABELS[bestLbl]

    return label, probability
