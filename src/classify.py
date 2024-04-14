import numpy as np
import json
import sys
sys.path.insert(0, "../src/")
# How I have to do imports to properly link the test profile
try:
    from src.oarUtils import predictSigmoid
except ModuleNotFoundError:
    from oarUtils import predictSigmoid

LABEL_NUM = 26  # Number of labels
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
EPSILON = 0.1   # Cutoff for accuracy of predictions


def classify(inputImg):
    try:
        # Retrieve OAR Model from record
        record = open('model.json', 'r')
        model = json.load(record)

        if (len(model) != 0):
            weights = np.asarray(model["weights"])
            bias = np.asarray(model["bias"])
        else:
            print("No classification model found")
            raise Exception

        prediction = np.empty(LABEL_NUM)
        for i in range(LABEL_NUM):
            w = weights[0, i]
            b = bias[i]
            yHat = predictSigmoid(inputImg, w, b)
            prediction[i] = yHat
        # Finds the index of the best prediction
        bestLbl = np.argmax(prediction)
        # Check if predicition is above cutoff
        if prediction[bestLbl] > EPSILON:
            probability = prediction[bestLbl]
            label = "THE LETTER " + LABELS[bestLbl]
        # Otherwise give confidence that it can not be one of the labels
        else:
            probability = 1 - prediction[bestLbl]
            label = "NOT CLASSIFIED"

        return label, probability
    except Exception:
        raise ValueError
