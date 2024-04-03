# This file computes the OAR model and outputs it's performance.

from oarUtils import logLossFunc, predictSigmoid
from metrics import confMatrix
from train import train
import numpy as np
import json
from emnist import extract_training_samples, extract_test_samples

PX_VAL = 255
REG = 0.0001                       # Regularization Parameter (for overfitting)
ALPHA = 0.0001                     # Learning Rate
EPOCHS = 500                       # Number of training epochs
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LABEL_NUM = 26                     # Number of Labels
EPSILON = 0.50                     # Used to improve training
RESET = False                      # Constant used to reset model

# Import letter image data and labels from EMNIST dataset
imageTrain, labelTrain = extract_training_samples('letters')
imageTest, labelTest = extract_test_samples('letters')


# flatten data and match syntax
dims = imageTrain.shape[1] * imageTrain.shape[2]
dataTrain = imageTrain.reshape(imageTrain.shape[0], dims)
dataTest = imageTest.reshape(imageTest.shape[0], dims)


# Rescale data to 0 -> 1 by dividing by max pixel value (255)
dataTrain = dataTrain.astype('float32')/PX_VAL
dataTest = dataTest.astype('float32')/PX_VAL

# Get length of training data
N = len(dataTrain)

# Initialize the bias and weights matrices with ones,
# if no model is found
dataset = open('model.json', 'r')
modelRecord = json.load(dataset)
# Checks if there is a pre-existing set of weights and biases and not reset
if (len(modelRecord) != 0) and not RESET:
    weights = np.array(modelRecord["weights"])
    bias = np.array(modelRecord["bias"])
# Generates random weights and biases otherwise
else:
    weights = np.ndarray((1, 26, 784))
    for i in range(LABEL_NUM):
        weights[0, i] = np.random.random(dataTrain.shape[1])
    bias = np.random.random(26)

# Initialize prediction and performance matrices
predictTrain = np.empty(len(labelTrain))
trainLoss = np.empty(len(labelTrain))
predicts = np.empty(LABEL_NUM)
predictTest = np.empty(len(labelTest))
testLoss = np.empty(len(labelTest))

# Training and Testing of model
for i in range(EPOCHS):
    for j in range(N):
        # Find the label of training data
        lblTrue = labelTrain[j]-1  # -1 for proper indexing
        # Train every letter model on the data
        for label in range(LABEL_NUM):
            w = weights[0, label]
            b = bias[label]
            # Binary classification if correct label send 1 else send 0
            if (label == lblTrue):
                w, b = train(dataTrain[j], 1, w, b, REG, ALPHA, N)
            else:
                w, b = train(dataTrain[j], 0, w, b, REG, ALPHA, N)
            weights[0, label] = w
            bias[label] = b
        # Here we test how good our model is per label
        w = weights[0, lblTrue]
        b = bias[lblTrue]
        yHat = predictSigmoid(dataTrain[j], w, b)
        # Want model to be over EPSILON % confident in output
        if (yHat > EPSILON):
            predictTrain[j] = 1
        else:
            predictTrain[j] = 0
        # We store the loss values in an array,
        # yTrue is always 1, as we're using a binary classifier
        # and there are no blank images in the dataset used
        trainLoss[j] = logLossFunc(labelTrain[j]/labelTrain[j], yHat)

    # Here we test how good the model is overall...
    # Similar to what the end product might be
    for j in range(len(labelTest)):
        for k in range(LABEL_NUM):
            w = weights[0, k]
            b = bias[k]
            yHat = predictSigmoid(dataTest[j], w, b)
            predicts[k] = yHat
        bestLbl = (np.abs(predicts - 1)).argmin()  # Finds the best prediction
        if (bestLbl == labelTest[j]):
            predictTest[j] = 1
        else:
            predictTest[j] = 0
        testLoss[j] = logLossFunc(labelTest[j]/labelTest[j], predicts[bestLbl])
    print("----- EPOCH %d COMPLETE -----" % (i+1))

# Calculate performance metrics
trPerf = confMatrix(labelTrain, predictTrain)
performance = confMatrix(labelTest, predictTest)


# Write model to model.json record

model = open('model.json', 'w')

format = {}
format['weights'] = weights.tolist()
format['bias'] = bias.tolist()
format['train data performance'] = trPerf.tolist()
format['test data performance'] = performance.tolist()
json_data = json.dumps(format)

model.write(json_data)
