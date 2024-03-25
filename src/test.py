# This file computes the output confusion matrix of the OAR model.
# This file is also used to output the comparison between OAR and
# the scikit-learn library model.
from oarUtils import logLossFunc, predictSigmoid
from train import train
import numpy as np
import json
from emnist import extract_training_samples, extract_test_samples

REG = 0.0001                                     # Regularization Parameter
ALPHA = 0.0001                                   # Learning Rate
EPOCHS = 5                                      # Number of training epochs
MODEL_DATA = ["weights", "bias", "performance"]  # Key for JSON model record


# Import letter image data and labels from EMNIST dataset
imageTrain, labelTrain = extract_training_samples('letters')
imageTest, labelTest = extract_test_samples('letters')


# flatten data and match syntax
dims = imageTrain.shape[1] * imageTrain.shape[2]
dataTrain = imageTrain.reshape(imageTrain.shape[0], dims)
dataTest = imageTest.reshape(imageTest.shape[0], dims)


# Rescale data to 0 -> 1 by dividing by max pixel value (255)
dataTrain = dataTrain.astype('float32')/255
dataTest = dataTest.astype('float32')/255

# Get length of training data
N = len(dataTrain)

# Initialize the bias and weights matrices with random numbers,
# if no model is found
weights = np.ndarray((1, 26, 784))
bias = np.full(26, np.random.rand(1, 1))

# dataset = open('model.json', 'r')
# modelRecord = json.load(dataset)

# if (len(modelRecord) != 0):

#     for i in len(0, 26):
#         bias[i] = modelRecord["bias"][i]

#     for i in len(0, 26):
#         weights[0, i] = modelRecord["weights"][0, i]           
#     else:
for i in range(0, 26):
    weights[0, i] = np.random.rand(dataTrain.shape[1])

predictTrain = np.empty(len(labelTrain))
trainLoss = np.empty(len(labelTrain))
predictTest = np.empty(len(labelTest))
testLoss = np.empty(len(labelTest))

# Training and Testing of model
for i in range(0, EPOCHS):
    for j in range(N):
        label = labelTrain[j] - 1  # -1 for proper indexing
        w = weights[0, label]
        b = bias[label]
        w, b = train(dataTrain[j], labelTrain[j], w, b, REG, ALPHA, N)
        weights[0, label] = w
        bias[label] = b

    for j in range(len(labelTrain)):
        label = labelTrain[j] - 1  # -1 for proper indexing
        w = weights[0, label]
        b = bias[label]
        yHat = predictSigmoid(dataTrain[j], w, b)
        predictTrain[j] = yHat
        # We store the loss values in an array,
        # yTrue is always 1, as we know the value of the label
        # will always be true
        trainLoss[j] = logLossFunc(1, predictTrain[j]/(j+1))
   
    for j in range(len(labelTest)):
        label = labelTest[j] - 1  # -1 for proper indexing
        w = weights[0, label]
        b = bias[label]
        yHat = predictSigmoid(dataTest[j], w, b)
        predictTest[j] = yHat
        # We store the loss values in an array,
        # yTrue is always 1, as we know the value of the label
        # will always be true
        testLoss[j] = logLossFunc(1, predictTest[j]/(j+1))

    print("----- EPOCH %d COMPLETE -----" % i)

# Calculate performance...
    
# Write model to model.json record

model = open('model.json', 'w')

format = {}
format['weights'] = weights.tolist()
format['bias'] = bias.tolist()
# format['performance'] = performance.tolist()
json_data = json.dumps(format)

model.write(json_data)
    