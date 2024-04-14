# This file computes the OAR model and outputs it's performance.
import numpy as np
import json
from emnist import extract_training_samples, extract_test_samples
from oarUtils import logLossFunc, predictSigmoid
from metrics import confMatrix, confMatrixLabels, lossGraph
from oarTrain import train

PX_VAL = 255
REG = 0.0001                       # Regularization Parameter (for overfitting)
ALPHA = 0.0001                     # Learning Rate
EPOCHS = 500                       # Number of training epochs
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LABEL_NUM = 26                     # Number of Labels
EPSILON = 0.50                     # Acceptable cutoff for confidence
RESET = False                      # Constant used to reset model

try:
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
except Exception:
    print("Error Initializing Dataset")

# Initialize prediction matrices
predictTrain = np.empty(len(labelTrain))
predictTest = np.empty(len(labelTest))
predict = np.empty(LABEL_NUM)
# Initialize Loss Matrices
trainLoss = np.zeros(EPOCHS)

# Training and Testing of model
for i in range(EPOCHS):
    for j in range(N):
        try:
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
            # Get training loss...
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
            loss = logLossFunc(1, yHat)
            trainLoss[i] = trainLoss[i] + loss
        except Exception:
            print("Error Training Model")
            break
    # Here we test how good the model is overall...
    # Similar to the end product
    for j in range(len(labelTest)):
        try:
            lblTrue = labelTest[j]-1  # -1 for proper indexing
            for k in range(LABEL_NUM):
                w = weights[0, k]
                b = bias[k]
                yHat = predictSigmoid(dataTest[j], w, b)
                predict[k] = yHat
            bestLbl = np.argmax(predict)  # Finds the best prediction index
            if (bestLbl == lblTrue):
                predictTest[j] = 1
            else:
                predictTest[j] = 0
        except Exception:
            print("Error Testing Model")
            break
    print("----- EPOCH %d COMPLETE -----" % (i+1))

try:
    # Get Loss Graph
    lossGraph(trainLoss, EPOCHS)

    # Calculate performance metrics
    trPerformance = confMatrix(labelTrain, predictTrain)
    performance = confMatrix(labelTest, predictTest)

    # Calculates performance per label
    modelPerformance = []
    for i in range(LABEL_NUM):
        modelTestLabels = np.empty(len(labelTest))
        modelTest = np.empty(len(labelTest))
        for j in range(len(labelTest)):
            if (labelTest[j]-1 == i):
                modelTestLabels[j] = 1
            else:
                modelTestLabels[j] = 0
            w = weights[0, i]
            b = bias[i]
            yHat = predictSigmoid(dataTest[j], w, b)
            if yHat > EPSILON:
                modelTest[j] = 1
            else:
                modelTest[j] = 0
        lblTitle = LABELS[i] + " Test"
        lblMatrix = confMatrixLabels(modelTestLabels, modelTest, lblTitle).tolist()
        modelPerformance.append(lblMatrix)
except Exception:
    print("Error calculating model performance")


# Write model to model.json record
try:
    model = open('model.json', 'w')

    format = {}
    format['weights'] = weights.tolist()
    format['bias'] = bias.tolist()
    format['train data performance'] = trPerformance.tolist()
    format['test data performance'] = performance.tolist()
    format['Per Label Test performance'] = modelPerformance
    json_data = json.dumps(format)

    model.write(json_data)
except Exception:
    print("Error writing model to .json file")
