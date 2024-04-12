import cv2


PX_SIZE = 28  # since the input images in the data set used to train were 28x28
PX_VAL = 255  # Max pixel value for grayscale image


def preprocess(baseImg):
    # resize the image
    dispResize = cv2.resize(baseImg, (200, 200))  # Larger for display to GUI
    inputResize = cv2.resize(baseImg, (PX_SIZE, PX_SIZE))  # For classification

    # Rearrange colour channels for displaying on GUI
    b, g, r = cv2.split(dispResize)
    dispImg = cv2.merge((r, g, b))

    # Now for processing the image to be classified...
    # First convert resized image to grayscale (Normalization)
    grayImage = cv2.cvtColor(inputResize, cv2.COLOR_BGR2GRAY)
    # Flatten Image into one array
    inputImg = grayImage.flatten()
    # Change scale of image data from 0-255 to 0-1
    inputImg = inputImg.astype('float32')/PX_VAL

    return dispImg, inputImg
