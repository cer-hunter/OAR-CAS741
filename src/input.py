import cv2
import os
from preprocess import preprocess


# Prompt for input done by GUI, this verifies and processes image
def input(imgPath):

    # check file is correct and contains an image of .BMP, .JPG, .PNG
    if (not os.path.isfile(imgPath)):
        raise ValueError("This is not a proper image file")
    ext = os.path.splitext(imgPath)[-1].lower()
    if (ext != '.png' and ext != '.jpg' and ext != '.bmp'):
        raise ValueError("This is not a JPG, BMP or PNG image file")

    baseImg = cv2.imread(imgPath)

    # first check size/shape is within bounds
    height = baseImg.shape[0]
    width = baseImg.shape[1]
    if ((height > 4096) or (width > 4096)):
        raise ValueError("This image is too large")
    elif ((height < 20) or (width < 20)):
        raise ValueError("This image is too small")
    else:
        # now process image
        dispImg, inputImg = preprocess(baseImg)
    return dispImg, inputImg
