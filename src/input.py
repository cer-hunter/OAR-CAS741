import cv2
import os
import sys
sys.path.insert(0, "../src/")
from src.preprocess import preprocess


# Prompt for input done by GUI, this verifies and processes image
def input(imgPath):

    # check file is correct and contains an image of .BMP, .JPG, .PNG
    if (not os.path.isfile(imgPath)):
        return Exception, Exception
    ext = os.path.splitext(imgPath)[-1].lower()
    if (ext != '.png' and ext != '.jpg' and ext != '.bmp'):
        return Exception, Exception

    baseImg = cv2.imread(imgPath)

    # first check size/shape is within bounds
    height = baseImg.shape[0]
    width = baseImg.shape[1]
    if ((height > 4096) or (width > 4096)):
        return Exception, Exception
    elif ((height < 20) or (width < 20)):
        return Exception, Exception
    else:
        # now process image
        dispImg, inputImg = preprocess(baseImg)
    return dispImg, inputImg
