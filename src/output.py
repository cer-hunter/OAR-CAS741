from input import input
from classify import classify


def output(imgPath):
    dispImg, inputImg = input(imgPath)
    resultLabel, resultConf = classify(inputImg)
    # Change probability to out of 100% and type to string
    resultConf = str(round(resultConf*100, 2))
    return resultLabel, resultConf, dispImg
