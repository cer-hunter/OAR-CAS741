import sys
from input import input
from classify import classify
sys.path.insert(0, "../OAR-CAS741/src/")


def output(imgPath):
    try:
        dispImg, inputImg = input(imgPath)
        resultLabel, resultConf = classify(inputImg)
        # Change probability to out of 100% and type to string
        resultConf = str(round(resultConf*100, 2))
        return resultLabel, resultConf, dispImg
    except Exception:
        return Exception, Exception, Exception
