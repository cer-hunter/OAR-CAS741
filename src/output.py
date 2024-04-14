import sys
sys.path.insert(0, "../src/")
# How I have to do imports to properly link the test profile
try:
    from src.input import input
    from src.classify import classify
except ModuleNotFoundError:
    from input import input
    from classify import classify


def output(imgPath):
    try:
        dispImg, inputImg = input(imgPath)
        resultLabel, resultConf = classify(inputImg)
        # Change probability to out of 100% and type to string
        resultConf = str(round(resultConf*100, 2))
        return resultLabel, resultConf, dispImg
    except Exception:
        raise ValueError
