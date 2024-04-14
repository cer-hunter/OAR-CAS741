import sys
import pytest
import numpy as np
sys.path.insert(0, "../src/")
from src.input import input

PATH = "/tests/test-images/"

INS = [PATH+"A.png", PATH+"A.jpg", PATH+"A.bmp"]
ERR_IN = [PATH+"A.pdf", PATH+"Empty", ""]
SIZES = [PATH+"toobig.png", PATH+"big.png",
         PATH+"small.png", PATH+"toosmall.png"]
COLORS = [PATH+"A.png", PATH+"A_Blue.png", PATH+"A_Green.png",
          PATH+"A_Red.png", PATH+"A_Gray.png"]
LETTERS = [PATH+"A.jpg", PATH+"B.jpg", PATH+"C.jpg", PATH+"D.jpg",
           PATH+"E.jpg", PATH+"F.jpg", PATH+"G.jpg", PATH+"H.jpg",
           PATH+"I.jpg", PATH+"J.jpg", PATH+"K.jpg", PATH+"L.jpg",
           PATH+"M.jpg", PATH+"N.jpg", PATH+"O.jpg", PATH+"P.jpg",
           PATH+"Q.jpg", PATH+"R.jpg", PATH+"S.jpg", PATH+"T.jpg",
           PATH+"U.jpg", PATH+"V.jpg", PATH+"W.jpg", PATH+"X.jpg",
           PATH+"Y.jpg", PATH+"Z.jpg"]
ANGLES = [PATH+"angle0.png", PATH+"angle1p.png", PATH+"angle1n.png",
          PATH+"angle2p.png", PATH+"angle2n.png",
          PATH+"angle5p.png", PATH+"angle5n.png",
          PATH+"angle15p.png", PATH+"angle15n.png",
          PATH+"angle45p.png", PATH+"angle45n.png",
          PATH+"angle90p.png", PATH+"angle90n.png",
          PATH+"angle135p.png", PATH+"angle135n.png", PATH+"angle180.png"]
ERR_OUT = [PATH+"ABC.jpg", PATH+"Blank.jpg"]


# System Tests
def returnInput(path):
    return input(path)


@pytest.mark.set1
def test_input_format():
    for i in range(INS):
        [result1, result2] = returnInput(INS[i])
        assert isinstance([result1, result2], [np.ndarray, np.ndarray])


# def test_input_exception():
#     for i in ERR_IN:
#         pytest.raises(ValueError, input, ERR_IN[i])

# Unit Tests
