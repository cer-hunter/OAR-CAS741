import pytest
import numpy as np
# from output import output
from input import input

PATH = "~/OAR-CAS741/test/test-images/"

INS = [PATH+"A.png", PATH+"A.jpg", PATH+"A.bmp"]
ERR_IN = [PATH+"A.pdf", PATH+"Empty", ""]
SIZES = []
COLORS = [PATH+"A.png", PATH+"A_Blue.png", PATH+"A_Green.png",
          PATH+"A_Red.png", PATH+"A_Gray.png"]
LETTERS = [PATH+"A.jpg", PATH+"B.jpg", PATH+"C.jpg", PATH+"D.jpg",
           PATH+"E.jpg", PATH+"F.jpg", PATH+"G.jpg", PATH+"H.jpg",
           PATH+"I.jpg", PATH+"J.jpg", PATH+"K.jpg", PATH+"L.jpg",
           PATH+"M.jpg", PATH+"N.jpg", PATH+"O.jpg", PATH+"P.jpg",
           PATH+"Q.jpg", PATH+"R.jpg", PATH+"S.jpg", PATH+"T.jpg",
           PATH+"U.jpg", PATH+"V.jpg", PATH+"W.jpg", PATH+"X.jpg",
           PATH+"Y.jpg", PATH+"Z.jpg"]
ERR_OUT = [PATH+"ABC.jpg", PATH+"Blank.jpg"]


def test_input_format():
    for i in INS:
        result1, result2 = input(INS[i])
        assert isinstance(result1, np.float32)
        assert isinstance(result2, np.float32)


def test_input_exception():
    for i in ERR_IN:
        pytest.raises(ValueError, input, ERR_IN[i])
