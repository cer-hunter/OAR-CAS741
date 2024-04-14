import sys
import pytest
import numpy as np
from os.path import join, dirname
sys.path.insert(0, "../src/")
from src.input import input

PATH = join(dirname(__file__), "test-images")

INS = [join(PATH, "A.png"), join(PATH, "A.jpg"), join(PATH, "A.bmp")]
ERR_IN = [join(PATH, "A.pdf"), join(PATH, "Empty"), None]
SIZES = [join(PATH, "toobig.png"), join(PATH, "big.png"),
         join(PATH, "small.png"), join(PATH, "toosmall.png")]
COLORS = [join(PATH, "A.png"), join(PATH, "A_Blue.png"),
          join(PATH, "A_Green.png"), join(PATH, "A_Red.png"),
          join(PATH, "A_Gray.png")]
LETTERS = [join(PATH, "A.jpg"), join(PATH, "B.jpg"), join(PATH, "C.jpg"),
           join(PATH, "D.jpg"), join(PATH, "E.jpg"), join(PATH, "F.jpg"),
           join(PATH, "G.jpg"), join(PATH, "H.jpg"), join(PATH, "I.jpg"),
           join(PATH, "J.jpg"), join(PATH, "K.jpg"), join(PATH, "L.jpg"),
           join(PATH, "M.jpg"), join(PATH, "N.jpg"), join(PATH, "O.jpg"),
           join(PATH, "P.jpg"), join(PATH, "Q.jpg"), join(PATH, "R.jpg"),
           join(PATH, "S.jpg"), join(PATH, "T.jpg"), join(PATH, "U.jpg"),
           join(PATH, "V.jpg"), join(PATH, "W.jpg"), join(PATH, "X.jpg"),
           join(PATH, "Y.jpg"), join(PATH, "Z.jpg")]
ANGLES = [join(PATH, "angle0.png"),
          join(PATH, "angle1p.png"), join(PATH, "angle1n.png"),
          join(PATH, "angle2p.png"), join(PATH, "angle2n.png"),
          join(PATH, "angle5p.png"), join(PATH, "angle5n.png"),
          join(PATH, "angle15p.png"), join(PATH, "angle15n.png"),
          join(PATH, "angle45p.png"), join(PATH, "angle45n.png"),
          join(PATH, "angle90p.png"), join(PATH, "angle90n.png"),
          join(PATH, "angle135p.png"), join(PATH, "angle135n.png"),
          join(PATH, "angle180.png")]
ERR_OUT = [join(PATH, "ABC.jpg"), join(PATH, "hunter_ok.jpg"),
           join(PATH, "Blank.jpg")]


# System Tests
def returnInput(path):
    return input(path)


@pytest.mark.vnv
def test_input_format():
    for i in range(len(INS)):
        result1, result2 = returnInput(INS[i])
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)


@pytest.mark.vnv
def test_invalid_input_format():
    for i in range(len(ERR_IN)):
        result1, result2 = returnInput(INS[i])
        assert isinstance(result1, ValueError)
        assert isinstance(result2, ValueError)

# Unit Tests
