import sys
import pytest
import numpy as np
from os.path import join, dirname
sys.path.insert(0, "../src/")
from src.input import input
from src.output import output
from src.oarTrain import train
# from src.oarUtils import sigmoid, logLossFunc, predictSigmoid
# from src.oarUtils import gradientW, gradientB

PATH = join(dirname(__file__), "test-images")

INS = [join(PATH, "A.png"), join(PATH, "A.jpg"), join(PATH, "A.bmp")]
ERR_IN = [join(PATH, "A.pdf"), join(PATH, "Empty"), join(PATH, "None")]
COLORS = [join(PATH, "A.png"), join(PATH, "A_Blue.png"),
          join(PATH, "A_Green.png"), join(PATH, "A_Red.png"),
          join(PATH, "A_Gray.png")]
SIZES = [join(PATH, "toobig.png"), join(PATH, "big.png"),
         join(PATH, "small.png"), join(PATH, "toosmall.png")]
LETTERS = [join(PATH, "A.jpg"), join(PATH, "B.jpg"), join(PATH, "C.jpg"),
           join(PATH, "D.jpg"), join(PATH, "E.jpg"), join(PATH, "F.jpg"),
           join(PATH, "G.jpg"), join(PATH, "H.jpg"), join(PATH, "I.jpg"),
           join(PATH, "J.jpg"), join(PATH, "K.jpg"), join(PATH, "L.jpg"),
           join(PATH, "M.jpg"), join(PATH, "N.jpg"), join(PATH, "O.jpg"),
           join(PATH, "P.jpg"), join(PATH, "Q.jpg"), join(PATH, "R.jpg"),
           join(PATH, "S.jpg"), join(PATH, "T.jpg"), join(PATH, "U.jpg"),
           join(PATH, "V.jpg"), join(PATH, "W.jpg"), join(PATH, "X.jpg"),
           join(PATH, "Y.jpg"), join(PATH, "Z.jpg")]
DEGEN = [join(PATH, "ABC.jpg"), join(PATH, "hunter_ok.jpg"),
         join(PATH, "Blank.jpg")]
ANGLES = [join(PATH, "angle0.png"),
          join(PATH, "angle1p.png"), join(PATH, "angle1n.png"),
          join(PATH, "angle2p.png"), join(PATH, "angle2n.png"),
          join(PATH, "angle5p.png"), join(PATH, "angle5n.png"),
          join(PATH, "angle15p.png"), join(PATH, "angle15n.png"),
          join(PATH, "angle45p.png"), join(PATH, "angle45n.png"),
          join(PATH, "angle90p.png"), join(PATH, "angle90n.png"),
          join(PATH, "angle135p.png"), join(PATH, "angle135n.png"),
          join(PATH, "angle180.png")]


# System Tests

# Checks valid inputs return valid outputs
@pytest.mark.vnv
def test_input_format():
    for i in range(len(INS)):
        result1, result2 = input(INS[i])
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)


# Check ValueError is thrown when inputs of invalid formats are entered
@pytest.mark.vnv
def test_invalid_input_format():
    for i in range(len(ERR_IN)):
        with pytest.raises(ValueError):
            input(ERR_IN[i])


# Check that images of different colours are able to be processed
@pytest.mark.vnv
def test_input_colors():
    for i in range(len(COLORS)):
        result1, result2 = input(COLORS[i])
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)


# Check that images of different sizeare able to be processed
# And that images outside the specified range throw a ValueError
@pytest.mark.vnv
def test_input_size():
    for i in range(len(SIZES)):
        if (i == 0 or i == 3):
            with pytest.raises(ValueError):
                input(SIZES[i])
        else:
            result1, result2 = input(SIZES[i])
            assert isinstance(result1, np.ndarray)
            assert isinstance(result2, np.ndarray)


# Checks that each label can be successfully classified
# Note: Does not care about performance, just whether an
# output is produced.
@pytest.mark.vnv
def test_output_labels():
    for i in range(len(LETTERS)):
        result1, result2, result3 = output(LETTERS[i])
        # Print the result for the test report
        print(result1, result2)
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, np.ndarray)


# Checks that degenerate input cases return "NOT CLASSIFIED"
@pytest.mark.vnv
def test_output_degen():
    for i in range(len(DEGEN)):
        result1, result2, result3 = output(DEGEN[i])
        assert result1 == "NOT CLASSIFIED"
        assert isinstance(result2, str)
        assert isinstance(result3, np.ndarray)


# Checks that degenerate input cases of different orientations
# of letters. Note: Technically it is assumed these will not
# be inputs to the program, but testing just because they are
# technically valid inputs the program will accept.
@pytest.mark.vnv
def test_output_angles():
    for i in range(len(ANGLES)):
        result1, result2, result3 = output(ANGLES[i])
        # Print to see what the results might be for the report.
        print(result1, result2)
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, np.ndarray)


# Unit Tests
