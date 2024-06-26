import sys
import pytest
import numpy as np
from os.path import join, dirname
sys.path.insert(0, "../src/")
from src.input import input
from src.output import output
from src.oarTrain import train
from src.oarUtils import sigmoid, logLossFunc, predictSigmoid
from src.oarUtils import gradientW, gradientB


# Test Inputs for System Tests
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
def test_input_format(record_testsuite_property):
    record_testsuite_property = ("id", "T-1")
    for i in range(len(INS)):
        result1, result2 = input(INS[i])
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
    print("---------- TEST PASS ----------")


# Check ValueError is thrown when inputs of invalid formats are entered
@pytest.mark.vnv
def test_invalid_input_format(record_testsuite_property):
    record_testsuite_property = ("id", "T-2")
    for i in range(len(ERR_IN)):
        with pytest.raises(ValueError):
            input(ERR_IN[i])
        with pytest.raises(ValueError):
            output(ERR_IN[i])
    print("---------- TEST PASS ----------")


# Check that images of different colours are able to be processed
@pytest.mark.vnv
def test_input_colors(record_testsuite_property):
    record_testsuite_property = ("id", "T-3")
    for i in range(len(COLORS)):
        result1, result2 = input(COLORS[i])
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
    print("---------- TEST PASS ----------")


# Check that images of different sizeare able to be processed
# And that images outside the specified range throw a ValueError
@pytest.mark.vnv
def test_input_size(record_testsuite_property):
    record_testsuite_property = ("id", "T-4")
    for i in range(len(SIZES)):
        if (i == 0 or i == 3):
            with pytest.raises(ValueError):
                input(SIZES[i])
        else:
            result1, result2 = input(SIZES[i])
            assert isinstance(result1, np.ndarray)
            assert isinstance(result2, np.ndarray)
    print("---------- TEST PASS ----------")


# Checks that each label can be successfully classified
# Note: Does not care about performance, just whether an
# output is produced.
@pytest.mark.vnv
def test_output_labels(record_testsuite_property):
    record_testsuite_property = ("id", "T-5")
    for i in range(len(LETTERS)):
        result1, result2, result3 = output(LETTERS[i])
        # Print the result for the test report
        print(result1, result2)
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, np.ndarray)
    print("---------- TEST PASS ----------")


# Checks that degenerate input cases return "NOT CLASSIFIED"
@pytest.mark.vnv
def test_output_degen(record_testsuite_property):
    record_testsuite_property = ("id", "T-6")
    for i in range(len(DEGEN)):
        result1, result2, result3 = output(DEGEN[i])
        assert result1 == "NOT CLASSIFIED"
        assert isinstance(result2, str)
        assert isinstance(result3, np.ndarray)
    print("---------- TEST PASS ----------")


# Checks that degenerate input cases of different orientations
# of letters. Note: Technically it is assumed these will not
# be inputs to the program, but testing just because they are
# technically valid inputs the program will accept.
@pytest.mark.vnv
def test_output_angles(record_testsuite_property):
    record_testsuite_property = ("id", "T-7")
    for i in range(len(ANGLES)):
        result1, result2, result3 = output(ANGLES[i])
        # Print to see what the results might be for the report.
        print(result1, result2)
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, np.ndarray)
    print("---------- TEST PASS ----------")


# Test Inputs for Unit Tests
STRING = "test"
V_INT = 1
V_INT_2 = 0
V_FLOAT = 0.5
INV_INT = 2
INV_FLOAT = -5.4
MATRIX_1 = np.ones(3)
MATRIX_2 = np.zeros((2, 2))


# Unit Tests


# Testing sigmoid function works properly
@pytest.mark.vnv
def test_sigmoid(record_testsuite_property):
    record_testsuite_property = ("id", "UT-1")
    assert sigmoid(V_INT_2) == 0.5
    with pytest.raises(ValueError):
        sigmoid(STRING)
    print("---------- TEST PASS ----------")


# Testing logLossFunc function works properly
@pytest.mark.vnv
def test_logLossFunc(record_testsuite_property):
    record_testsuite_property = ("id", "UT-2")
    res1 = logLossFunc(V_INT, V_INT)
    res2 = logLossFunc(V_INT_2,  V_INT_2)
    assert res1 == 0
    assert res2 == 0
    with pytest.raises(ValueError):
        logLossFunc(V_INT, INV_FLOAT)
    with pytest.raises(ValueError):
        logLossFunc(INV_INT, V_FLOAT)
    with pytest.raises(ValueError):
        logLossFunc(STRING, STRING)
    print("---------- TEST PASS ----------")


# Testing predictSigmoid function works properly
@pytest.mark.vnv
def test_predict(record_testsuite_property):
    record_testsuite_property = ("id", "UT-3")
    res = predictSigmoid(MATRIX_1, MATRIX_1, V_INT)
    assert isinstance(res, float)
    with pytest.raises(ValueError):
        predictSigmoid(MATRIX_1, MATRIX_2, V_INT)
    with pytest.raises(ValueError):
        predictSigmoid(STRING, STRING, V_INT)
    with pytest.raises(ValueError):
        predictSigmoid(MATRIX_1, MATRIX_1, STRING)
    print("---------- TEST PASS ----------")


# Testing gradientW function works properly
@pytest.mark.vnv
def test_gradientW(record_testsuite_property):
    record_testsuite_property = ("id", "UT-4")
    res = gradientW(MATRIX_1, V_INT, MATRIX_1, V_INT, V_INT, V_INT)
    assert isinstance(res, np.ndarray)
    with pytest.raises(ValueError):
        gradientW(MATRIX_1, STRING, MATRIX_1, V_INT, V_INT, V_INT)
    with pytest.raises(ValueError):
        gradientW(MATRIX_1, INV_INT, MATRIX_1, V_INT, V_INT, V_INT)
    with pytest.raises(ValueError):
        gradientW(MATRIX_1, V_INT, MATRIX_1, V_INT, STRING, V_INT)
    with pytest.raises(ValueError):
        gradientW(MATRIX_1, V_INT, MATRIX_1, V_INT, V_INT, STRING)
    print("---------- TEST PASS ----------")


# Testing gradientB function works properly
@pytest.mark.vnv
def test_gradientB(record_testsuite_property):
    record_testsuite_property = ("id", "UT-5")
    res = gradientB(MATRIX_1, V_INT, MATRIX_1, V_INT)
    assert isinstance(res, float)
    with pytest.raises(ValueError):
        gradientB(MATRIX_1, STRING, MATRIX_1, V_INT)
    with pytest.raises(ValueError):
        gradientB(MATRIX_1, INV_INT, MATRIX_1, V_INT)
    print("---------- TEST PASS ----------")


# Testing train function works properly
@pytest.mark.vnv
def test_train(record_testsuite_property):
    record_testsuite_property = ("id", "UT-6")
    r1, r2 = train(MATRIX_1, V_INT, MATRIX_1, V_INT, V_FLOAT, V_FLOAT, V_INT)
    assert isinstance(r1, np.ndarray)
    assert isinstance(r2, float)
    with pytest.raises(ValueError):
        train(MATRIX_1, V_INT, MATRIX_1, V_INT, V_FLOAT, STRING, V_INT)
    print("---------- TEST PASS ----------")
