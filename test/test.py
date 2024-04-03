import pytest
import numpy as np
import sys
sys.path.insert(1, "~/OAR-CAS741/src")
from output import output
from input import input


def test_png_input():
    input1, input2 = input(
        "C:/Users/MasqO/OneDrive/Documents/CAS-741/OAR-CAS741/test/A.png")
    # Gives an output
    assert isinstance(input1, np.float32)
    assert isinstance(input2, np.float32)


def test_jpg_input():
    input1, input2 = input(
        "C:/Users/MasqO/OneDrive/Documents/CAS-741/OAR-CAS741/test/A.jpg")
    # Gives an output
    assert isinstance(input1, np.float32)
    assert isinstance(input2, np.float32)


def test_bmp_input():
    input1, input2 = input(
        "C:/Users/MasqO/OneDrive/Documents/CAS-741/OAR-CAS741/test/A.bmp")
    # Gives an output
    assert isinstance(input1, np.float32)
    assert isinstance(input2, np.float32)


def test_exception_no_path_input():
    with pytest.raises(Exception):
        input1, input2 = input()
