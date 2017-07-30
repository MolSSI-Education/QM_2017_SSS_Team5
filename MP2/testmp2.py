"""
testing for the mp2new.py module
"""

import project as pj
import pytest

#Using `test_` as function so pytest can recognize.
def test_add():
    assert pj.math.add(5,2) == 7
    assert pj.math.add(2,5) == 7

testdata = [
    (2, 5, 10),
    (1, 2, 2),
    (11, 9, 99),
    (0, 0, 0)
]
@pytest.mark.parametrize("a, b, expected", testdata)
def test_mult(a, b, expected):
    assert pj.math.mult(a, b) == expected
assert pj.math.mult(b, a) == expected
