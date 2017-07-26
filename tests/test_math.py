"""
Testing for the math.py module
This is called testing framework.
Use
python -c "import pytest"
before running this script to check if `pytest` is installed.
After scripts are done, use
`py.test -v`
"""
import mp2
import pytest

#Using `test_` as function so pytest can recognize.
def test_add():
    assert mp2.math.add(5,2) == 7
    assert mp2.math.add(2,5) == 7

testdata = [
    (2, 5, 10),
    (1, 2, 2),
    (11, 9, 99),
    (0, 0, 0)
]
@pytest.mark.parametrize("a, b, expected", testdata)
def test_mult(a, b, expected):
    assert mp2.math.mult(a, b) == expected
    assert mp2.math.mult(b, a) == expected

testdata = [
    (2, 5, 10),
    (1, 2, 2),
    (11, 9, 99),
    (0, 0, 0)
]

