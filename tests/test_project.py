"""
Testing for the `project.py' module
This is called testing framework.
Use
python -c "import pytest"
before running this script to check if `pytest` is installed.
After scripts are done, use
`py.test -v`
"""

import pytest
import psi4
import numpy as np

# Import your moldule here
from project import scf
from project import diis
# from project import jk
# from project import soscf
# from project import mp2


#Initialize testing env
mol = psi4.geometry("""
        O
        H 1 1.1
        H 1 1.1 2 104
        """)

basis = "sto-3g"

# Parametrize mol to pass as argument.[mol] has be as type `list`.
@pytest.mark.parametrize("mol",[mol])

#Using `test_` as function so pytest can recognize.
def test_scf(mol):
    #Each method has one test.
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
    E_total = scf.scf(mol)
    assert np.allclose( E_total, psi4_energy)

@pytest.mark.parametrize("mol",[mol])
def test_diis(mol):
    #Each method has one test.
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
    E_total = diis.diis(mol)
    assert np.allclose(E_total, psi4_energy)


#def test_jk(mol):
#    pass





# # testdata = [
#     (2, 5, 10),
#     (1, 2, 2),
#     (11, 9, 99),
#     (0, 0, 0)
# ]
# @pytest.mark.parametrize("a, b, expected", testdata)
# def test_mult(a, b, expected):
#     assert pj.math.mult(a, b) == expected
#     assert pj.math.mult(b, a) == expected
