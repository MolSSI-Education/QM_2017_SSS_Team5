"""
MP2
"""
import time
import numpy as np
import psi4


def mp2(mol, basis="aug-cc-pvdz", ):

    print('\nStarting MP2...')
    # Memory for Psi4 in GB
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)

    # Memory for numpy in GB
    numpy_memory = 2

    psi4.set_options({'basis': basis,
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})

    # First compute SCF energy using Psi4
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    # Grab data from wavfunction class
    ndocc = wfn.nalpha()
    nmo = wfn.nmo()
    SCF_E = wfn.energy()
    eps = np.asarray(wfn.epsilon_a())

    # Compute size of ERI tensor in GB
    ERI_Size = (nmo ** 4) * 8e-9
    print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
    memory_footprint = ERI_Size * 2.5
    if memory_footprint > numpy_memory:
        clean()
        raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                        limit of %4.2f GB." % (memory_footprint, numpy_memory))


    # Integral generation from Psi4's MintsHelper
    mints = psi4.core.MintsHelper(wfn.basisset())
    Co = wfn.Ca_subset("AO", "OCC")
    Cv = wfn.Ca_subset("AO", "VIR")
    MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))

    Eocc = eps[:ndocc]
    Evirt = eps[ndocc:]

    e_denom = 1 / (Eocc.reshape(-1, 1, 1, 1) - Evirt.reshape(-1, 1, 1) + Eocc.reshape(-1, 1) - Evirt)

    # Get the two spin cases
    MP2corr_OS = np.einsum('iajb,iajb,iajb->', MO, MO, e_denom)
    MP2corr_SS = np.einsum('iajb,iajb,iajb->', MO - MO.swapaxes(1, 3), MO, e_denom)

    MP2corr_E = MP2corr_SS + MP2corr_OS
    MP2_E = SCF_E + MP2corr_E

    print("MP2 Energy is : ",MP2_E)

    return MP2_E

if __name__ == "__main__":
    import psi4
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """)

    MP2_E = mp2(mol, "aug-cc-pvdz")
    print("Comparing MP2 energy with psi4...")
    psi4.energy('MP2')
    a = psi4.compare_values(psi4.core.get_variable('MP2 TOTAL ENERGY'), MP2_E, 6, 'MP2 Energy')
    print(a)


