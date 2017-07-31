"""
MP2
"""
# ==> Import statements & Global Options <==
import psi4
import numpy as np

psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file('output.dat', False)
# ==> Molecule & Psi4 Options Definitions <==
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis':        'aug-cc-pvdz',
                  'scf_type':     'pk',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})
# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
# ==> Get orbital information & energy eigenvalues <==
# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()

# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]
# ==> ERIs <==
# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Memory check for ERI tensor
I_size = (nmo**4) * 8.e-9
print('\nSize of the ERI tensor will be %4.2f GB.' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                     limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]

# Naive Algorithm for ERI Transformation
Imo = np.einsum('pi,qa,pqrs,rj,sb->iajb', Cocc, Cvirt, I, Cocc, Cvirt)
# ==> Transform I -> I_mo @ O(N^5) <==
tmp = np.einsum('pi,pqrs->iqrs', Cocc, I)
tmp = np.einsum('qa,iqrs->iars', Cvirt, tmp)
tmp = np.einsum('iars,rj->iajs', tmp, Cocc)
I_mo = np.einsum('iajs,sb->iajb', tmp, Cvirt)
# ==> Compare our Imo to MintsHelper <==
Co = scf_wfn.Ca_subset('AO','OCC')
Cv = scf_wfn.Ca_subset('AO','VIR')
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
print("Do our transformed ERIs match Psi4's? %s" % np.allclose(I_mo, np.asarray(MO)))


# ==> Compute MP2 Correlation & MP2 Energy <==
# Compute energy denominator array
e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1, 1) - e_ab)

# Compute SS & OS MP2 Correlation with Einsum
mp2_os_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo, e_denom)
mp2_ss_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3), e_denom)

# Total MP2 Energy
MP2_E = scf_e + mp2_os_corr + mp2_ss_corr

# ==> Compare to Psi4 <==
psi4.driver.p4util.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')

