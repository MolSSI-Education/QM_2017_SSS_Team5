import numpy as np
import psi4

def jk(mol,basis="sto-3g",iteration=20):
    """
    Function that calcultate JK intergrals
    """
    np.set_printoptions(suppress=True, precision=4)

    # Build a molecule
    mol.update_geometry()
    mol.print_out()

    e_conv = 1.e-6
    d_conv = 1.e-6
    nel = 5
    damp_value = 0.20
    damp_start = 5

    # Build a basis
    bas = psi4.core.BasisSet.build(mol, target=basis)
    bas.print_out()

    # Build a Auxiliary basis set(df)
    aux = psi4.core.BasisSet.build(mol, key="DF_BASIS_SCF", fitrole="JKFIT", other=basis.upper())

    # The zero basis set
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()

    if (nbf > 100):
        raise Exception("More than 100 basis functions!")

    # Build (Q|λσ) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
    Qrs_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
    Qrs_tilde = np.squeeze(Qrs_tilde) # remove the 1-dimensions
    Qrs_tilde = np.array(Qrs_tilde)

    # Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1.e-14)
    metric = np.squeeze(metric) # remove the 1-dimensions
    metric = np.array(metric)

    #Build (P|λσ)
    Prs = np.einsum("PQ,Qrs->Prs", metric, Qrs_tilde)

    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())

    # Core Hamiltonian
    H = T + V

    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    # Diagonalize Core H
    def diag(F, A):
        Fp = A.T @ F @ A
        eps, Cp = np.linalg.eigh(Fp)
        C = A @ Cp
        return eps, C

    eps, C = diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

    E_old = 0.0
    F_old = None
    for iteration in range(iteration):
        Xj = np.einsum("Prs,rs->P", Prs, D)#Build χP
        Xk = np.einsum("Pqs,rs->Pqr", Prs, D)#Build Xk

        J = np.einsum("pqP,P->pq", Prs.T, Xj)#Build J
        K = np.einsum("prP,Pqr->pq", Prs.T, Xk)#Build K

        F_new = H + 2.0 * J - K

        # conditional iteration > start_damp
        if iteration >= damp_start:
            F = damp_value * F_old + (1.0 - damp_value) * F_new
        else:
            F = F_new

        F_old = F_new

        # Build the AO gradient
        grad = F @ D @ S - S @ D @ F

        grad_rms = np.mean(grad ** 2) ** 0.5

        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + mol.nuclear_repulsion_energy()

        E_diff = E_total - E_old
        E_old = E_total
#        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
#                (iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (grad_rms < d_conv):
            break

        eps, C = diag(F, A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T

#    print("SCF has finished!\n")





    #psi4.set_output_file("output.dat")
    #psi4.set_options({"scf_type": "pk"})
    #psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
    #print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
    return E_total

#mol = psi4.geometry("""
#         O
#         H 1 1.1
#         H 1 1.1 2 104
#         symmetry c1
#         """)
# print(jk(mol, iteration = 40))