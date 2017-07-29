import numpy as np
import psi4



def scf(mol, nel = 5, basis="sto-3g",iteration=25, prt = False):
    import project as pj
    from project import scf_tools as st

    #import scf_tools as st

    np.set_printoptions(suppress=True, precision=4)
    mol.update_geometry()
    if prt is True:
        mol.print_out()
    else:
        pass
    bas = psi4.core.BasisSet.build(mol, target= basis)
    nel = nel


    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()
    if (nbf > 100):
        raise Exception("More than 100 basis functions!")

    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())

    # Core Hamiltonian
    H = T + V

    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    S = np.array(A)
    # print(A)
    A.power(-0.5, 1.e-14)  # Neglect small numbers
    # print(A @ S @ A) #@ is for dot product here
    A = np.array(A)


    # J = np.einsum("pqrs, rs -> pq", g, D)
    # K = np.einsum("prqs, rs -> pq", g, D)
    # F = H + 2.0 * J - K
    # Diagonalize core H
    eps, C = st.diag(H, A)

    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    E_old = 0e0
    for i in range(iteration):
        # F_pq = H_pq + 2 * g_pqrs * D_rs - g_prqs * D_rs
        # g = (7, 7, 7, 7)
        # D = (1, 1, 7, 7)

        # Jsum = np.sum(g * D, axis =(2,3) )
        J = np.einsum("pqrs, rs -> pq", g, D)
        K = np.einsum("prqs, rs -> pq", g, D)
        F = H + 2.0 * J - K
        # conditional iteration > damp_start
        # F = (damp-value) Fold + (??) Fnew


        # Build the AO gradient
        grad = F @ D @ S - S @ D @ F  # ? What is this AO gradient?
        grad_rms = np.mean(grad ** 2) ** 0.5

        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + mol.nuclear_repulsion_energy()
        E_diff = E_total - E_old
        E_old = E_total
        if prt is True:
            print("Iter=%3d E = %16.12f E_diff = % 8.4e D_diff = %8.4e" % (i, E_total, E_diff, grad_rms))
        else:
            pass

        # Break if e_conv and d_conv are met
        eps, C = st.diag(F, A)
        #    print(F,H)
        #    print(F-H)

        #    raise Exception('end')
        #    print(diag(F,A), diag(H,A))
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T
    print(E_total)
    return E_total


