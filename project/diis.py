import numpy as np
import psi4
def diis(mol,nel=5,basis="sto-3g",cycle=50):
    from project import scf_tools as st

    np.set_printoptions(suppress=True, precision=4)
    # number of electrons
    nel = nel

    #parameters for convergence
    e_conv = 1.e-10
    d_conv = 1.e-10

    mol.update_geometry()
    mol.print_out()

    # Basis Set defined
    bas = psi4.core.BasisSet.build(mol, target=basis)
    bas.print_out()


    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)

    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())


    # Core Hamiltonian
    H = T+V

    S = np.array(mints.ao_overlap())
    G = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    ###################################


    # Diagonalize core Hamiltonian
    eps, C = st.diag(H,A)
    Cocc = C[:, :nel]

    # Compute Density
    D = Cocc @ Cocc.T

    ###################################################
    # Buliding diis matrix
    def lang2(A):
        for i in range(len(A[:,:,1])):
            for j in range(len(A[:,:,1])):
                B2 = np.dot(A[i,:,:],A[j,:,:])
                if (j > 0):
                    B2tem = np.array(B2[0,0])
                    #print(B2tem)
                    #print(B1)
                    B1 = np.hstack((B1,B2tem))
                else:
                    B1 = B2[0,0]
            if(i > 0):
                B = np.concatenate((B,B1), axis=0)
            else:
                B = B1
        return B
    ###################################################


    for i in range (1,cycle):
        # Build Fock Matrix
        # F = H + 2 * G_pqrs D_rs - G_prqs D_rs
        # G = (7, 7, 7, 7)
        # D = (1, 1, 7, 7)

        J = np.einsum("pqrs,rs->pq", G, D)
        K = np.einsum("prqs,rs->pq", G, D)
        F = H + 2.0 * J - K

        # Arranging Fock Matrix side by side
        if i == 1:
            Fbig = F
        elif (i > 1):
            Fbig = np.concatenate((Fbig, F), axis=0)
            Fbig1 = np.reshape(Fbig,(i,len(S[1,:]),len(S[1,:])))

        # Build gradient
        grad = F @ D @ S -S @ D @ F


        #############################################################
        # error vector
        ervec = (F @ D @ S - S @ D @ F)

        if (i > 1):
            ervec =  np.concatenate((e1, ervec), axis=0)
            ervec1 = ervec
            ervec1 = np.reshape(ervec1,(i,len(S[1,:]),len(S[1,:])))
            B2 = lang2 (ervec1)
            B2 = np.reshape(B2,(i,i))
            on = np.repeat(-1, i)
            on = np.insert(on, i, 0)
            B = np.zeros((i+1, i+1))
            B[:i,:i] = B2
            B[i,:] = -1
            B[:,i] = -1
            B[i,i] = 0
            C1 = np.linalg.inv(B)
            C  = (C1[i,:i])
            C = -1 * C
        #############################################################
        e1 = ervec

        grad_rms = np.mean(grad**2) ** 0.5

        E_electric = np.sum((F + H) * D)
        #print(E_electric)

        E_total = E_electric + mol.nuclear_repulsion_energy()
        if (i == 1):
            E_diff = 0
        elif (i > 1):
            E_diff = E_total - E_old

        if (E_diff < 1e-10 and i > 1):
            F = H + 2.0 * J - K

        E_old = E_total
        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
                (i, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (grad_rms < d_conv):
                break

        # New fock as sum of previous fock
        if (i > 1):
                F = (np.tensordot(C,Fbig1,1))

        # diagonalize fock to get the new coeffecients
        eps, C = st.diag(F,A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T

    print("SCF has finished!\n")
    return E_total


if __name__ == "__main__":
    print('Testing: water')
    mol = psi4.geometry("""
            O
            H 1 1.1
            H 1 1.1 2 104
            """)
    E_total = diis(mol)

    # psi4.set_output_file("output.dat")
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
    print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))