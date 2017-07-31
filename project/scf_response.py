# Homework complete damping
#27 July 2017
#SCF_Response by Sahil Gulania
#Not complete yet

import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()
mol.print_out()

# Basis Set defined
bas = psi4.core.BasisSet.build(mol, target="sto-3g")
bas.print_out()

#if(nbf > 100):
#  raise Exception("More that 100 basis function")
nel = 5
e_conv = 1.e-10
d_conv = 1.e-10

# Build a MintsHelper
mints = psi4.core.MintsHelper(bas)

V = np.array(mints.ao_potential())
T = np.array(mints.ao_kinetic())


# Core Hamiltonian
H = T+V

S = np.array(mints.ao_overlap())
G = np.array(mints.ao_eri())

#print(S.shape)
#print(G.shape)

A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)

# Diagonalize core Hamiltonian
def diag(M,A):
    Fp = A.T @ M @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C

eps, C = diag(H,A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T

for i in range (1,50):
    # Build Fock Matrix
    # F = H + 2 * G_pqrs D_rs - G_prqs D_rs
    # G = (7, 7, 7, 7)
    # D = (1, 1, 7, 7)

    #Jsum = np.sum(G * D, axis=(2,3) )
    #Jein = np.einsum("pqrs,rs->pq", G, D)

    J = np.einsum("pqrs,rs->pq", G, D)
    K = np.einsum("prqs,rs->pq", G, D)
    F = H + 2.0 * J - K

    K1 = np.empty([len(S[1,:]), len(S[1,:])])
    #################################
    # SOSCF
    HK = np.empty([len(S[1,:]), len(S[1,:])])
    for p in range(1,len(S[1,:])):
        for q in range(1,len(S[1,:])):

            for o in range(1,len(S[1,:])):
                HK[p,q] = HK[p,q] + K1[p,o]*H[o,q] + K1[q,o]*H[p,o]



    ##
    GK = np.empty([len(S[1,:]), len(S[1,:]), len(S[1,:]), len(S[1,:]) ])
    for p in range(1,len(S[1,:])):
        for q in range(1,len(S[1,:])):

            for r in range(1,len(S[1,:])):
                for s in range(1,len(S[1,:])):
                    for o in range(1,len(S[1,:])):
                        GK[p,q,r,s] =  GK[p,q,r,s] + \
                            K1[p,o]*G[o,p,r,s] + K1[q,o]*G[p,o,r,s] + \
                            K1[r,o]*G[p,q,o,s] + K1[s,o]*G[p,q,r,o]
    #
    JK = np.einsum("pqrs,rs->pq", GK, D)
    KK = np.einsum("prqs,rs->pq", GK, D)
    FK = HK + 2.0 * JK - KK

    # Build gradient
    grad = F @ D @ S -S @ D @ F

    grad_rms = np.mean((np.mean(grad) - grad)**2)**0.5
    #print(np.mean((np.mean(grad) - grad)**2)**0.5)
    #print(np.mean(grad**2) ** 0.5)

    E_electric = np.sum((F + H) * D)
    #print(E_electric)

    E_total = E_electric + mol.nuclear_repulsion_energy()
    if (i == 1):
        E_diff = 0
    elif (i > 1):
        E_diff = E_total - E_old

    E_old = E_total
    print("Iter=%3d  E = % 16.12f  dE = % 8.4e  dRMS = % 8.4e" %
            (i, E_total, E_diff, grad_rms))

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    if (i == 1):
        eps, C = diag(F,A)
    elif(i > 4):
        KK = np.dot(-1*grad,np.linalg.inv(FK))
        I = np.identity(len(S[1,:]))
        U = np.add(I,KK)
        C = np.dot(U,C)

    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

print("SCF has finished!\n")

psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
