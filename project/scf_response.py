################################
# SCF_response by sahil gulania
# QM TEAM 5
################################
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

# gram_schmidt_orthogonalize
def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

eps, C = diag(H,A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T

for i in range (1,10):
    # Build Fock Matrix
    # F = H + 2 * G_pqrs D_rs - G_prqs D_rs
    # G = (7, 7, 7, 7)
    # D = (1, 1, 7, 7)

    #Jsum = np.sum(G * D, axis=(2,3) )
    #Jein = np.einsum("pqrs,rs->pq", G, D)

    J = np.einsum("pqrs,rs->pq", G, D)
    K = np.einsum("prqs,rs->pq", G, D)
    F = H + 2.0 * J - K

    occ = nel
    vir = len(F[0,:]) - nel


    #########################################
    # Compute Fock Matrix in MO basis.
    FMO = np.dot(C.T,np.dot(F,C))
    #print(FMO)
    #########################################
    #print(C)
    #break


    ##########################################
    # Initial Hessian and
    # Initial gradient

    Hess = (occ*vir, occ*vir)
    Hess = np.zeros(Hess)

    orbgrad = (occ*vir)
    orbgrad = np.zeros(orbgrad)
    #print(FMO)
    kk = 0
    for j in range(0,occ):
        for k in range (occ, occ + vir ):
            Hess [kk,kk] = 4*(FMO[k,k] - FMO[j,j])
            #print(FMO[k,k],FMO[j,j],4*(FMO[k,k] - FMO[j,j]))
            #print(FMO[j][k])
            orbgrad [kk] = 4*FMO[j,k]
            kk = kk +1
            #print(j,k)
    ##########################################
    #break
    #print(orbgrad)
    # Build gradient
    grad = F @ D @ S -S @ D @ F
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


    #######################################
    #SOSCF STEP
    XX  = np.dot(orbgrad,-1*np.linalg.inv(Hess))

    AA = np.identity(occ+vir)
    kk = 0
    for j in range(occ, occ+vir):
        for k in range (0, occ):
            AA [j,k] = XX [kk]
            AA [k,j] = -XX [kk]
            kk = kk+1

    GAA = gram_schmidt_columns(AA)
    #C = np.dot(C,GAA)
    #######################################

    if (grad_rms > 0.01):
        eps, C = diag(F,A)

    if (grad_rms < 0.01):
        print(GAA)
        print(AA)
        C = np.dot(C,GAA)

    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T


print("SCF has finished!\n")

psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))

