"""
Created on Thu Jan 21 14:01:18 2016

@author: Diako Darian

"""


from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn, rfft, irfft
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransform import ShenBasis, ShenBiharmonicBasis, ChebyshevTransform
from SpectralDiff import *
import SFTc
import matplotlib.pylab as pl
import scipy.sparse as sps
from scipy import sparse
import sys


M = 5
N = array([2**M, 2**(M-1), 2**(M-1)])
L = array([2, 2*pi, 2*pi])
    
dx = (L / N).astype(float)
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N / num_processes
# Get points and weights for Chebyshev weighted integrals
quad = "GC"
BC1 = array([1,0,0, 1,0,0])
BC2 = array([0,1,0, 0,1,0])
BC3 = array([0,1,0, 1,0,0])
SC = ChebyshevTransform(quad)
ST = ShenBasis(BC1, quad)
SN = ShenBasis(BC2, quad, Neumann = True)
SR = ShenBasis(BC3, quad)
SB = ShenBiharmonicBasis(quad, fast_transform=False)

points, weights = ST.points_and_weights(N[0])
pointsN, weightsN = SN.points_and_weights(N[0])

x1 = arange(N[1], dtype=float)*L[1]/N[1]
x2 = arange(N[2], dtype=float)*L[2]/N[2]

# Get grid for velocity points
X = array(meshgrid(points[rank*Np[0]:(rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)
Y = array(meshgrid(pointsN[rank*Np[0]:(rank+1)*Np[0]], x1, x2, indexing='ij'), dtype=float)

Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
Nu = N[0]-2   # Number of velocity modes in Shen basis

U     = empty((3, Np[0], N[1], N[2]))
U_hat = empty((3, N[0], Np[1], Nf), dtype="complex")
P     = empty((Np[0], N[1], N[2]))
P_hat = empty((N[0], Np[1], Nf), dtype="complex")

dU      = empty((4, N[0], Np[1], Nf), dtype="complex")
F_tmp   = empty((3, N[0], Np[1], Nf), dtype="complex")
Uc      = empty((Np[0], N[1], N[2]))
Uc2     = empty((Np[0], N[1], N[2]))
Uc_hat  = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hat2 = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hat3 = empty((N[0], Np[1], Nf), dtype="complex")
Uc_hatT = empty((Np[0], N[1], Nf), dtype="complex")
U_mpi   = empty((num_processes, Np[0], Np[1], Nf), dtype="complex")
U_mpi2  = empty((num_processes, Np[0], Np[1], N[2]))

kx = arange(N[0]).astype(float)
ky = fftfreq(N[1], 1./N[1])[rank*Np[1]:(rank+1)*Np[1]]
kz = fftfreq(N[2], 1./N[2])[:Nf]
kz[-1] *= -1.0

# scale with physical mesh size. 
# This takes care of mapping the physical domain to a computational cube of size (2, 2pi, 2pi)
# Note that first direction cannot be different from 2 (yet)
Lp = array([2, 2*pi, 2*pi])/L
K  = array(meshgrid(kx, ky, kz, indexing='ij'), dtype=float)
K[0] *= Lp[0]; K[1] *= Lp[1]; K[2] *= Lp[2] 
K2 = sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)


def fss(u, fu, S):
    """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = S.fastShenScalar(Uc_hat, fu)
    return fu

def ifst(fu, u, S):
    """Inverse Shen transform of x-direction, Fourier in y and z"""
    Uc_hat3[:] = S.ifst(fu, Uc_hat3)
    comm.Alltoall([Uc_hat3, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        Uc_hatT[:, i*n0:(i+1)*n0] = U_mpi[i]
    u[:] = irfft2(Uc_hatT, axes=(1,2))
    return u

def fst(u, fu, S):
    """Fast Shen transform of x-direction, Fourier transform of y and z"""
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = S.fst(Uc_hat, fu)
    return fu

def fct(u, fu):
    """Fast Cheb transform of x-direction, Fourier transform of y and z"""
    Uc_hatT[:] = rfft2(u, axes=(1,2))
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = ST.fct(Uc_hat, fu)
    return fu

def ifct(fu, u):
    """Inverse Cheb transform of x-direction, Fourier in y and z"""
    Uc_hat3[:] = ST.ifct(fu, Uc_hat3)
    comm.Alltoall([Uc_hat3, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    n0 = U_mpi.shape[2]
    for i in range(num_processes):
        Uc_hatT[:, i*n0:(i+1)*n0] = U_mpi[i]
    u[:] = irfft2(Uc_hatT, axes=(1,2))
    return u

def fct0(u, fu):
    """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do fct."""
    n0 = U_mpi2.shape[2]
    for i in range(num_processes):
        U_mpi2[i] = u[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi2, MPI.DOUBLE], [UT[0], MPI.DOUBLE])
    fu = ST.fct(UT[0], fu)
    return fu

def ifct0(fu, u):
    """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do ifct"""
    UT[0] = ST.ifct(fu, UT[0])
    comm.Alltoall([UT[0], MPI.DOUBLE], [U_mpi2, MPI.DOUBLE])
    n0 = U_mpi2.shape[2]
    for i in range(num_processes):
        u[:, i*n0:(i+1)*n0] = U_mpi2[i]
    return u

ck = chebNormalizationFactor(N[0], quad)
e = zeros(N[0]+3)
d_a = zeros(N[0])
d_b = zeros(N[0])
d2_a = zeros(N[0])
d2_b = zeros(N[0])
d2_c = zeros(N[0])

alpha_k = zeros(N[0])
beta_k  = zeros(N[0])
gamma_k = zeros(N[0])
d_k     = zeros(N[0])

# Wavenumbers: 
kk = wavenumbers(N[0]+1)
# Shen coefficients for the basis functions
a_k, b_k = shenCoefficients(kk, BC2)

for i in range(N[0]):
    e[i] = 1
 
for k in xrange(1,N[0]):
    d_a[k-1] = -e[k+2]/(2*k)
    d_b[k-1] = ck[k-1]/(2*k)

for k in xrange(2,N[0]):
    d2_a[k-2] = -e[k+2]/(2*(k**2-1.)) 
    d2_c[k-2] = ck[k-2]/(4*k*(k-1))
    if k<N[0]-1:
        d2_b[k-2] = e[k+4]/(4*k*(k+1))  

for i in xrange(N[0]-3):    
    beta_k[i]  = d2_c[i+1]+ b_k[i+1]*d2_a[i+1]
    if i < (N[0]-5):
        alpha_k[i] = d2_a[i+1] + b_k[i+3]*d2_b[i+1]
        gamma_k[i] = b_k[i+1]*d2_c[i+3]
d_k[:-7] = d2_b[1:-6]

# Initial conditions
U[0] = ((4./pi)+ (pi/2.))*cos(pi*X[0]/2.)*sin(X[1])*sin(X[2])
U[1] = 0#cos(pi*X[0])*cos(X[1])*sin(X[2])
U[2] = 0#cos(pi*X[0])*sin(X[1])*cos(X[2])
for i in range(3):
    U_hat[i] = fst(U[i], U_hat[i], ST)        

# Exact pressure
P_exact = sin(pi*Y[0]/2)*sin(Y[1])*sin(Y[2])
# Wavenumbers
beta = K[1, 0]**2+K[2, 0]**2

#=======================================================000========
#              RHS of the pressure equation
#=======================================================000========
def RHS(d_a,d_b,d2_a,d2_b,d2_c, U_hat, dU):
    
    F_tmp[0] = SFTc.MatVecMult1(d_a, d_b, U_hat[0], F_tmp[0]) 
    F_tmp[1] = SFTc.MatVecMult2(d2_a, d2_b,d2_c, U_hat[1], F_tmp[1]) 
    F_tmp[2] = SFTc.MatVecMult2(d_a, d_b,d2_c, U_hat[2], F_tmp[2]) 
    
    dU[0] = F_tmp[0] +1j*K[1]*F_tmp[1] + 1j*K[2]*F_tmp[2]
    
    return dU
#=======================================================000========

#=======================================================000========
#              Solver for pressure equation
#=======================================================000========
def PSolve(beta,b_k,alpha_k,beta_k,gamma_k,d_k, dU, P_hat):
    
    P_hat[1:-2,:,:] = SFTc.PressureSolver(beta,b_k[1:],alpha_k,beta_k,gamma_k,d_k, dU[0,3:,:,:], P_hat[1:-2,:,:])
    
    return P_hat
#=======================================================000========

dU = RHS(d_a,d_b,d2_a,d2_b,d2_c, U_hat, dU) 

P_hat = PSolve(beta,b_k,alpha_k,beta_k,gamma_k,d_k, dU, P_hat)
        
     
P = ifst(P_hat, P, SN)
print "Error: ", linalg.norm(P[:,3,2]-P_exact[:,3,2],inf)
assert allclose(P,P_exact)
#=================================================================000======