"""
Created on Tue 12 Jan 18:27:16 2016

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

#===========================================00=================================================
#
#                                  1D Helmholtz solver
#
#===========================================00=================================================
def Helmholtz1D(beta, N, p_hat):
    
    u_hat = zeros(N)

    tmp_mat = beta*D2x-Ix
    lhs = dot(tmp_mat, Bmat) 
    rhs = dot(D2x,p_hat)
    u_hat[:-2] = linalg.solve(lhs[2:,:-2],rhs[2:])

    return u_hat
#===========================================00=================================================
#
#                                  1D Poisson solver - Dirichlet BCs
#
#===========================================00=================================================
def Poisson1D(beta, N, p_hat):
    
    u_hat = zeros(N)

    ll = -beta*D2x + Ix
    lhs = dot(ll, Bmat) 
    rhs = dot(D2x,p_hat)
    u_hat[:-2] = linalg.solve(lhs[2:,:-2],rhs[2:])
    return u_hat
#===========================================00=================================================
#
#               1D Poisson solver - Neumann BCs
#
#===========================================00=================================================
def Poisson1DNeumann(beta, N, rhs):
    
    u_hat = zeros(N)
      
    ll = -beta*D2x + Ix
    lhs = dot(ll, BN) 
    rhs = dot(D2x,rhs)    
    u_hat[1:-2] = linalg.solve(lhs[3:,1:-2],rhs[3:])
    return u_hat
#===========================================00=================================================
#
#               1D Poisson solver - Pressure Correction - Neumann BCs
#
#===========================================00=================================================
def Poisson1DPcorr(beta, N, rhs):
    
    u_hat = zeros(N)
      
    ll = -beta*D2x + Ix
    lhs = dot(ll, BN)    
    u_hat[1:-2] = linalg.solve(lhs[3:,1:-2],rhs[3:])
    return u_hat

#===========================================00=================================================
#===========================================00=================================================

for i in range(4):
    if i == 0:
        print "Pressure correction solver:\n"
        # Initial conditions
        U[0] = ((4./pi)+ (pi/2.))*cos(pi*X[0]/2.)*sin(X[1])*sin(X[2])
        U[1] = 0#cos(pi*X[0])*cos(X[1])*sin(X[2])
        U[2] = 0#cos(pi*X[0])*sin(X[1])*cos(X[2])
        for i in range(3):
            U_hat[i] = fst(U[i], U_hat[i], ST)        
        
        P_exact = sin(pi*Y[0]/2)*sin(Y[1])*sin(Y[2])
        beta = K[1, 0]**2+K[2, 0]**2

        # Pseudo-inverse matrices
        BD = B_matrix(N[0], BC1)
        BN = B_matrix(N[0], BC2)        
        Ix, Imx = QI2(N[0])
        Dx, D2x, D4x = QIM(N[0], quad)
        
        r0 = dot(Dx,BD)
        r1 = dot(Ix, r0)
        r2 = dot(D2x, BD)
    
        for i in range(Np[1]):
            for j in range(Nf):
                rhs1 = dot(r1,U_hat[0,:, i, j])
                rhs2 = 1j*K[1,0,i,j]*dot(r2,U_hat[1,:,i,j])       
                rhs3 = 1j*K[2,0,i,j]*dot(r2,U_hat[2,:,i,j])  

                rhs = rhs1 + rhs2 + rhs3
                P_hat[:,i,j].real = Poisson1DPcorr(beta[i,j], N[0], rhs.real)
                P_hat[:,i,j].imag = Poisson1DPcorr(beta[i,j], N[0], rhs.imag)        
                
        P = ifst(P_hat, P, SN)

        print "Error: ", linalg.norm(P[:,2,2]-P_exact[:,2,2],inf)
        assert allclose(P,P_exact)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import colors, ticker, cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Y[0,:,:,1], Y[1,:,:,1], P_exact[:,:,1], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        #=================================================================000======
    elif i == 1:
        print "Poisson solver Neumann BCs:\n"
        # Initial conditions 
        U_exact = sin(pi*X[0]/2.)*sin(X[1])*sin(X[2]) 
        P[:] = -(2.+ (pi/2.)**2)*sin(pi*X[0]/2.)*sin(X[1])*sin(X[2]) 
        beta = K[1, 0]**2+K[2, 0]**2

        P_hat = fct(P,P_hat)

        # Pseudo-inverse matrices
        BN = B_matrix(N[0], BC2)
        Ix, Imx = QI2(N[0])
        Dx, D2x, D4x = QIM(N[0], quad)
            
        for i in range(Np[1]):
            for j in range(Nf):
                U_hat[0,:,i,j].real = Poisson1DNeumann(beta[i,j], N[0], P_hat[:,i,j].real)
                U_hat[0,:,i,j].imag = Poisson1DNeumann(beta[i,j], N[0], P_hat[:,i,j].imag)        
                
        U[0] = ifst(U_hat[0], U[0], SN)
        print "Error: ", linalg.norm(U[0,:,2,2]-U_exact[:,2,2],inf)
        assert allclose(U[0],U_exact)
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import colors, ticker, cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X[0,:,:,1], X[1,:,:,1], U[0,:,:,1], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        #=================================================================000======
    elif i == 2:
        print "Poisson solver Dirichlet BCs:\n"
        # Initial conditions
        U_exact = 2.*(1.0-X[0]**2)*sin(X[1])*sin(X[2])
        P[:] = -4*(2.-X[0]**2)*sin(X[1])*sin(X[2])
        beta = K[1, 0]**2+K[2, 0]**2

        P_hat = fct(P,P_hat)

        # Pseudo-inverse matrices
        Bmat = B_matrix(N[0], BC1)
        Ix, Imx = QI2(N[0])
        Dx, D2x, D4x = QIM(N[0], quad)
            
        for i in range(Np[1]):
            for j in range(Nf):
                U_hat[0,:,i,j].real = Poisson1D(beta[i,j], N[0], P_hat[:,i,j].real)
                U_hat[0,:,i,j].imag = Poisson1D(beta[i,j], N[0], P_hat[:,i,j].imag)        
                
        U[0] = ifst(U_hat[0], U[0], ST)

        #print "Error: ", linalg.norm(U[0,:,2,2]-U_exact[:,2,2],inf)
        assert allclose(U[0],U_exact)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import colors, ticker, cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X[0,:,:,1], X[1,:,:,1], U[0,:,:,1], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        #=================================================================000======
    elif i == 3:
        print "Helmholtz solver:\n"
        # Initial conditions
        U_exact = 2.*(1.0-X[0]**2)*sin(X[1])*sin(X[2])
        alpha = 1.e3
        P[:] = alpha*U_exact+4*(2.-X[0]**2)*sin(X[1])*sin(X[2])
        beta = alpha+K[1, 0]**2+K[2, 0]**2

        P_hat = fct(P,P_hat)

        # Pseudo-inverse matrices
        Bmat = B_matrix(N[0], BC1)
        Ix, Imx = QI2(N[0])
        Dx, D2x, D4x = QIM(N[0], quad)
            
        for i in range(Np[1]):
            for j in range(Nf):
                U_hat[0,:,i,j].real = Helmholtz1D(beta[i,j], N[0], P_hat[:,i,j].real)
                U_hat[0,:,i,j].imag = Helmholtz1D(beta[i,j], N[0], P_hat[:,i,j].imag)        
                
        U[0] = ifst(U_hat[0], U[0], ST)
        
        #print "Error: ", linalg.norm(U[0,:,2,2]-U_exact[:,2,2],inf)
        assert allclose(U[0],U_exact)

        #=================================================================000======
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import colors, ticker, cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X[0,:,:,1], X[1,:,:,1], U[0,:,:,1], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)        
        plt.show()
       