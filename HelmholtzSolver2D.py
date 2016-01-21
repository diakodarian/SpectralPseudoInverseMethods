"""
Created on Tue 12 Jan 15:51:26 2016

@author: Diako Darian

"""


from numpy import *
from numpy.fft import fftfreq
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransform import ShenBasis, ShenBiharmonicBasis, ChebyshevTransform
from FFTChebTransforms import *
from SpectralDiff import *
import SFTc
import matplotlib.pylab as pl
import scipy.sparse as sps
from scipy import sparse
import sys

#=====================================================================
#       2D Helmholtz solver (Non-periodic in x periodic in y)
#===================================================================== 
def HelmholtzNonPeriodic(M, quad, ST, num_processes):

    N = array([2**M, 2**(M)]) 
    L = array([2, 2*pi])

    kx = arange(N[0]).astype(float)
    ky = fftfreq(N[1], 1./N[1])

    Lp = array([2, 2*pi])/L
    K  = array(meshgrid(kx, ky, indexing='ij'), dtype=float)
    K[0] *= Lp[0]; K[1] *= Lp[1] 

    points, weights = ST.points_and_weights(N[0])
    x1 = arange(N[1], dtype=float)*L[1]/N[1]
    X = array(meshgrid(points, x1, indexing='ij'), dtype=float)

    Ix = identity(N[0])
    Iy = identity(N[1])
    Dx = chebDiff2(N[0],quad)
    Dy = diag(-K[1,0,:]**2)
    
    U     = empty((N[0], N[1]))
    U_hat = empty((N[0], N[1]), dtype="complex")
    P     = empty((N[0], N[1]))
    P_hat = empty((N[0], N[1]), dtype="complex")

    alpha = 2.e4
    exact = sin(pi*X[0])*sin(X[1])
    P[:] = (alpha-1.0-pi**2)*exact

    P_hat = fct(P, P_hat, ST, num_processes, comm)
    
    U_hat = U_hat.reshape((N[0]*N[1],1))
    P_hat = P_hat.reshape((N[0]*N[1],1))

    lhs = alpha*kron(Ix,Iy) + kron(Dx,Iy) + kron(Ix,Dy)
    rhs = dot(kron(Ix,Iy),P_hat)
    #sparse.kron(Dy,Ix) + sparse.kron(Iy,Dx)
    testz = kron(kron(Dy,Ix),Ix) + kron(kron(Iy,Dy),Ix) + kron(Ix,kron(Iy,Dx)) 
    pl.spy(testz,precision=1.0e-16, markersize=3)
    pl.show()     
    sys.exit()
    
    U_hat = linalg.solve(lhs,rhs)

    U_hat = U_hat.reshape((N[0],N[1]))
    U = ifct(U_hat, U, ST, num_processes, comm)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import colors, ticker, cm
    cs = plt.contourf(X[0], X[1], U, 50, cmap=cm.coolwarm)
    cbar = plt.colorbar()
    plt.show()
        
    print "Error: ", linalg.norm(U-exact,inf)
    assert allclose(U,exact)


#=====================================================================
#      Periodic 2D Helmholtz solver 
#===================================================================== 
def HelmholtzPeriodic(M, num_processes, approach = "Matrix", plotU = True):

    # Set the size of the doubly periodic box N**2
    N = 2**M
    L = 2 * pi
    dx = L / N
   
    Np = N / num_processes

    Uc      = empty((Np, N))
    Uc2     = empty((Np, N))
    Uc_hat  = empty((N, Np), dtype="complex")
    Uc_hat2 = empty((N, Np), dtype="complex")
    Uc_hat3 = empty((N, Np), dtype="complex")
    Uc_hatT = empty((Np, N), dtype="complex")
    U_mpi   = empty((num_processes, Np, Np), dtype="complex")
    U_mpi2  = empty((num_processes, Np, Np))
  
    # Create the mesh
    X = mgrid[rank*Np:(rank+1)*Np, :N].astype(float)*L/N

    # Solution array and Fourier coefficients
    # Because of real transforms and symmetries, N/2+1 coefficients are sufficient
    Nf = N/2+1
    Npf = Np/2+1 if rank+1 == num_processes else Np/2
    
    # Set wavenumbers in grid
    kx = fftfreq(N, 1./N)
    ky = kx[:Nf].copy(); ky[-1] *= -1
    K = array(meshgrid(kx, ky[rank*Np/2:(rank*Np/2+Npf)], indexing='ij'), dtype=int)
    K2 = sum(K*K, 0)
    KK_inv = 1.0/where(K2==0, 1, K2).astype(float)
    K_over_K2 = array(K, dtype=float) / where(K2==0, 1, K2)

    U     = empty((Np, N))
    U_hat = empty((N, Npf), dtype="complex")
    P     = empty((Np, N))
    Ptest     = empty((Np, N))
    P_hat = empty((N, Npf), dtype="complex")

    U_send = empty((num_processes, Np, Np/2), dtype="complex")
    U_sendr = U_send.reshape((N, Np/2))

    U_recv = empty((N, Np/2), dtype="complex")
    fft_y = empty(N, dtype="complex")
    fft_x = empty(N, dtype="complex")
    plane_recv = empty(Np, dtype="complex")

    P[:] = -1.9*sin(X[0])*sin(X[1])
    exact = sin(X[0])*sin(X[1])
    alpha = 0.1    
    
    #--------------------
    # Standard and Matrix approach
    #-------------------
    if approach == "standard":
        P_hat = rfft2_mpi(P, P_hat,num_processes)
        
        U_hat = P_hat/(.1 - K2)
        U = irfft2_mpi(U_hat, U)
        
        print "Error: ", linalg.norm(U-exact,inf)
        assert allclose(U,exact)
    
    elif approach == "Matrix": 
        P_hat = rfft2_mpi(P, P_hat,num_processes)
        U_hat = U_hat.reshape((N*Npf,1))
        P_hat = P_hat.reshape((N*Npf,1))
        
        Ix = identity(N)
        Iy = identity(Npf)
        Dx = diag(-K[0,:,0]**2)
        Dy = diag(-K[1,0,:]**2)
        
        lhs = alpha*kron(Ix,Iy) + kron(Dx,Iy) + kron(Ix,Dy)
        rhs = dot(kron(Ix,Iy),P_hat)
        
        U_hat = linalg.solve(lhs,rhs)
        
        U_hat = U_hat.reshape((N,Npf))
        U = irfft2_mpi(U_hat, U,num_processes)
        
        print "Condition number: ", linalg.cond(lhs)
        print "Error: ", linalg.norm(U-exact,inf)
        assert allclose(U,exact)
        
    if plotU == True:    
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import colors, ticker, cm
        cs = plt.contourf(X[0], X[1], U, 50, cmap=cm.coolwarm)
        cbar = plt.colorbar()
        plt.show()


if __name__ == '__main__':   
    
    M = 2
    quad = "GL"
    BC1 = array([1,0,0, 1,0,0])
    BC2 = array([0,1,0, 0,1,0])
    BC3 = array([0,1,0, 1,0,0])
    SC = ChebyshevTransform(quad)
    ST = ShenBasis(BC1, quad)
    SN = ShenBasis(BC2, quad, Neumann = True)
    SR = ShenBasis(BC3, quad)
    SB = ShenBiharmonicBasis(quad, fast_transform=False)
    
    
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()
    
    for i in range(2):
        if i == 0:
            HelmholtzNonPeriodic(M, quad, ST, num_processes)    
        elif i == 10:
            HelmholtzPeriodic(M,num_processes,"Matrix", True)
