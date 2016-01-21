# -*- coding: utf-8 -*-
"""
Created on Tue 7 Dec 16:15:26 2015

@author: Diako Darian

"""

from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn
from mpi4py import MPI
import matplotlib.pyplot as plt
from shentransform import ShenBasis, ShenBiharmonicBasis, ChebyshevTransform
import SFTc
from scipy.fftpack import dct
import time
import matplotlib.pylab as pl
import scipy.sparse as sps
import sys

try:
    from cbcdns.fft.wrappyfftw import *
except ImportError:
    pass # Rely on numpy.fft routines

M = 6
N = array([2**M, 2**(M-1)])
L = array([2, 2*pi])

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = N / num_processes

Uc      = empty((Np[0], N[1]))
Uc2     = empty((Np[0], N[1]))
Uc_hat  = empty((N[0], Np[1]), dtype="complex")
Uc_hat2 = empty((N[0], Np[1]), dtype="complex")
Uc_hat3 = empty((N[0], Np[1]), dtype="complex")
Uc_hatT = empty((Np[0], N[1]), dtype="complex")
U_mpi   = empty((num_processes, Np[0], Np[1]), dtype="complex")
U_mpi2  = empty((num_processes, Np[0], Np[1]))

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

#=====================================================================
#           Wavenumbers in spectral space
#===================================================================== 
def wavenumbers(N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return arange(N).astype(float)
        else:
            kk = mgrid[:N[0], :N[1], :N[2]].astype(float)
            return kk[0]
        
#=====================================================================
#           Chebyshev normalization factor
#===================================================================== 
def chebNormalizationFactor(N, quad):
    if quad == "GC":
        ck = ones(N); ck[0] = 2
    elif quad == "GL":
        ck = ones(N); ck[0] = 2; ck[-1] = 2
    return ck
#=====================================================================
#           Shen Coefficients for second order problems
#===================================================================== 
def shenCoefficients(k, BC):
    """
    Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
    satisfy the imposed boundary conditions for a unique set of {a_k, b_k}.  
    """
    am = BC[0]; bm = BC[1]; cm = BC[2]
    ap = BC[3]; bp = BC[4]; cp = BC[5]

    detk = 2*am*ap + ((k + 1.)**2 + (k + 2.)**2)*(am*bp - ap*bm) - 2.*bm*bp*(k + 1.)**2*(k + 2.)**2

    Aa = am - bm*(k + 2.)**2; Ab= -ap - bp*(k + 2.)**2  
    Ac = am - bm*(k + 1.)**2; Ad= ap + bp*(k + 1.)**2

    y1 = -ap - bp*k**2 + cp; y2= -am + bm*k**2 + cm/((-1)**k) 

    ak = (1./detk)*(Aa*y1 + Ab*y2)
    bk = (1./detk)*(Ac*y1 + Ad*y2)

    return ak, bk
#=====================================================================
#            Identity Matrix I
#=====================================================================               
def identity(N):
    Id = eye(N)
    return Id
#=====================================================================
#     Pseudo-inverse Identity Matrices I^{-2} and I^{2}
#=====================================================================
def QI2(N):
    Ix   = eye(N)
    Imx  = eye(N)
    Ix[0,0]  = 0
    Ix[1,1]  = 0
    Imx[-1,-1] = 0
    Imx[-2,-2] = 0
    return Ix, Imx
#=====================================================================
#     Pseudo-inverse Identity Matrices I^{-4} and I^{4}
#=====================================================================
def QI4(N):
    I4x   = eye(N)
    I4mx  = eye(N)

    I4x[0,0]  = 0
    I4x[1,1]  = 0
    I4x[2,2]  = 0
    I4x[3,3]  = 0

    I4mx[-1,-1] = 0
    I4mx[-2,-2] = 0
    I4mx[-3,-3] = 0
    I4mx[-4,-4] = 0
    return I4x, I4mx
#=====================================================================
#            Pseudo-inverse Matrices  D^{1} D^{2} and D^{4}
#=====================================================================
def QIM(N,quad):
    Dx   = zeros((N,N))
    D2x  = zeros((N,N))
    Ix, Imx = QI2(N)
    I4x, I4mx = QI4(N)
    ck = chebNormalizationFactor(N, quad)
    e = zeros(N+3)
    for i in range(N):
        e[i] = 1
        
    for k in xrange(1,N):
        for j in xrange(N):
            if k == j+1:
                Dx[k,j] = ck[k-1]/(2*k)
            if k == j-1:
                Dx[k,j] = -e[k+2]/(2*k)
    for k in xrange(2,N):
        for j in xrange(N):
            if k == j:
                D2x[k,j] = -e[k+2]/(2*(k**2-1.)) 
            elif k == j+2:
                D2x[k,j] = ck[k-2]/(4*k*(k-1))
            elif k == j-2:
                D2x[k,j] = e[k+4]/(4*k*(k+1))           
    LM = dot(Ix,Dx)
    RM = dot(Dx,Imx)
    D2 = dot(LM,RM)
    
    LM2 = dot(I4x,D2)
    RM2 = dot(D2,I4mx)
    D4 = dot(LM2,RM2)
    return Dx, D2, D4
#=====================================================================
#       Tranfsorm of Shen_Robin to Chebyshev
#=====================================================================
def B_matrix(N):
    
    # Wavenumbers: 
    K = wavenumbers(N)
    # Shen coefficients for the basis functions
    a_k, b_k = shenCoefficients(K, BC1)
    #a_j, b_j = shenCoefficients(K, BC2)
    # Chebyshev normalization factor
    #ck = chebNormalizationFactor(N, ST.quad)
    
    Bmat = zeros((N,N))
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                Bmat[k,j] = 1.0
            elif k == j+1:    
                Bmat[k,j] = a_k[k] 
            elif k == j+2:    
                Bmat[k,j] = b_k[k] 
    #pl.spy(D4,precision=0.0000000001, markersize=3)
    #pl.show()            
    return Bmat
#=====================================================================
#       Tranfsorm of Shen_biharmonic to Chebyshev
#=====================================================================
def S_matrix(N):
    
    Smat = zeros((N,N)) 
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                Smat[k,j] = 1.0
            elif k == j+2:    
                Smat[k,j] = -2.*(j+2.)/(j+3.) 
            elif k == j+4:    
                Smat[k,j] = (j+1.)/(j+3.)
    return Smat
#=====================================================================
#                 1D Poisson solver
#=====================================================================
def Poisson1D(N, f):
    
    f_hat = zeros(N)
    v_hat = zeros(N)
    v = zeros(N)
    
    Bmat = B_matrix(N)
    Ix, Imx = QI2(N)
    Dx, D2x, D4 = QIM(N, quad)
    pl.spy(D4,precision=0.0000000001, markersize=3)
    pl.show()      
    
    f_hat = SC.fct(f,f_hat)

    lhs = dot(Ix, Bmat) 
    rhs = dot(D2x,f_hat)

    v_hat[:-2] = linalg.solve(lhs[2:,:-2],rhs[2:])
    v = ST.ifst(v_hat, v)
    return v

#=====================================================================
#                 1D Helmholtz solver
#=====================================================================
def Helmholtz1D(alpha,N,f):
    
    f_hat = zeros(N)
    v_hat = zeros(N)
    v = zeros(N)
    
    f_hat = SC.fct(f,f_hat)
    
    Bmat = B_matrix(N)
    Ix, Imx = QI2(N)
    Dx, D2x, D4 = QIM(N, quad)

    lm = alpha*D2x-Ix
    lhs = dot(lm, Bmat) 
    rhs = dot(D2x,f_hat)

    v_hat[:-2] = linalg.solve(lhs[2:,:-2],rhs[2:])
    v = ST.ifst(v_hat, v)
    return v

#=====================================================================
#                 1D Biharmonic solver
#=====================================================================
def Biharmonic1D(alpha, beta, N, f):
    
    f_hat = zeros(N)
    v_hat = zeros(N)
    v = zeros(N)

    f_hat = SC.fct(f,f_hat)
    
    Smat = S_matrix(N)
    I4x, I4mx = QI4(N)
    Id = identity(N)
    Dx, D2, D4 = QIM(N, quad)
   
    h1 = dot(D4,I4mx)
    h2 = Id-alpha*D2+beta*h1
    h3 = dot(I4x,h2)
    lhs = dot(h3, Smat) 
    rhs = dot(D4,f_hat)

    v_hat[:-4] = linalg.solve(lhs[4:,:-4],rhs[4:])   
    v = SB.ifst(v_hat, v)
    return v

#=====================================================================
#                 2D Poisson solver (NOT FINISHED!)
#===================================================================== 
def Poisson2Dv1(M, quad):

    N = array([2**M, 2**(M-1)]) 
    L = array([2, 2*pi])
   
    kx = arange(N[0]).astype(float)
    ky = fftfreq(N[1], 1./N[1])

    Lp = array([2, 2*pi])/L
    K  = array(meshgrid(kx, ky, indexing='ij'), dtype=float)
    K[0] *= Lp[0]; K[1] *= Lp[1] 

    points, weights = ST.points_and_weights(N[0])
    x1 = arange(N[1], dtype=float)*L[1]/N[1]
    X = array(meshgrid(points, x1, indexing='ij'), dtype=float)

    Bmat = B_matrix(N[0])
    I2x, I2mx = QI2(N[0])
    Idx = identity(N[0])
    Idy = identity(N[1])
    Dx, D2, D4 = QIM(N[0], quad)

    p_hat = empty((N[0],N[1]), dtype="complex")
    u_hat = empty((N[0],N[1]), dtype="complex")

    p = empty((N[0],N[1]))
    u = empty((N[0],N[1]))

    u = -2.*sin(X[0]+X[1])
    u_hat = SC.fct(u,u_hat)

    u_hat = u_hat.reshape((N[0]*N[1]))
    p_hat = p_hat.reshape((N[0]*N[1]))

    alpha = K[1, 0]**2
    alphaI = diag(alpha)
    D2y = zeros(alphaI.shape)
    D2y[1:,1:] = linalg.inv(alphaI[1:,1:])

    lhs = kron(I2x,D2y) + kron(D2,Idy)
    rhsR = dot(kron(D2,D2y),u_hat.real)
    rhsI = dot(kron(D2,D2y),u_hat.imag)

    #pl.spy(lhs,precision=0.0000000001, markersize=3)
    #pl.show()    
    #sys.exit()
    
    p_hat.real[:-N[0]] = linalg.solve(lhs[N[0]:,:-N[0]],rhsR[N[0]:])
    p_hat.imag[:-N[0]] = linalg.solve(lhs[N[0]:,:-N[0]],rhsI[N[0]:])
    
    p_hat = p_hat.reshape((N[0],N[1]))        
    p = SC.ifct(p_hat, p)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[0], X[1], p)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p(x,y)')

    plt.show()

#=====================================================================
#                 2D Poisson solver (NOT FINISHED!)
#===================================================================== 
def Poisson2Dv2(M, quad):

    N = array([2**M, 2**(M-1)]) 
    L = array([2, 2])

    pointsx, weightsx = ST.points_and_weights(N[0])
    pointsy, weightsy = ST.points_and_weights(N[1])
    X = array(meshgrid(pointsx, pointsy, indexing='ij'), dtype=float)

    I2x, I2mx = QI2(N[0])
    I2y, I2my = QI2(N[1])
    Idx = identity(N[0])
    Idy = identity(N[1])
    Dx, D2x, D4x = QIM(N[0], quad)
    Dy, D2y, D4y = QIM(N[1], quad)
    
    p_hat = empty((N[0],N[1]))
    u_hat = empty((N[0],N[1]))

    p = empty((N[0],N[1]))
    u = empty((N[0],N[1]))

    u = -2.*sin(X[0]+X[1])
    u_hat = SC.fct(u,u_hat)

    u_hat = u_hat.reshape((N[0]*N[1]))
    p_hat = p_hat.reshape((N[0]*N[1]))

    alpha = K[1, 0]**2
    alphaI = diag(alpha)
    D2y = zeros(alphaI.shape)
    D2y[1:,1:] = linalg.inv(alphaI[1:,1:])

    lhs = kron(I2x,D2y) + kron(D2,Idy)
    rhsR = dot(kron(D2,D2y),u_hat.real)
    rhsI = dot(kron(D2,D2y),u_hat.imag)

    #pl.spy(lhs,precision=0.0000000001, markersize=3)
    #pl.show()    
    #sys.exit()
    
    p_hat.real[:-N[0]] = linalg.solve(lhs[N[0]:,:-N[0]],rhsR[N[0]:])
    p_hat.imag[:-N[0]] = linalg.solve(lhs[N[0]:,:-N[0]],rhsI[N[0]:])
    
    p_hat = p_hat.reshape((N[0],N[1]))        
    p = SC.ifct(p_hat, p)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[0], X[1], p)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p(x,y)')

    plt.show()

#=====================================================================
#                 2D Poisson solver (NOT FINISHED!)
#===================================================================== 
def Poisson2D(M, quad):

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

    Bmat = B_matrix(N[0])
    I2x, I2mx = QI2(N[0])
    Id = identity(N[0])
    Dx, D2, D4 = QIM(N[0], quad)

    p_hat = empty((N[0],N[1]), dtype="complex")
    u_hat = empty((N[0],N[1]), dtype="complex")
    #v_hat = empty((N[0],N[1]), dtype="complex")

    p = empty((N[0],N[1]))
    u = empty((N[0],N[1]))
    #v = empty((N[0],N[1]))

    u = -2.*sin(X[0]+X[1])
    #v = 2.*(1.-X[0]**2)
    u_hat = SC.fct(u,u_hat)
    #v_hat = SC.fct(v,v_hat)

    #v_hat *=1j*K[1]
    alpha = K[1, 0]**2
    alphaI = diag(alpha)

    lhs = I2x - dot(D2,alphaI)
    rhsR = dot(D2,u_hat.real)
    rhsI = dot(D2,u_hat.imag)
    print linalg.cond(D2[2:,:-2])
    #for i in range(N[1]):
        #l1 = dot(I2x, Dx)
        #l2 = dot(Bmat,Bmat)
        #l12 = dot(l1,l2)
        #l3 = dot(D2,Bmat)    

        #lhsR = dot(l12,u_hat[:,i].real) + dot(l3,v_hat[:,i].real)
        #lhsI = dot(l12,u_hat[:,i].imag) + dot(l3,v_hat[:,i].imag)

        #rhs = -alpha[i]*D2 + I2x
    pl.spy(lhs,precision=0.0000000001, markersize=3)
    pl.show()    
        #p_hat[:-2,i].real = linalg.solve(rhs[2:,:-2],lhsR[2:])
        #p_hat[:-2,i].imag = linalg.solve(rhs[2:,:-2],lhsI[2:])

    p_hat.real[:-3] = linalg.solve(lhs[3:,1:-2],rhsR[2:-1])
    p_hat.imag[:-3] = linalg.solve(lhs[3:,1:-2],rhsI[2:-1])    
    p = SC.ifct(p_hat, p)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[0], X[1], p)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p(x,y)')

    plt.show()

#=====================================================================
#            Finite difference 1D Poisson solver
#===================================================================== 
def FD_Poisson1D(n,f,a,b,method):
    """ Solve the two-point boundary value problem -u''(x)=f(x)
        on [0,1] with u'(0)=a and u(1)=b on n+1 grid points """

    from scipy.sparse import spdiags
    from scipy.sparse.linalg import spsolve

    h = 1.0/n                  # Grid spacing
    xi = linspace(0,1,n+1)     # Grid points
    rhs = h**2*f(xi[:-1])      # Evaluate forcing function
    rhs[-1] += b               # Dirichlet data
  
    if method == 1:            # first-order symmetric
        rhs[0] = -h*a 
        stencil = array((-1, 2, -1))   # Approx for 2nd derivative 
        diags = range(-1,2)            # Offsets of diags [-1,0,1]
        bands = tile(stencil,(n,1)).T  # Default bands
        bands[1,0] *= 0.5              # Modify for Neumman 
    elif method == 2:         # second-order nonsymmetric
         rhs[0] = -2*h*a 
         stencil = array((-1, 2, -1, 0)) # Approx for 2nd derivative 
         diags = range(-1,3)                # Offsets of diags [-1,0,1,2]
         bands = tile(stencil,(n,1)).T   # Default bands
         bands[1,0] = 3                     # one-sided differences
         bands[2,1] = -4
         bands[3,2] = 1
    elif method == 3:         # second-order symmetric 
         rhs[0] *= 0.5
         rhs[0] -= h*a 
         stencil = array((-1, 2, -1))    # Approx for 2nd derivative 
         diags = range(-1,2)                # Offsets of diags [-1,0,1]
         bands = tile(stencil,(n,1)).T   # Default bands
         bands[1,0] *= 0.5                  # Modify for Neumman 

    A = spdiags(bands,diags,n,n).tocsc()    # Form sparse matrix

    u = zeros(n+1)
    u[-1] = b                               # Set boundary value
    u[:-1] = spsolve(A,rhs)

    return xi, u    
    
    
if __name__ == '__main__':   
    
    #N = int(sys.argv[1])        # Number of unknowns
    M = 6
    N = 2**M
    test = "3"
    #solver = "Poisson1D"    
    #solver = "Helmholtz1D"
    #solver = "Biharmonic1D"
    solver = "Poisson2D"
    
    if test == "1":
        quad = "GC"
        # Get points and weights for Chebyshev weighted integrals

        # Get points and weights for Chebyshev weighted integrals
        BC1 = array([1,0,0, 1,0,0])
        BC2 = array([0,1,0, 0,1,0])
        BC3 = array([0,1,0, 1,0,0])
        SC = ChebyshevTransform(quad)
        ST = ShenBasis(BC1, quad)
        SN = ShenBasis(BC2, quad, Neumann = True)
        SR = ShenBasis(BC3, quad)
        SB = ShenBiharmonicBasis(quad, fast_transform=False)

        points, weights = SB.points_and_weights(N)
        x = points
        #from IPython import embed; embed() 
        #pl.spy(lhs,precision=0.0000000001, markersize=3)
        #pl.show()
        if solver == "Poisson1D":
            u = 2.*(1.-x**2)
            f = empty(N)
            f[:] = -4.
            v = Poisson1D(N,f)
            print linalg.norm(v-u,inf)
            plt.plot(x, u, x, v)
            plt.show()
        elif solver == "Helmholtz1D":
            alpha = 2.0
            u = sin(pi*x)
            f = (alpha+pi**2)*sin(pi*x)
            v = Helmholtz1D(alpha,N,f)
            print linalg.norm(v-u,inf)
            plt.plot(x, u, x, v)
            plt.show()
        elif solver == "Biharmonic1D":
            alpha = 2*N**2
            beta = N**4
            u = (sin(4*pi*x))**2#sin(2*pi*x)**2
            f = 2048*pi**4*sin(4*pi*x)**2 - 2048*pi**4*cos(4*pi*x)**2-alpha*(-32*pi**2*sin(4*pi*x)**2 + 32*pi**2*cos(4*pi*x)**2) + beta*(sin(4*pi*x))**2 #128*pi**4*(sin(2*pi*x)**2 - cos(2*pi*x)**2)
            v = Biharmonic1D(alpha,beta,N,f)
            print linalg.norm(v-u,inf)
            plt.plot(x, u, x, v)
            plt.show()
        
    elif test == "2":    
        from sympy import diff, lambdify, exp, sin
        from sympy.abc import u,x
        
        n = int(sys.argv[1])        # Number of unknowns
        xi = linspace(0,1,n+1)   # Grid points
        
        u = exp(x)/(1+x)            # Manufactured solution to test methods
        ux = diff(u,x)              # First derivative
        uxx = diff(ux,x)            # Second derivative
        f = -uxx                    # Forcing function
    
        U = lambdify(x,u,"numpy")   # make u a callable function 
        Ux = lambdify(x,ux,"numpy") # make ux a callable function
        F = lambdify(x,f,"numpy")   # make f a callable function

        a = Ux(0)                   # Neumann boundary value
        b = U(1)                    # Dirichlet boundary value
        
        # Compute solution with all three methods
        xi,u1 = FD_Poisson1D(n,F,a,b,1) 
        xi,u2 = FD_Poisson1D(n,F,a,b,2) 
        xi,u3 = FD_Poisson1D(n,F,a,b,3) 

        # Compute error for each approximation
        uex = U(xi)      # Evaluate exact solution
        e1 = uex-u1
        e2 = uex-u2
        e3 = uex-u3

        print 'h = ', 1.0/n
        print 'The max error of method 1 is:', max(abs(e1))
        print 'The max error of method 2 is:', max(abs(e2))
        print 'The max error of method 3 is:', max(abs(e3))
    
        plt.plot(xi,e1,xi,e2,xi,e3)
        plt.show()


    if test == "3":
        quad = "GC"
        BC1 = array([1,0,0, 1,0,0])
        BC2 = array([0,1,0, 0,1,0])
        BC3 = array([0,1,0, 1,0,0])
        SC = ChebyshevTransform(quad)
        ST = ShenBasis(BC1, quad)
        SN = ShenBasis(BC2, quad, Neumann = True)
        SR = ShenBasis(BC3, quad)
        SB = ShenBiharmonicBasis(quad, fast_transform=False)
        Poisson2Dv1(M, quad)
