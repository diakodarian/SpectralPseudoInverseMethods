"""
Created on Tue 12 Jan 15:51:26 2016

@author: Diako Darian

"""

from numpy import *

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
#      Chebyshev Differentiation Matrix in Spectral Space
#=====================================================================
def chebDiff(N,quad):
    D = zeros((N,N))
    ck = chebNormalizationFactor(N, quad)
    
    for k in xrange(N):
        for j in xrange((k+1),N,2):
            D[k,j] = (2./ck[k])*j
    return D       

#=====================================================================
#  Chebyshev Second Order Differentiation Matrix in Spectral Space
#=====================================================================
def chebDiff2(N,quad):
    D = zeros((N,N))
    ck = chebNormalizationFactor(N, quad)
    
    for k in xrange(N):
        for j in xrange((k+2),N):
            if (k+j)%2 == 0:
                D[k,j] = (1./ck[k])*j*(j**2-k**2)
    return D       

#=====================================================================
#       Tranfsorm of Shen_Robin to Chebyshev
#=====================================================================
def B_matrix(N,BC):
    
    # Wavenumbers: 
    K = wavenumbers(N)
    # Shen coefficients for the basis functions
    a_k, b_k = shenCoefficients(K, BC)
    #a_j, b_j = shenCoefficients(K, BC2)
    # Chebyshev normalization factor
    #ck = chebNormalizationFactor(N, ST.quad)
    Bmat = zeros((N,N))
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                Bmat[k,j] = 1.0
            elif k == j+1:    
                Bmat[k,j] = a_k[j] 
            elif k == j+2:    
                Bmat[k,j] = b_k[j] 
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