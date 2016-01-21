import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import scipy.sparse.linalg as la
import SFTc
from numpy import linalg, inf

"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},

where a_k and b_k are calculated from 

   1) a_minus U(-1) + b_minus U(-1) = c_minus 
and
   2) a_plus U(1) + b_plus U(1) = c_plus

The array BC = [a_minus, b_minus, c_minus, a_plus, b_plus, c_plus] that determines the 
boundary conditions must be given. The code automatically calculates a_k and b_k, and it gives the
Shen transform.

In particular, for homogeneous Dirichlet boundary conditions we have:
 
    a_k = 0  and  b_k = -1
    
For homogeneous Neumann boundary conditions:
    
     a_k = 0  and  b_k = -(k/k+2)**2 
     
For Robin boundary conditions:

     a_k = \pm 4*(k+1)/((k+1)**2 + (k+2)**2)  and  
     b_k = -(k**2 + (k+1)**2)/((k+1)**2 + (k+2)**2)

Here a_k is positive for Dirichlet BC at x = -1 and Neumann BC at x = +1,
and it is negative for Neumann BC at x = -1 and Dirichlet BC at x = +1.

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.
"""
pi, zeros, ones, cos = np.pi, np.zeros, np.ones, np.cos

dct1 = dct
def dct(x, i, axis=0):
    if np.iscomplexobj(x):
        xreal = dct1(x.real, i, axis=axis)
        ximag = dct1(x.imag, i, axis=axis)
        return xreal + ximag*1j
    else:
        return dct1(x, i, axis=axis)

class ChebyshevTransform(object):
    
    def __init__(self, quad="GC", fast_transform=True): 
        self.quad = quad
        self.fast_transform = fast_transform
        self.points = None
        self.weights = None
        
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix.
        self.V = n_cheb.chebvander(self.points, N-1).T
    
    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GC":
            points = n_cheb.chebpts2(N)[::-1]
            weights = np.zeros((N))+np.pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2
        elif self.quad == "GL":
            points, weights = n_cheb.chebgauss(N)
        return points, weights
        
    def chebDerivativeCoefficients(self, fk, fj):
        SFTc.chebDerivativeCoefficients(fk, fj)  
        return fj

    def chebDerivative_3D(self, fj, fd, fk, fkd):
        fk = self.fct(fj, fk)
        fkd = SFTc.chebDerivativeCoefficients_3D(fk, fkd)
        fd = self.ifct(fl, fd)
        return fd
    
    def fastChebDerivative(self, fj, fd, fk, fkd):
        """Compute derivative of fj at the same points."""
        fk = self.fct(fj, fk)
        fkd = self.chebDerivativeCoefficients(fk, fkd)
        fd  = self.ifct(fkd, fd)
        return fd
        
    def fct(self, fj, cj):
        """Fast Chebyshev transform."""
        N = fj.shape[0]
        if self.quad == "GL":
            cj = dct(fj, 2, axis=0)
            cj /= N
            cj[0] /= 2        
        elif self.quad == "GC":
            cj = dct(fj, 1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
        return cj

    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GL":
            cj = 0.5*dct(fk, 3, axis=0)
            cj += 0.5*fk[0]
        elif self.quad == "GC":
            cj = 0.5*dct(fk, 1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]
        return cj
    
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        if self.fast_transform:
            N = fj.shape[0]
            if self.quad == "GL":
                fk = dct(fj, 2, axis=0)*np.pi/(2*N)
            elif self.quad == "GC":
                fk = dct(fj, 1, axis=0)*np.pi/(2*(N-1))
        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:] = np.dot(self.V, fj*self.weights)
            
        return fk

class ShenBasis(ChebyshevTransform):

    def __init__(self, BC, quad="GC", fast_transform=True, Neumann = False):
        self.quad = quad
        self.fast_transform = fast_transform
        self.Neumann = Neumann
        self.BC = BC
        self.points = None
        self.weights = None
        self.N = -1
        
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k  = self.wavenumbers(N)
        ak, bk = self.shenCoefficients(k, self.BC)
        # Build Vandermonde matrix. Note! N points in real space gives N-2 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T + ak[:, np.newaxis]*n_cheb.chebvander(self.points, N-2)[:, 1:].T + bk[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T

    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-2).astype(np.float)
        else:
            kk = np.mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

    def chebNormalizationFactor(self, N, quad):
	if self.quad == "GL":
            ck = ones(N[0]-2); ck[0] = 2
        elif self.quad == "GC":
            ck = ones(N[0]-2); ck[0] = 2; ck[-1] = 2
        return ck
    
    def shenCoefficients(self, k, BC):
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

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product 
        B u_hat = sum_{j=0}{N} u_j phi_k(x_j) w_j,
        for Shen basis functions given by
        phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}
        """
        if self.fast_transform:
            k  = self.wavenumbers(fj.shape)
            fk = self.fastChebScalar(fj, fk)
            ak, bk = self.shenCoefficients(k, self.BC)
            
            fk_tmp = fk
            fk[:-2] = fk_tmp[:-2] + ak*fk_tmp[1:-1] + bk*fk_tmp[2:]
        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:-2] = np.dot(self.V, fj*self.weights)    
        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform for general BC.
        """
        if len(fk.shape)==3:
            k = self.wavenumbers(fk.shape)
            w_hat = np.zeros(fk.shape, dtype=fk.dtype)
        elif len(fk.shape)==1:
            k = self.wavenumbers(fk.shape[0])
            w_hat = np.zeros(fk.shape[0])
        ak, bk = self.shenCoefficients(k, self.BC)
        w_hat[:-2] = fk[:-2]
        w_hat[1:-1] += ak*fk[:-2]
        w_hat[2:]   += bk*fk[:-2]
            
        if self.Neumann:
            w_hat[0] = 0.0
        fj = self.ifct(w_hat, fj)
        return fj
        
    def fst(self, fj, fk):
        """Fast Shen transform for general BC.
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        k = self.wavenumbers(N) 
        k1 = self.wavenumbers(N+1) 
        ak, bk = self.shenCoefficients(k, self.BC)
        ak1, bk1 = self.shenCoefficients(k1, self.BC)
        
        if self.quad == "GL":
            ck = ones(N-2); ck[0] = 2
        elif self.quad == "GC":
            ck = ones(N-2); ck[0] = 2; ck[-1] = 2  
        
        a = (pi/2)*(ck + ak**2 + bk**2)
        b = ones(N-3)*(pi/2)*(ak[:-1] + ak1[1:-1]*bk[:-1])
        c = ones(N-4)*(pi/2)* bk[:-2]

        if len(fk.shape) == 3:
            if self.Neumann: 
                fk[1:-2] = SFTc.PDMA_3D_complex(a[1:], b[1:], c[1:], fk[1:-2])
            else:
                fk[:-2] = SFTc.PDMA_3D_complex(a, b, c, fk[:-2])
        elif len(fk.shape) == 1:
            if self.Neumann: 
                fk[1:-2] = SFTc.PDMA_1Dl(a[1:], b[1:], c[1:], fk[1:-2])
            else:
                fk[:-2] = SFTc.PDMA_1Dl(a, b, c, fk[:-2])
            
        return fk    

class ShenBiharmonicBasis(ChebyshevTransform):
    
    def __init__(self, quad="GC", fast_transform=False):
        ChebyshevTransform.__init__(self, quad, fast_transform)
        self.factor1 = None
        self.factor2 = None
        self.w_hat = None
        
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        #from IPython import embed; embed()
        # Build Vandermonde matrix.
        self.V = n_cheb.chebvander(self.points, N-5).T - (2*(k+2)/(k+3))[:, np.newaxis]*n_cheb.chebvander(self.points, N-3)[:, 2:].T + ((k+1)/(k+3))[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 4:].T
        
    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-4).astype(float)
        
        else:
            kk = np.mgrid[:N[0]-4, :N[1], :N[2]].astype(float)
            return kk[0]

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        """        
        if self.fast_transform:
            k  = self.wavenumbers(fj.shape)
            Tk = fk.copy()
            Tk = self.fastChebScalar(fj, Tk)
            fk[:] = Tk[:]
            fk[:-4] -= 2*(k+2)/(k+3) * Tk[2:-2]
            fk[:-4] += ((k+1)/(k+3)) * Tk[4:]

        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:-4] = np.dot(self.V, fj*self.weights)
            
        fk[-4:] = 0
        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        if self.w_hat is None:
            self.w_hat = fk.copy()
        elif not self.w_hat.shape == fk.shape:
            self.w_hat = fk.copy()

        recreate = False
        if isinstance(self.factor1, np.ndarray):
            if not self.factor1.shape == fk.shape:
                recreate = True
            
        if self.factor1 is None:
            recreate = True
            
        if recreate:
            if len(fk.shape)==3:
                k = self.wavenumbers(fk.shape)                
            elif len(fk.shape)==1:
                k = self.wavenumbers(fk.shape[0])
            self.factor1 = -2*(k+2)/(k+3)
            self.factor2 = (k+1)/(k+3)
            
        self.w_hat[:] = 0
        self.w_hat[:-4] = fk[:-4]
        self.w_hat[2:-2] += self.factor1*fk[:-4]
        self.w_hat[4:]   += self.factor2*fk[:-4]
        fj = self.ifct(self.w_hat, fj)
        return fj

    def fst(self, fj, fk):
        """Fast Shen transform .
        """
        fk = self.fastShenScalar(fj, fk)
        N = fj.shape[0]
        k = self.wavenumbers(N)
        N -= 4
        ckp = ones(N)
        if self.quad == "GL":
            ck = ones(N); ck[0] = 2
        elif self.quad == "GC":
            ck = ones(N); ck[0] = 2; ck[-1] = 2  
        c = (ck + 4*((k+2)/(k+3))**2 + ckp*((k+1)/(k+3))**2)*pi/2.
        d = -((k[:-2]+2)/(k[:-2]+3) + (k[:-2]+4)*(k[:-2]+1)/((k[:-2]+5)*(k[:-2]+3)))*pi
        e = (k[:-4]+1)/(k[:-4]+3)*pi/2
        b = d.copy()
        a = e.copy()
         
        if len(fk.shape) == 3:
            fk[:-4] = SFTc.PDMA_2Version(a, b, c, d, e, fk[:-4], fk[:-4])
        elif len(fk.shape) == 1:
            fk[:-4] = SFTc.PDMA_1D_2Version(a, b, c, d, e,fk[:-4],fk[:-4])
            
        return fk  
  
if __name__ == "__main__":
    
    N = 2**6
    BC = np.array([1,0,0, 1,0,0])
    af = np.zeros(N, dtype=np.complex)
    SR = ShenBasis(BC, quad="GC")
    pointsr, weightsr = SR.points_and_weights(N)
    x = pointsr
    a = cos(pi*x) + x*x#x**2+np.cos(np.pi*x)#x -(8./13.)*(-1 + 2*x**2) - (5./13.)*(-3*x + 4*x**3) # Chebyshev polynomial that satisfies the Robin BC
    
    f_hat = zeros(N, dtype=np.complex)
    f_hat = SR.fst(a, f_hat)
    a = SR.ifst(f_hat, a)
    
    af = SR.fst(a, af)
    a0 = a.copy()
    a0 = SR.ifst(af, a0)
    print "Error in Shen-Robin transform: ",linalg.norm((a - a0), inf) 
    # Out: Error in Shen-Robin transform: 4.57966997658e-16
    assert np.allclose(a0, a)
    
    
    N = 2**6
    SB = ShenBiharmonicBasis(quad="GL", fast_transform=False)
    points, weights = SB.points_and_weights(N)
    x = points
    f = np.sin(2*pi*x)**2#(1-x**2)*np.sin(2*pi*x)        
    fj = f
    u0 = zeros(N, dtype=float)
    u0 = SB.fst(fj, u0)
    u1 = u0.copy()
    u1 = SB.ifst(u0, u1)
    print "Error in Shen-Biharmonic transform: ",linalg.norm((fj - u1), inf)    
    assert np.allclose(u1, fj)