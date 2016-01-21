"""
Created on Tue 12 Jan 15:51:26 2016

@author: Diako Darian


1D solvers for Poisson, Helmholtz and Biharmonic equations 
by using the pseudo-inverse technics.

Two Poisson-solverse are given, one for Dirichlet and one
for Neumann boundary conditions  
"""

from numpy import *
from shentransform import ShenBasis, ShenBiharmonicBasis, ChebyshevTransform
from FFTChebTransforms import *
from SpectralDiff import *
import SFTc
import matplotlib.pylab as pl
import scipy.sparse as sps
from scipy import sparse
import sys

#=====================================================================
#                 1D Poisson solver Dirichlet BCs
#=====================================================================
def Poisson1D(N, f, quad, SC, ST, BC):
    
    f_hat = zeros(N)
    v_hat = zeros(N)
    v = zeros(N)
    
    Bmat = B_matrix(N, BC)
    Ix, Imx = QI2(N)
    Dx, D2x, D4 = QIM(N, quad)
    #pl.spy(Dx,precision=0.0000000001, markersize=3)
    #pl.show()      
    
    f_hat = SC.fct(f,f_hat)

    lhs = dot(Ix, Bmat) 
    rhs = dot(D2x,f_hat)

    v_hat[:-2] = linalg.solve(lhs[2:,:-2],rhs[2:])
    v = ST.ifst(v_hat, v)
    return v

#=====================================================================
#         1D Poisson solver - Neumann BCs
#=====================================================================
def Poisson1DNeumann(N, f, quad, SC, SN, BC):
    
    f_hat = zeros(N)
    v_hat = zeros(N)
    v = zeros(N)
    
    Bmat = B_matrix(N, BC)
    Ix, Imx = QI2(N)
    Dx, D2x, D4 = QIM(N, quad)

    f_hat = SC.fct(f,f_hat)

    lhs = dot(Ix, Bmat) 
    rhs = dot(D2x,f_hat)
    #pl.spy(lhs,precision=1.e-16, markersize=3)
    #pl.show()      
    #sys.exit()
    v_hat[1:-2] = linalg.solve(lhs[3:,1:-2],rhs[3:])
    v = SN.ifst(v_hat, v)
    return v

#=====================================================================
#                 1D Helmholtz solver
#=====================================================================
def Helmholtz1D(alpha, N, f, quad, SC, ST, BC):
    
    f_hat = zeros(N)
    v_hat = zeros(N)
    v = zeros(N)
    
    f_hat = SC.fct(f,f_hat)
    
    Bmat = B_matrix(N, BC)
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
def Biharmonic1D(alpha, beta, N, f, SC, SB):
    
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


if __name__ == '__main__':   
    print "==============================="
    print "          Menu                 "
    print "==============================="    
    print "0 - Poisson Neumann BCs"
    print "1 - Poisson Dirichlet BCs"
    print "2 - Helmholtz Dirichlet BCs" 
    print "3 - Biharmonic Dirichlet BCs"    
    print "==============================="
    test = raw_input('Your choice: ')

    M = 2**3
    quad = "GL"
    BC1 = array([1,0,0, 1,0,0])
    BC2 = array([0,1,0, 0,1,0])
    BC3 = array([0,1,0, 1,0,0])
    SC = ChebyshevTransform(quad)
    ST = ShenBasis(BC1, quad)
    SN = ShenBasis(BC2, quad, Neumann = True)
    SR = ShenBasis(BC3, quad)
    SB = ShenBiharmonicBasis(quad, fast_transform=False)

    if test == "0":
        v_exact = zeros(M)
        f = zeros(M)
        points, weights = SN.points_and_weights(M)
        f[:] = (-8./3.)*points#-((pi/2.)**2)*sin(pi*points/2.)
        v_exact = -(4./9.)*(points**3 - 3.*points)#sin(pi*points/2.)
        v = Poisson1DNeumann(M, f, quad, SC, SN, BC2)

        print "Error: ", linalg.norm(v-v_exact,inf)
        assert allclose(v,v_exact)
        pl.plot(points, v_exact, points,v)
        pl.show()              
    elif test == "1":
        v_exact = zeros(M)
        f = zeros(M)
        points, weights = ST.points_and_weights(M)
        f[:] = -4.
        v_exact = 2*(1.-points**2)
        v = Poisson1D(M, f, quad, SC, ST, BC1)
        print "Error: ", linalg.norm(v-v_exact,inf)
        assert allclose(v,v_exact)
        pl.plot(points, v_exact, points,v)
        pl.show()
    elif test == "2":    
        v_exact = zeros(M)
        f = zeros(M)
        points, weights = ST.points_and_weights(M)
        alpha = 2.
        v_exact = 2*(1.-points**2)
        f[:] = 2.*v_exact+4.
        v = Helmholtz1D(alpha, M, f, quad, SC, ST, BC1)
        print "Error: ", linalg.norm(v-v_exact,inf)
        assert allclose(v,v_exact)
        pl.plot(points, v_exact, points,v)
        pl.show()
    elif test == "3":
        v_exact = zeros(M)
        f = zeros(M)
        points, weights = ST.points_and_weights(M)
        alpha = 2.
        beta = .1
        v_exact = (sin(4.*pi*points))**2
        f = -2**11*pi**4*cos(8*pi*points)-alpha*2**5*pi**2*cos(8.*pi*points) + beta*v_exact
        v = Biharmonic1D(alpha, beta, M, f, SC, SB)
        print "Error: ", linalg.norm(v-v_exact,inf)
        assert allclose(v,v_exact)
        pl.plot(points, v_exact, points,v)
        pl.show()
    
    