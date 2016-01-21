"""
Created on Tue 12 Jan 15:51:26 2016

@author: Diako Darian

"""

from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2, rfftn, irfftn, rfft, irfft
from mpi4py import MPI
from scipy.fftpack import dct

def rfft2_mpi(u, fu, num_processes):
    if num_processes == 1:
        fu[:] = rfft2(u, axes=(0,1))
        return fu    
    
    Uc_hatT[:] = rfft(u, axis=1)
    Uc_hatT[:, 0] += 1j*Uc_hatT[:, -1]
    
    # Align data in x-direction
    for i in range(num_processes): 
        U_send[i] = Uc_hatT[:, i*Np/2:(i+1)*Np/2]
            
    # Communicate all values
    comm.Alltoall([U_send, MPI.DOUBLE_COMPLEX], [U_recv, MPI.DOUBLE_COMPLEX])
    
    fu[:, :Np/2] = fft(U_recv, axis=0)
        
    # Handle Nyquist frequency
    if rank == 0:        
        f = fu[:, 0]        
        fft_x[0] = f[0].real;
        fft_x[1:N/2] = 0.5*(f[1:N/2]+conj(f[:N/2:-1]))
        fft_x[N/2] = f[N/2].real        
        fu[:N/2+1, 0] = fft_x[:N/2+1]        
        fu[N/2+1:, 0] = conj(fft_x[(N/2-1):0:-1])
        
        fft_y[0] = f[0].imag
        fft_y[1:N/2] = -0.5*1j*(f[1:N/2]-conj(f[:N/2:-1]))
        fft_y[N/2] = f[N/2].imag
        fft_y[N/2+1:] = conj(fft_y[(N/2-1):0:-1])
        
        comm.Send([fft_y, MPI.DOUBLE_COMPLEX], dest=num_processes-1, tag=77)
        
    elif rank == num_processes-1:
        comm.Recv([fft_y, MPI.DOUBLE_COMPLEX], source=0, tag=77)
        fu[:, -1] = fft_y 
        
    return fu

def irfft2_mpi(fu, u, num_processes):
    if num_processes == 1:
        u[:] = irfft2(fu, axes=(0,1))
        return u

    Uc_hat[:] = ifft(fu, axis=0)    
    U_sendr[:] = Uc_hat[:, :Np/2]

    comm.Alltoall([U_send, MPI.DOUBLE_COMPLEX], [U_recv, MPI.DOUBLE_COMPLEX])

    for i in range(num_processes): 
        Uc_hatT[:, i*Np/2:(i+1)*Np/2] = U_recv[i*Np:(i+1)*Np]
    
    if rank == num_processes-1:
        fft_y[:] = Uc_hat[:, -1]

    comm.Scatter(fft_y, plane_recv, root=num_processes-1)
    Uc_hatT[:, -1] = plane_recv
    
    u[:] = irfft(Uc_hatT, 1)
    return u

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

def fct(u, fu, ST, num_processes, comm):
    """Fast Cheb transform of x-direction, Fourier transform of y"""
    N = u.shape
    Np = N[1] / num_processes
    
    Uc_hat  = empty((N[0], Np), dtype="complex")
    Uc_hatT = empty((N[0], Np), dtype="complex")
    U_mpi   = empty((num_processes, Np, Np), dtype="complex")
    
    Uc_hatT[:] = fft(u, axis=1)
    n0 = U_mpi.shape[1]
    for i in range(num_processes):
        U_mpi[i] = Uc_hatT[:, i*n0:(i+1)*n0]
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [Uc_hat, MPI.DOUBLE_COMPLEX])
    fu = ST.fct(Uc_hat, fu)
    return fu

def ifct(fu, u, ST, num_processes, comm):
    """Inverse Cheb transform of x-direction, Fourier in y"""
    N = u.shape
    Np = N[1] / num_processes
    Uc_hat3 = empty((N[0], Np), dtype="complex")
    Uc_hatT = empty((Np, N[1]), dtype="complex")
    U_mpi   = empty((num_processes, Np, Np), dtype="complex")
    
    Uc_hat3[:] = ST.ifct(fu, Uc_hat3)
    comm.Alltoall([Uc_hat3, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    n0 = U_mpi.shape[1]
    for i in range(num_processes):
        Uc_hatT[:, i*n0:(i+1)*n0] = U_mpi[i]
    u[:] = ifft(Uc_hatT, axis=1)
    return u

def fct2( fj, cj):
    """Fast Chebyshev transform."""
    N = fj.shape[0]
    M = fj.shape[1]
    if quad == "GC":
        cj = dct(fj, 2, axis=0)
        cj /= N
        cj[0,:] /= 2
        cj = dct(cj, 2, axis=1)
        cj /=M
        cj[:,0] /= 2 
    elif quad == "GL":
        cj = dct(fj, 1, axis=0)/(N-1)
        cj[0,:] /= 2
        cj[-1,:] /= 2
        cj = dct(cj, 1, axis=1)/(M-1)
        cj[:,0] /= 2
        cj[:,-1] /= 2        
    return cj

def ifct2(fk, cj):
    """Inverse fast Chebyshev transform."""
    if quad == "GC":
        cj = 0.5*dct(fk, 3, axis=0)
        for i in range(fk.shape[1]):
            cj[:,i] += 0.5*fk[0,i]
        cj = 0.5*dct(cj, 3, axis=1)
        for i in range(fk.shape[0]):
            cj[i,:] += 0.5*cj[i,0]        
    elif quad == "GL":
        cj = 0.5*dct(fk, 1, axis=1)
        for i in range(fk.shape[0]):
            cj[i,:] += 0.5*fk[i,0]
            cj[i,::2] += 0.5*fk[i,-1]
            cj[i,1::2] -= 0.5*fk[i,-1]
        cj = 0.5*dct(cj, 1, axis=0)
        for i in range(fk.shape[1]):
            cj[:,i] += 0.5*cj[0,i]
            cj[::2,i] += 0.5*cj[-1,i]
            cj[1::2,i] -= 0.5*cj[-1,i]            
    return cj