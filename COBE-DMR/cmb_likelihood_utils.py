# ****************************************************************
#       Utility module for cmb_likelihood.py
# ****************************************************************
#
#   The module contains the following functions, usage being
#   returned_quantity = function(arguments)
#
#      N_cov    = get_noise_cov(rms)
#      F_cov    = get_foreground_cov(x,y,z)
#      C_ell    = get_C_ell_model(Q,n,lmax)
#      polys    = get_legendre_coeff(lmax)
#      P_ell_ij = get_legendre_mat(lmax,x,y,z)
#      S_cov    = get_signal_cov()
#      lnL      = get_lnL()
#
#


import numpy as np
from scipy.special import legendre, gamma
import scipy.linalg as spl
from stat import S_IEXEC

def get_noise_cov(rms):
    """
    To be completed:
    Compute the noise covariance matrix from the pixel standard deviations
    """
    # 1: Compute a matrix with element (i,i) = sigma_i^2
    N_cov = np.diag(rms**2)

    return N_cov


def get_foreground_cov(x,y,z):
    """
    Computing the foreground template covariance matrix, to marginalize over
    any monopole and dipole components in the map
    F_cov = large_value * sum(template_cov), where
    template_cov = np.outer(f, f^t). 
    For the monopole template, f is a constant.
    To account for a dipole of any orientation, we use each of the unit vector 
    components as a dipole template.
    """
    large_value = 1.0e3
    monopole = np.ones((len(x),len(x)))
    dipole = np.outer(x,x) + np.outer(y,y) + np.outer(z,z)
    return large_value * (monopole + dipole)

def get_C_ell_model(Q,n,lmax):
    """
    To be completed:
    Recursively compute a model power spectrum, C_ell, given the amplitude and
    spectral index parameters Q and n, on the range ell in [0,lmax],
    but with monopole and dipole terms set to 0.
    #C_ell[l] = gamma(l + ((n-1.0)/2.0))/gamma(l+((5.0-n)/2.0))
    #C_ell = C_ell*(4.0*np.pi/5.0)*(Q**2)*(gamma((9.0-n)/2.0)/gamma((3.0+n)/2.0))
    #C_ell[2] = ((4.0*np.pi)/(5.0*Q**4))
    """
    # 1: Define array for power spectrum
    C_ell = np.zeros(lmax+1)
    # 2: Compute quadrupole (ell=2) term
    C_ell[2] = (4.0*np.pi)/5.0*Q**2

    # 3: Compute multipoles 3 through lmax recursively
    for l in range(3,lmax+1):
        
        C_ell[l] = C_ell[l-1]*((l + (n-1.0)/2.0)/(l+(5.0-n)/2.0))

    #print C_ell
    
    return C_ell 

def get_legendre_coeff(lmax):
    '''
    Helper routine for get_legendre_full. Computes Legendre polynomial
    coefficients for each multipole l, using scipy.special.legendre.
    Stores the result in a list of poly1d objects.
    Each such object returns the polynomial value when called with a 
    cos(theta) argument: P_ell = pol[l](costheta)
    '''
    leg = []
    for l in range(lmax+1):
        leg.append(legendre(l))
    return leg


def get_legendre_mat(lmax,x,y,z):
    '''
    Computing the full set of Legendre polynomial values needed to build the 
    signal covariance matrix.
    Uses helper function get_legendre_coeff for polynomial coefficients, and
    assembles a matrix of dimensions (ndata, ndata, lmax+1)
    '''
    leg = get_legendre_coeff(lmax)
    pos_vec = np.vstack([x,y,z]).T
    costheta =  np.dot(pos_vec,pos_vec.T)

    ndata = len(x)
    p_ell_ij = np.zeros((ndata,ndata,lmax+1))
    for l in range(lmax+1):
        p_ell_ij[:,:,l] = leg[l](costheta)
        
    return p_ell_ij


def get_signal_cov(C_ell, beam, pixwin, p_ell_ij):
    '''
    To be completed:
    Compute a (ndata,ndata) signal covariance matrix using the
    model power spectrum, instrument beam and pixel window function, and
    precomputed Legendre polynomials as input

    Hint: This can be done using a triple for-loop, but it is not necessary.
    Using NumPy array operations may get you a significant speed-up.
    # 1: Compute all the elements of the sum over ell, as arrays     
    # 2: Assemble a single array with all the ell terms which are independent of (i,j)        
    # 3: Compute the covariance matrix by an appropriate inner product
 
 
    #First self-made method
    #S_cov = np.zeros((len(p_ell_ij),len(p_ell_ij[0])))#np.zeros_like(p_ell_ij[:,:,0])
    #for l in range(lmax):
    #    S_cov[:,:] += (2*l+1)*(beam[l]*pixwin[l])**2*C_ell[l]*p_ell_ij[:,:,l] #S_l[l]*

    #hans-petter's method
    #S_cov = np.zeros_like(p_ell_ij[:,:,0])
    #l = np.arange(lmax+1)
    #for l in range(lmax):
    #    S_cov[:,:] += (2.*l + 1.)*(np.power(beam[l]*pixwin[l],2))*C_ell[l]*p_ell_ij[:,:,l]

    #daniel's method
    S_cov = np.zeros_like(p_ell_ij[:,:,0])    
    for i,j in zip(range(len(p_ell_ij)),range(len(p_ell_ij[0]))): #zip(range(p_ell_ij.shape[0]),range(p_ell_ij.shape[1])):
        for l in range(lmax):
            S_cov[i,j] += (2*l + 1)*(beam[l]*pixwin[l])**2*C_ell[l]*p_ell_ij[i,j,l] 
    
    '''
    lmax = len(C_ell)      
    ll = np.arange(lmax)

    ell_dep_array = (2*ll + 1)*(beam*pixwin)**2*C_ell#np.array([((2*ell + 1)*(beam*pixwin)**2*C_ell),]*(p_ell_ij.shape[1])) 
    S_cov = np.einsum('ijl,...l->ij',p_ell_ij,ell_dep_array)

    #print S_cov
 
    return S_cov/(4.*np.pi)

def get_lnL(data, cov):
    '''
    To be completed:
    Compute the quantity -2*lnL using the complete covariance matrix
    C = S+N+F, and the input data vector.

    Hint: This can be done directly by inverting the covariance matrix
    and computing a determinant, calling suitable NumPy or SciPy routines.
    Significant speedup can be gained by rather using the Cholesky decomposition
    method discussed in the project description notes.
     
    #depricated
    #L = np.array(L)
    #c = np.mean(np.diag(L))
    #L = L/c
    #print c
    #print len(L[0])*np.log(c)
    
    #detA = 2*np.linalg.slogdet(L)[1]
    #detA = len(L[0])*np.log(c)*detL
    #detA = detA**2
    '''
 
    # 1: Cholesky-decompose C into a lower triangular matrix L, using scipy.linalg.cholesky
    L = spl.cholesky(cov, lower=True)

    # 2: Compute log(det(C)) from L
    detA = 2*np.sum(np.log(np.diag(L)))
    #print "det =",detA

    # 3: Solve for L^-1 d using scipy.linalg.solve_triangular 
    x = spl.solve_triangular(L,np.identity(len(L[0])),lower=True) #data
    x = np.array(x)

    # 4: Assemble -2*lnL using the components just computed
    chi_inv = np.dot(np.transpose(x),x)
    chi_sq = np.dot(np.dot(np.transpose(data),chi_inv),data)

    print "chi_sq = ",chi_sq
    
    #if not -1e3<chi_sq<8e4:
    #    print "There may be something wrong, got chi value of %g" %chi_sq
        
    result = chi_sq + detA
    print "lnL=",result
    return result

