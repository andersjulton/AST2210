# Simple contour plotting script for visualizing the lnL computed by
# cmb_likelihood.py. 
# For convenience, it takes as input either the .npy file or the .dat file.
# In the .dat case you also have to supply the number of grid points in each 
# direction so that we can define the grid correctly.

import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    if len(sys.argv)<2:
        print 'Wrong number if input arguments.'
        print 'Usage: python plot_contours.py resultfile.npy'
        print 'Or: python plot_contours.py resultfile.dat numpoints_Q numpoints_n'
        sys.exit()

    inputfile = sys.argv[1]
    if inputfile[inputfile.rfind('.'):]=='.npy':
        a = np.load(inputfile)
        Q_values = a[0,:]
        n_values = a[1,:]
        lnL_init = a[2:,:]
        qgrid, ngrid = np.meshgrid(Q_values,n_values, indexing='ij')

    else: # ascii file
        n_Q = int(sys.argv[2])
        n_n = int(sys.argv[3])
        a = np.loadtxt(inputfile)
        qgrid = np.reshape(a[:,0],(n_Q, n_n))
        ngrid = np.reshape(a[:,1],(n_Q, n_n))
        lnL_init = np.reshape(a[:,2],(n_Q, n_n))
        Q_values = qgrid[:,0]
        n_values = ngrid[0,:]
    #lnL = np.zeros_like(lnL_init)
    lnL = lnL_init - np.amax(lnL_init) # arbitrarily "normalizing" to make the numbers more manageable

    # For a Gaussian distribution, the 1, 2 and 3 sigma (68%, 95% and
    # 99.7%) confidence regions correspond to where -2 lnL increases by
    # 2.3, 6.17 and 11.8 from its minimum value. 0.1 is close to the
    # peak. 
    my_levels = [0.1, 2.3, 6.17, 11.8]
    cs = plt.contour(qgrid,ngrid, -2.*lnL, levels=my_levels) #, colors='k'
    plt.grid()
    #plt.legend(["0.1", "2.3", "6.17", "11.8"])
    plt.show()


    
    
    # First, exponentiate, and subtract off the biggest
    # value to avoid overflow and find P(d| Q,n)    
    P = np.exp(lnL_init-np.max(lnL_init))  
    dn = n_values[1] - n_values[0]
    dQ = Q_values[1] - Q_values[0]
    
    # Normalize to unit integral    
    P = P / (np.sum(P)*dn*dQ)    
    
    # Compute marginal distribution, P(n|d)
    n_num = len(n_values)
    P_n = np.zeros(n_num)
    for i in range(0, n_num):
        # Here we are integrating
        # to get P_n, for P_Q be sure to integrate
        # over the other dimension in P!
        P_n[i] = np.sum(P[:,i])  
    
    # We now normalize (note the dn - replace for P_Q!)   
    P_n = P_n / (sum(P_n) * dn)  
    
    # We can now compute the mean    
    mu_n = sum(P_n * n_values)*dn                    
    print "mu_n=",mu_n
    
    # And lastly, we find the uncertainty   
    sigma_n = np.sqrt(sum(P_n * (n_values-mu_n)**2)*dn) 
    print "sigma_n=",sigma_n
    
