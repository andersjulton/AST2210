# Simple contour plotting script for visualizing the lnL computed by cmb_likelihood.py. 
import numpy as np
import matplotlib.pyplot as plt
import sys
import pylab

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
    # 2.3, 6.17 and 11.8 from its minimum value. 0.1 is close to the peak. 
    my_levels = [0.1, 2.3, 6.17, 11.8]
    cs = plt.contour(qgrid,ngrid, -2.*lnL, levels=my_levels) #, colors='k'
    plt.grid()
    plt.xlabel("Q") 
    plt.ylabel("n",rotation=0) 
    plt.title("")
    labels = ["Peak", "3 Sigma", "2 Sigma", "1 Sigma"]
    for i in range(len(labels)): 
        cs.collections[i].set_label(labels[i])
    plt.legend(loc='upper right')
    plt.show()


    
    
    # exponentiate, and subtract off the biggest value to avoid overflow
    P = np.exp(lnL_init-np.max(lnL_init))  
    dn = n_values[1] - n_values[0] #(n_values[len(n_values)-1] - n_values[0])/len(n_values)
    dQ = Q_values[1] - Q_values[0] #(Q_values[1] - Q_values[0])/len(Q_values)
    
    # Normalize to unit integral    
    P = P / (np.sum(P)*dn*dQ)    
    
    # Compute marginal distribution, P(n|d) 
    P_n = np.zeros(len(n_values))
    P_Q = np.zeros(len(Q_values))
    for i in range(0, len(n_values)):
        P_n[i] = np.sum(P[:,i])  
        P_Q[i] = np.sum(P[i,:])
    
    # We now normalize 
    P_n = P_n / (sum(P_n) * dn)  
    P_Q = P_Q / (sum(P_Q) * dQ)  
    
    # We can now compute the mean mu and  the uncertainty sigma
    mu_n = sum(P_n * n_values)*dn                    
    sigma_n = np.sqrt(sum(P_n*(n_values-mu_n)**2)*dn)                 
    print "   mu_n =",mu_n  
    print "sigma_n =",sigma_n,"\n"
    
    mu_Q = sum(P_Q * Q_values)*dQ  
    sigma_Q = np.sqrt(sum(P_Q * (Q_values-mu_Q)**2)*dQ) 
    print "   mu_Q =",mu_Q
    print "sigma_Q =",sigma_Q
    
    plt.plot(n_values,P_n)
    plt.xlabel("n") 
    plt.ylabel("P(n)") 
    plt.axis([-1,3, min(P_n), max(P_n)*1.1])
    plt.legend(["P(n)"]) 
    plt.show() 
    plt.plot(Q_values,P_Q) 
    plt.axis([0,50,min(P_Q), max(P_Q)*1.1])  
    plt.ylabel("P(Q)")  
    plt.legend(["P(Q)"]) 
    plt.show() 