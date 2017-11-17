import numpy as np


#Metropolis-Hastings algorithm
u = np.random.rand(N)
y = np.zeros(N)
y[0] = np.random.normal(mu,sigma)

for i in range(N-1):
    ynew = np.random.normal(mu,sigma)
    alpha = min(1, p(ynew)*q(y[i])/(p(y[i])*q(ynew)))
    
    if u[i] < alpha:
        y[i+1] = ynew
    else: 
        y[i+1] = y[i]