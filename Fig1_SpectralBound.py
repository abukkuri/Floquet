import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import math
from numpy import log, exp, sqrt

pj = 0.1
pa = 0.1
trans = 0.1
fluct = 1
time = 80
k = 2

def beta(t):
    return fluct * np.sin(2 * np.pi * t) + fluct

IC = [1,1,1]

def evoLV(X,t):
    
    J = X[0]
    A = X[1]
    v = X[2]
        
    dJdt = beta(t)*(v-1)*A-pj*v**2*J-trans*J
    dAdt = trans*J-pa*A 
    dvdt = k*(-pj*v + (2*beta(t)*trans - 2*pa*pj*v + 2*pj**2*v**3 + 2*pj*trans*v)/(2*sqrt(4*beta(t)*trans*v - 4*beta(t)*trans + pa**2 - 2*pa*pj*v**2 - 2*pa*trans + pj**2*v**4 + 2*pj*trans*v**2 + trans**2)))
    
    dxvdt = np.array([dJdt, dAdt,dvdt])

    return dxvdt

intxv = np.array(IC)
n_steps = time*10
t_points = np.linspace(0, time, n_steps)
pop = odeint(evoLV, intxv, t_points)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Population Dynamics: Spectral Bound')
plt.plot(pop[:,0],lw=2,color='b',label='Juvenile')
plt.plot(pop[:,1],lw=2,color='r',label='Adult')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.xlim(0,n_steps)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Life History Strategy Dynamics: Spectral Bound")
plt.plot(pop[:,2],lw=2,color='k',label='Strategy')
plt.xlim(0,n_steps)
plt.xlabel('Time')
plt.ylabel('v')
plt.legend()
plt.tight_layout()
plt.show()

print(pop[:,0][-1])
print(pop[:,1][-1])
print(pop[:,2][-1])
