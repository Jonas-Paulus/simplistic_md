

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
n_frames = 100
def integrator_Euler(x,v,H, dH):
    # a bad integrator, see how bad it is
    
    a = -dH(x)
    v = v + a*dt
    x = x + v*dt
    return(x,v)

def integrator_velocity_verlet(x,v,H, dH):
    # this integrator is valid
    #eps = 0.00001 #precision for numerical differentiation
    a_init = -dH(x)
    x_new = x + v*dt + 0.5*a_init*dt*dt
    a_new = -dH(x_new)
    v_new = v + (a_init + a_new)/2 * dt
    return(x_new, v_new)

#standard lennard jones potential
H_LJ = lambda x: x**-12-x**-6 
dH_LJ = lambda x: -12*x**-13 + 6*x**-7

#initial configuration
x = 0.8
v = 0

H = H_LJ
dH = dH_LJ
integrator = integrator_velocity_verlet

#to keep track of what is happening
history_x = np.zeros(n_frames)
history_v = np.zeros(n_frames)
history_Epot = np.zeros(n_frames)
history_Ekin = np.zeros(n_frames)
#md loop
for i in range(n_frames):
    x,v = integrator(x,v,H, dH)
    history_x[i] = x
    history_v[i] = v
    history_Epot[i] = H(x)
    history_Ekin[i] = 0.5*v*v

plt.plot(history_Epot, label = "E_pot")
plt.plot(history_Ekin, label = "E_kin")
plt.plot(history_Epot + history_Ekin, label = "E_ges")
plt.legend()
plt.show()

#plt.plot(np.linspace(1,5,10000), H(np.linspace(1,5,10000)))
#plt.show()