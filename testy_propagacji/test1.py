import numpy as np
import matplotlib.pyplot as plt
import imageio


time = 100
dt = 0.1
n = 100
dx = 1/n

A = np.zeros((n,n))  
E = np.zeros((n,n))    
lap = np.zeros((n-2,n))    

A_old = np.zeros((n,n))  
E_old = np.zeros((n,n))    

E_old[1,20] = 3.

for t in range(0, time):
    lap = A[2:n,:] - 2 * A[1:n-1,:] + A[0:n-2,:]

    E[1:n-1] = E_old[1:n-1] + dt * lap / dx**2
    A = A_old + E*dt

    E_old = E
    A_old = A

    #plt.imshow(np.flip(E.T, axis=0), interpolation='nearest', origin='lower', aspect='auto')
    #plt.colorbar()
    #plt.savefig('animacja/' + str(t))
        





plt.imshow(np.flip(E.T, axis=0), interpolation='nearest', origin='lower', aspect='auto')
plt.colorbar()
plt.show()

plt.scatter(np.linspace(0,1,n),E[:,20], s=5)
plt.show()


plt.imshow(np.flip(A.T, axis=0), interpolation='nearest', origin='lower', aspect='auto')
plt.colorbar()
plt.show()
 
plt.scatter(np.linspace(0,1,n),A[:,20], s=5)
plt.show()






















def animate(total_time, step):
    with imageio.get_writer('animacja2.gif', mode='I', fps=5) as writer:
        for i in range(0, total_time, step):
            image = imageio.imread(f'animacja/img{i:06d}.png')
            writer.append_data(image)


#animate(10000,1)