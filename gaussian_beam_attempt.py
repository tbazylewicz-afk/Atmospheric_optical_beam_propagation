import numpy as np
from matplotlib import pyplot as plt

# parameters
N = 100
L = 0.1
lambd = L / 20
dx = L / N
beam_waist = L / 5
k = 2 * np.pi / lambd
z = 0.5 # distance at which E_out is computed

# grid
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Gaussian beam
E_in = np.exp(-(X**2 + Y**2) / beam_waist**2)
# plot
plt.imshow(np.abs(E_in)**2, extent=[x.min(), x.max(), y.min(), y.max()])
plt.title("E_in at z=0")
plt.colorbar()
plt.show()

# grid in Fourier space
k_x = 2 * np.pi * np.fft.fftfreq(N, d=dx)
k_y = 2 * np.pi * np.fft.fftfreq(N, d=dx)
K_X, K_Y = np.meshgrid(k_x, k_y)

# k_z (from a formula)
k_z = np.sqrt(k**2 - K_X**2 - K_Y**2 + 0j)  # 0j to make the format imaginary

# transformation of E_in into E_k (E in Fourier space) with FFT
E_k = np.fft.fft2(E_in)

# propagation of E_in
propagator = np.exp(1j * k_z * z)
E_k_prop = E_k * propagator

# transformation of E_in into E_out with FFT
E_out = np.fft.ifft2(E_k_prop)
# plot
plt.imshow(np.abs(E_out)**2, extent=[x.min(), x.max(), y.min(), y.max()])
plt.title(f"E_out at z={z}")
plt.colorbar()
plt.show()

print(E_out)