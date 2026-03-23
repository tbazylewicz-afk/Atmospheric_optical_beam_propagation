import numpy as np
from matplotlib import pyplot as plt

""" initialization """

# parameters
N = 100 # grid resolution (number of points)
L = 0.1 # simulation window width
lambd = L / 20
dx = L / N
beam_waist = L / 5 # beam radius at its narrowest point
k = 2 * np.pi / lambd
z = .3 # distance at which E_out is computed

# grid definition
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Gaussian beam
E_in = np.exp(-(X**2 + Y**2) / beam_waist**2)
# plot of E_in
plt.imshow(np.abs(E_in)**2, extent=[x.min(), x.max(), y.min(), y.max()])
plt.title("E_in at z=0")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()

""" beam propagation with FFT """

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

# plot of E_out
plt.imshow(np.abs(E_out)**2, extent=[x.min(), x.max(), y.min(), y.max()])
plt.title(f"E_out at z={z}")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()

# cross-section plot
plt.plot(np.abs(E_out[N//2, :])**2)
plt.title(f"E_out cross-section at y={N//2}, z={z}")
plt.xlabel("x")
plt.show()

print(E_out)

""" comparision with analytical Gaussian beam """

# parameters
I_0 = 1.
w_0 = beam_waist # the same parameter, but different name

z_R = np.pi * w_0**2 / lambd
w_z = w_0 * np.sqrt(1 + (z / z_R)**2)
r = np.sqrt(X**2 + Y**2)

# analytical intensity
I_analytical = I_0 * (w_0 / w_z)**2 * np.exp(-2 * r**2 / w_z**2)

# plot comparing the cross-sections (our and analytical)
plt.plot(x, np.abs(E_out[N//2, :])**2, label="numerical")
plt.plot(x, I_analytical[N//2, :], label="analytical")
plt.xlabel("x")
plt.title(f"E_out cross-section comparison at y={N//2}, z={z}")
plt.legend()
plt.show()
