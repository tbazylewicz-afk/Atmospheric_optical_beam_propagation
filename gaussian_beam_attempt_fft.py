import numpy as np
from matplotlib import pyplot as plt

""" initialization """

# parameters
N = 512 # grid resolution (number of points)
L = 0.01 # simulation window width
lambd = 500e-9 # wavelength
dx = L / N # grid spacing
beam_waist = L / 5 # beam radius at its narrowest point
k = 2 * np.pi / lambd # wave number
z = 30 # distance at which E_out is computed

# grid definition
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Gaussian beam
E_in = np.exp(-(X**2 + Y**2) / beam_waist**2)

""" beam propagation with FFT """

# grid in Fourier space
k_x = 2 * np.pi * np.fft.fftfreq(N, d=dx)
k_y = 2 * np.pi * np.fft.fftfreq(N, d=dx)
K_X, K_Y = np.meshgrid(k_x, k_y)

# k_z (angular spectrum propagator - taylor approximation)
k_z = k - (K_X**2 + K_Y**2) / (2*k) + 0j # 0j to make the format imaginary

# transformation of E_in into E_k (E in Fourier space) with FFT
E_k = np.fft.fft2(E_in)

# propagation of E_in
propagator = np.exp(1j * k_z * z)
E_k_prop = E_k * propagator

# transformation of E_in into E_out with FFT
E_out = np.fft.ifft2(E_k_prop)

""" plot """

figure, axis = plt.subplots(1, 2, figsize=(14, 5))
# E_in at z=0
axis_0 = axis[0].imshow(np.abs(E_in)**2, extent=[x.min(), x.max(), y.min(), y.max()])
axis[0].set_title("E_in at z=0")
axis[0].set(xlabel="x")
axis[0].set(ylabel="y")
figure.colorbar(axis_0, ax=axis[0])
# E_out
axis_1 = axis[1].imshow(np.abs(E_out)**2, extent=[x.min(), x.max(), y.min(), y.max()])
axis[1].set_title(f"E_out at z={z}")
axis[1].set(xlabel="x")
axis[1].set(ylabel="y")
figure.colorbar(axis_1, ax=axis[1])
plt.show()

print(E_out)

""" comparision with analytical Gaussian beam """
# I(r, z) from https://en.wikipedia.org/wiki/Gaussian_beam

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
