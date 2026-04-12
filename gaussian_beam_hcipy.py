# propagacja z biblioteki hcipy na podstawie przykładów
import numpy as np
import matplotlib.pyplot as plt
from hcipy import *

""" initialization """

# v1 (mieszanka parametrów z książki + nasz dystans)
N = 1024 # grid resolution (number of points)
L = 0.015 # simulation window width
D = 0.01 # initial beam diameter
lambd = 500e-9 # wavelength
# dx = L / N # grid spacing
z = 24622000 # distance at which E_out is computed

# v2 (z kodu z FFT)
N = 512 # grid resolution (number of points)
L = 15 # simulation window width
D = 0.2 * L # aperture diameter
lambd = 500e-9 # wavelength
# dx = L / N # grid spacing
z = 24622000 # distance at which E_out is computed

# grid definition
pupil_grid = make_pupil_grid(N, L)

""" beam propagation """

# circular aperture - old code leftover
# E_in = make_circular_aperture(D)(pupil_grid)

# Gaussian beam
E_in = Field(np.exp(-(pupil_grid.x**2 + pupil_grid.y**2) / (D / 2)**2), pupil_grid)
# transformation of E_in to wavefront (E_in for given wavelength)
E_in_wf = Wavefront(E_in, lambd)
# propagator
propagator = FresnelPropagator(pupil_grid, z)
# propagator applied to E_in gives another wavefront
E_out_wf = propagator(E_in_wf)
# final E_out
E_out = E_out_wf.intensity # .intensity computes |E|^2

# plot
figure, axis = plt.subplots(1, 2, figsize=(14, 5))
# E_in at z=0
axis_0 = axis[0].imshow(np.abs(E_in.shaped)**2, extent=[-L/2, L/2, -L/2, L/2]) # .shaped reshapes one long vector into a matrix
axis[0].set_title("E_in at z=0")
axis[0].set(xlabel="x")
axis[0].set(ylabel="y")
figure.colorbar(axis_0, ax=axis[0])
# E_out
axis_1 = axis[1].imshow(E_out.shaped, extent=[-L/2, L/2, -L/2, L/2]) # without ()^2 as .intensity already did that
axis[1].set_title(f"E_out at z={z}")
axis[1].set(xlabel="x")
axis[1].set(ylabel="y")
figure.colorbar(axis_1, ax=axis[1])
plt.suptitle(fr"$N = {N}, L = {L}, D = {D}, \lambda = {lambd}, z = {z}$")
plt.show()

""" comparision with analytical Gaussian beam """
# I(r, z) from https://en.wikipedia.org/wiki/Gaussian_beam

# grid-based values preparation
x = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, x)

# parameters
I_0 = 1.
w_0 = D / 2
z_R = np.pi * w_0**2 / lambd
w_z = w_0 * np.sqrt(1 + (z / z_R)**2)
r = np.sqrt(X**2 + Y**2)

# analytical intensity
I_analytical = I_0 * (w_0 / w_z)**2 * np.exp(-2 * r**2 / w_z**2)

# plot comparing the cross-sections (our and analytical)
plt.plot(x, np.abs(E_out.shaped[N//2, :]), label="numerical")
plt.plot(x, I_analytical[N//2, :], label="analytical")
plt.xlabel("x")
plt.title(f"E_out cross-section comparison at y={N//2}, z={z}")
plt.legend()
plt.show()

print(np.sum(np.abs(E_in.shaped)**2) * pupil_grid.weights)
print(np.sum(E_out.shaped) * pupil_grid.weights)