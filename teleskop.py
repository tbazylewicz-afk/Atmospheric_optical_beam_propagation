import numpy as np
from matplotlib import pyplot as plt

""" GRID + PHYSICAL PARAMETERS """

# Grid parameters
N = 512          # grid resolution
L = 15            # simulation window width
dx = L / N           # grid sapcing

# grid definition
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Wavelength
lambd = 500e-9       # 500 nm (green light)

""" INPUT FIELD """

# We assume the incoming field is Gaussian
# after atmosphere+propagation

beam_waist = L/5

E_in = np.exp(-(X**2 + Y**2) / beam_waist**2)



""" TELESCOPE MODEL """

# Telescope parameters
D = 1.0   # diameter of telescope aperture in meters

# Aperture (pupil function)
# Telescope mirror = circular mask

R = np.sqrt(X**2 + Y**2) #conversion from cartesian coords to radial coords

aperture = np.zeros_like(R) #array for values representing light or lack of it
aperture[R <= D/2] = 1 #this makes aperture circular, light is only in this radius D/2

# Field after telescope pupil
E_pupil = E_in * aperture #this 'clips' the field so it's visible only in the limits of aperutre


# Focusing (Fourier transform)
# A lens performs a Fourier transform

E_focal = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_pupil))) #how lens changes light distribution to
#an image in focal plane
#first shift is to place (0,0) point in the center instead of in the corner
#second is fourier transform by the lense
#third shift back
#the image of a point source is the fourier transform of the aperture

# Intensity in focal plane (PSF)
#point spread function: image produced by optical system of a single point source
#PSF width ~ lambda/D
I_focal = np.abs(E_focal)**2 #detectors measure intensity not E field and I=|E|^2

# Normalize for visualization
I_focal = I_focal / np.max(I_focal)
#rescaling so peak=1

""" OPTICAL FIBER COUPLING """

# Fiber mode (Gaussian profile)
#optical fiber mode: in a fiber light forms stable electromagnetic pattern (mode)
#in a single-mode fiber he only mode that propagates is fundamental
#single mode fiber removes turbulence distortions, aberrations and gives stable phase reference
#in astronomy enables precision measurements
#fundamental mode: maximum intensity in the center, smooth decay outward
w_fiber = 0.2
#w_fiber controls how wide the light distribution is inside the fiber
#w_fiber should be around PSF width

fiber_mode = np.exp(-(X**2 + Y**2) / w_fiber**2)
#models fundamental mode of a single-mode fiber which is ~gaussian
#exp creates max at the center and smooth decay outward, E field distribution the fiber accepts

# Normalize fiber mode
fiber_mode = fiber_mode / np.sqrt(np.sum(np.abs(fiber_mode)**2))
#normalization because coupling efficiency depends on relative shape not absolute power
#coupling efficiency: measures how much optical power from one field successfully transfers into another mode


# Normalize focal field
E_focal_norm = E_focal / np.sqrt(np.sum(np.abs(E_focal)**2))
#normalizing E focal field so it's comparable with fiber_mode


# Overlap integral (coupling efficiency)
overlap = np.sum(E_focal_norm * np.conj(fiber_mode))
#measures how well incoming field matches fiber mode in amplitude and phase



eta = np.abs(overlap)**2
#fraction of optical power that enters the fiber

#this part determines how much of incoming light is actually going into the fiber
#size matching: w_fiber
#shape matching: gaussian vs psf
#phase matching: abserrations reduce coupling


print("Fiber coupling efficiency =", eta)

""" PLOTS """

fig, axis = plt.subplots(1, 4, figsize=(20, 5))

# Input field
axis[0].imshow(np.abs(E_in)**2, extent=[x.min(), x.max(), y.min(), y.max()])
axis[0].set_title("Incoming field (intensity)")
axis[0].set(xlabel="x", ylabel="y")
#light before it enters the telescope (plane wave-star, distorted wavefromt-turbulence)

# Aperture
axis[1].imshow(aperture, extent=[x.min(), x.max(), y.min(), y.max()])
axis[1].set_title("Telescope aperture")
axis[1].set(xlabel="x", ylabel="y")
#this shows the pupil, where the light is allowed to come through
#shapes the final image (PSF, aperture determines diffraction)

# Field at pupil
axis[2].imshow(np.abs(E_pupil)**2, extent=[x.min(), x.max(), y.min(), y.max()])
axis[2].set_title("Field at pupil")
axis[2].set(xlabel="x", ylabel="y")
#light in the telescope after aperture is applied
#incoming wave is clipped, possibly carrying phase distortions

# Focal plane (PSF)
axis[3].imshow(I_focal, extent=[x.min(), x.max(), y.min(), y.max()])
axis[3].set_title("Focal plane (PSF)")
axis[3].set(xlabel="x", ylabel="y")
#image formed by the telescope, psf is the fourier transform of the pupil field

plt.show()

""" CROSS-SECTION """
#this takes 2D optical field and makes it 1D slice to show
#how light profile evolves along one line


# finding middle row of the 2D grid (y = 0 → middle row)
center_index = N // 2

# Cross-sections
input_cross = np.abs(E_in[center_index, :])**2
#intensity profile of the incoming beam along x; shows beam width, uniformity or distortions
pupil_cross = np.abs(E_pupil[center_index, :])**2
#same slice but after aperture
focal_cross = I_focal[center_index, :]
#cross-section of the PSF

# Plot cross-sections
plt.figure(figsize=(8, 5))

plt.plot(x, input_cross, label="Input beam")
plt.plot(x, pupil_cross, label="After aperture")
plt.plot(x, focal_cross, label="Focal plane (PSF)")

plt.xlabel("x")
plt.ylabel("Intensity")
plt.title("Cross-section at y = 0")
plt.legend()


plt.show()
