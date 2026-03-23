import numpy as np
import matplotlib.pyplot as plt

from hcipy import *



pupil_grid_2 = make_pupil_grid(1024, 0.015)
aperture_2 = circular_aperture(0.01)(pupil_grid_2)

fresnel_prop = FresnelPropagator(pupil_grid_2, 2)

wf = Wavefront(aperture_2, 500e-9)
img = fresnel_prop(wf)

imshow_field(img.intensity)
plt.colorbar()
plt.show()


