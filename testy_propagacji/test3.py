#próba propagacji z biblioteką aotools
import numpy
from aotools import opticalpropagation
import matplotlib.pyplot as plt
from scipy import stats

wavelength = 500e-9

propagation_distance = 1000000.


pxl_scale = 20 / 128





size = 100
srednia = size / 2
gauss = numpy.zeros((size,size, 2))
gauss2 = numpy.zeros((size,size))
def gaussian(x):
	return numpy.exp((-(x[0]-srednia)**2 - (x[1]-srednia)**2) / 100)


for i in range(size):
	for j in range(size):
		gauss[i][j] = [i, j]
		gauss2[i][j] = gaussian(gauss[i][j])


wavefront = numpy.exp(j*gauss2)
#gauss2 = gaussian(gauss)

plt.imshow(wavefront)
plt.colorbar()
plt.show()




ttt = opticalpropagation.angularSpectrum(wavefront, wavelength, pxl_scale, pxl_scale, propagation_distance)

after = numpy.abs(ttt)**2

plt.imshow(after)
plt.colorbar()
plt.show()

