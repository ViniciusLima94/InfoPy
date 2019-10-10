import numpy             as np 
import matplotlib.pyplot as plt
from   sklearn.neighbors import KernelDensity

def heaviside(x):
	if x <= 0:
		return 1
	else:
		return 0

def pxy(x, y, r):

	X = np.vstack([x,y]).T

	L = len(X)

	xmin, xmax = x.min(), x.max()
	ymin, ymax = y.min(), y.max()

	N = 100

	grid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, N),
                                               np.linspace(ymin, ymax, N)))).T

	P = np.zeros(N**2)

	for i in range(N**2):
		for j in range(N):
			aux  =  np.max( np.abs( grid[i,:]-X[j,:] ) ) - r
			P[i] += (1/L)*heaviside(aux)

def pxy(x, y, z, r):

	X = np.vstack([x,y,z]).T

	L = len(X)

	xmin, xmax = x.min(), x.max()
	ymin, ymax = y.min(), y.max()
	zmin, zmax = y.min(), y.max()

	N = 20

	grid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, N),
                                               np.linspace(ymin, ymax, N),
                                               np.linspace(zmin, zmax, N)))).T

	Pxyz = np.zeros(N**3)

	P = np.zeros(N**2)

	for i in range(N**2):
		for j in range(N):
			aux  =  np.max( np.abs( grid[i,:]-X[j,:] ) ) - r
			P[i] += (1/L)*heaviside(aux)

	for i in range(N**3):
		for j in range(N):
			aux  =  np.max( np.abs( grid[i,:]-X[j,:] ) ) - r
			Pxyz[i] += (1/L)*heaviside(aux)



kde = KernelDensity(bandwidth=.3, kernel='tophat', metric='chebyshev')
log_dens2 = kde.fit(X).score_samples(grid)
dens2 = np.exp(log_dens2).reshape((N, N))