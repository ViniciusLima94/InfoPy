import numpy             as np 
import matplotlib.pyplot as plt 
import pandas            as pd
from   infoPy.kernel     import *

def box_kernel(X, grid, r):
	N     = X.shape[0]
	Ngrid = grid.shape[0]

	P     = np.zeros(Ngrid)
	for i in range(Ngrid):
		for j in range(N):
			if np.sqrt(  np.sum( (grid[i]-X[j])**2 ) ) / r < 1:
				P[i] += 1 / N
	return P

rawData = pd.read_csv('schreiber/data.txt', header = None, delimiter = ',')
x = rawData[0].values
y = rawData[1].values

x = (x - np.mean(x))/np.std(x)
y = (y - np.mean(y))/np.std(y)

z = x[0:-1]
x = x[1:]
y = y[0:-1]

X = np.vstack([x,y,z]).T

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()

grid  = np.vstack([x,y,z]).T

r=0.3

Px = box_kernel(x, np.linspace(xmin, xmax, 1000), r)
Py = box_kernel(y, np.linspace(ymin, ymax, 1000), r)

px = kde_sklearn(x, np.linspace(xmin, xmax, 1000), bandwidth=r)
py = kde_sklearn(y, np.linspace(ymin, ymax, 1000), bandwidth=r)

Pxy = box_kernel(X[:,0:2], X[:,0:2], r)

pxy = kde_estimator(x, y, x, y, bandwidth=r)

Pxyz = box_kernel(X, X, r)
pxyz = kde_estimator2(x, y, z, x, y, z, bandwidth=r)























L = len(x)

# P(x), P(y)

Px = np.zeros(L)
Py = np.zeros(L)
for i in range(L):
	for j in range(L):
		if np.abs( x[i] - x[j] ) / r < 1:
			Px[i] += 1 / (L)
		if np.abs( y[i] - y[j] ) / r < 1:
			Py[i] += 1 / (L)

Cxb, _ = np.histogram(x, bins=p1)
Cyb, _ = np.histogram(y, bins=p2)

#P(x,y)
Pxy = np.zeros(L+1)
for i in range(L+1):
	for j in range(L+1):
		if np.sqrt( ( grid[i,0] - x[j] )**2 + ( grid[i,1] - y[j] )**2 ) / r < 1:
			Pxy[i] += 1 / (L+1)


z = x[0:-1]
x = x[1:]
y = y[0:-1]

X = np.vstack([x, y, z]).T





'''
#P(x,y,z)
Pxyz = np.zeros(L)
N = 0
for i in range(L):
	for j in range(L):
		if np.sqrt( ( grid[i,0] - x[j] )**2 + ( grid[i,1] - y[j] )**2  + ( grid[i,2] - z[j] )**2 ) / r < 1:
			Pxyz[i] += 1 / L
			N += 1



P = np.zeros_like(grid[:,0])

P2 = kde_estimator(x, y, grid[:,0], grid[:,1], kernel='tophat', bandwidth=r)
P3 = kde_estimator2(x, y, z, grid[:,0], grid[:,1], grid[:,2], bandwidth=r, kernel='tophat')