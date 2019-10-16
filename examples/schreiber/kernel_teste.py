import numpy             as np
import matplotlib.pyplot as plt
import os 
from   jpype             import *

class KernelDensityEstimator:

	def __init__(self, bandwidth):
		self.bandwidth = bandwidth

	def setObservations(self, data):
		self.observations = data
		self.Nobs         = data.shape[0]
		self.kernels      = dict()
		for i in range(self.Nobs):
			self.kernels[self.observations[i]] = self.box_kernel(self.observations[i])

	def box_kernel(self, x_i):
		lowerb = (x_i - self.bandwidth)
		upperb = (x_i + self.bandwidth)
		def evaluate(x):
			if   x <= lowerb: return 0.0
			elif x >  upperb: return 0.0
			else:             return 1.0/(2*self.bandwidth)
		return evaluate

	def score(self, x):
		pdfs = []
		for i in range(self.Nobs):
			pdfs.append( self.kernels[self.observations[i]](x) )
		return np.sum(pdfs) / self.Nobs

def KernelDensityEstimator(X, d, bandwidth):
	r'''
		Computes the KDE for uni and multivariable, i'm using the algorithm by Lizier which uses
		a java backend
		Inputs:
		X : Data matrix, must be Nobservations x Ndimensions
	'''

	jarLocation = os.path.join('infodynamics.jar')
	if isJVMStarted() == False:
		startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

	if d == 1:
		kernel = JPackage('infodynamics.measures.continuous.kernel').KernelEstimatorUniVariate
		kernel = kernel()
		kernel.setNormalise(False)
		kernel.initialise(bandwidth)
		kernel.setObservations(X)
		p = np.array( [kernel.getProbability(obs) for obs in X] )

	else:
		kernel = JPackage('infodynamics.measures.continuous.kernel').KernelEstimatorMultiVariate
		kernel = kernel()
		kernel.setNormalise(False)
		kernel.initialise(d, bandwidth)
		kernel.setObservations(X)
		p = np.array( [kernel.getProbability(obs) for obs in X] )

	return p


def KernelDensityEstimator2(X, d, bandwidth):
	from sklearn.neighbors import KernelDensity

	kde = KernelDensity(bandwidth=bandwidth, kernel='tophat', breadth_first=False)

	if d == 1:
		kde.fit(X[:, np.newaxis])
		p = kde.score_samples(X[:, np.newaxis])

	else:
		kde.fit(X)
		p = kde.score_samples(X)
	
	return np.exp(p)

data = np.loadtxt('data_maps/ulam/ulam_1.dat', delimiter=',')
x = data[:,0]
y = data[:,1]

delay = 1
bw    = 0.3

x_t   = y[delay:].copy()
y_tm1 = x[0:-delay].copy()
x_tm1 = y[0:-delay].copy()

p_xtm1 = KernelDensityEstimator(x_tm1, 1, bw)
p_xt_xtm1 = KernelDensityEstimator(np.vstack([x_t, x_tm1]).T, 2, bw)
p_ytm1_xtm1 = KernelDensityEstimator(np.vstack([y_tm1, x_tm1]).T, 2, bw)
p_xt_ytm1_xtm1 = KernelDensityEstimator(np.vstack([x_t, y_tm1, x_tm1]).T, 3, bw)


x = (x-x.mean())/x.std()
y = (y-y.mean())/y.std()

x_t   = y[delay:].copy()
y_tm1 = x[0:-delay].copy()
x_tm1 = y[0:-delay].copy()

p_xtm12 = KernelDensityEstimator2(x_tm1, 1, bw)
p_xt_xtm12 = KernelDensityEstimator2(np.vstack([x_t, x_tm1]).T, 2, bw)
p_ytm1_xtm12 = KernelDensityEstimator2(np.vstack([y_tm1, x_tm1]).T, 2, bw)
p_xt_ytm1_xtm12 = KernelDensityEstimator2(np.vstack([x_t, y_tm1, x_tm1]).T, 3, bw)


kde    = gaussian_kde(x_tm1, 0.3) 
p_xtm1 = kde(x_tm1)

kde       = gaussian_kde(np.vstack([x_t, x_tm1]), 0.3)
p_xt_xtm1 = kde(np.vstack([x_t, x_tm1]))

kde       = gaussian_kde(np.vstack([y_tm1, x_tm1]), 0.3)
p_ytm1_xtm1 = kde(np.vstack([y_tm1, x_tm1]))

kde       = gaussian_kde(np.vstack([x_t, y_tm1, x_tm1]), 0.3)
p_xt_ytm1_xtm1 = kde(np.vstack([x_t, y_tm1, x_tm1]))

H_xtm1 = 0
H_xt_xtm1 = 0
H_ytm1_xtm1 = 0
H_xt_ytm1_xtm1 = 0

for i in range(9999):
	if p_xtm1[i] > 0:
		H_xtm1 -= np.log2(p_xtm1[i])

	if p_xt_xtm1[i] > 0:
		H_xt_xtm1 -= np.log2(p_xt_xtm1[i])

	if p_ytm1_xtm1[i] > 0:
		H_ytm1_xtm1 -= np.log2(p_ytm1_xtm1[i])

	if p_xt_ytm1_xtm1[i] > 0:
		H_xt_ytm1_xtm1 -= np.log2(p_xt_ytm1_xtm1[i])
