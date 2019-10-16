'''
	Python module to compute information theoretical quantities
'''

import numpy             as np
import os
from   utils.tools       import *
from   jpype             import *

def KernelEstimatorEntropy(X, d, bw = 0.3, norm = True):
	r'''
		> Description:
		Computes the d-variables entropy of X: H(X_1, X_2, ..., X_d) = E[log2(p(X_1, X_2, ..., X_d))].
		> Inputs:
		X : Data matrix must have size Nobservations x d.
		d : Number of dimensions (or variables).
		bw : bandwidth of the kernel estimator.
		> Outputs:
		H_X : The joint entropy of X (if d = 1 it is the single-variable entropy).
	'''

	# Normalizing data
	if norm == True:
		X = normalize_data(X, d)

	Nobs = X.shape[0]

	p_X  = KernelDensityEstimator(X, d, bw)

	H_X  = np.zeros_like(p_X)

	for i in range(Nobs):
		if p_X[i] > 0:
			H_X[i] = -np.log2(p_X[i])

	return H_X.mean()

def KernelEstimatorMI(X, d, bw = 0.3, norm = True):
	r'''
		> Description: 
		Computes the mutual information between d signals I(X_1, X_2, ..., X_d) = H(X_1)+H(X_2)+...+H(X_d)-H(X_1, X_2, ...X_d).
		For more information see Equation. 3.14 in "An introduction to transfer entropy: Information flow in complex systems".
		> Inputs:
		X : Data matrix must have size Nobservations x d.
		d : Number of dimensions (or variables)
		bw : bandwidth of the kernel estimator.
		Outputs:
		I : Returns the mutual information between x and y.
	'''

	# Compute all single variable entropies
	H_uni = np.zeros_like(X)

	for i in range(d):
		H_uni[:,i] = KernelEstimatorEntropy(X[:,i], 1, bw = bw, norm = norm)

	# Compute the multivariable entropy
	H_multi = KernelEstimatorEntropy(X, d, bw = bw, norm = norm)

	return np.sum( H_uni.mean(axis=0) ) - H_multi.mean()

def KernelEstimatorDelayedMI(X, Y, bw = 0.3, delay = 0, norm = True):
	r'''
		> Description: 
		Computes the delayed mutual information between 2 signals.
		> Inputs:
		X, Y : Data signals.
		delay : applyed delay
		bw : bandwidth of the kernel estimator.
		Outputs:
		I : Returns the mutual information between x and y.
	'''

	# Applying delays
	x_t   = Y[delay:].copy()
	y_tm1 = X[0:-delay].copy()

	return KernelEstimatorMI(np.vstack([x_t, y_tm1]).T, 2, bw = bw, norm = norm)

def KernelEstimatorConditionalMI(X, Y, Z, bw = 0.3, norm = True):
	r'''
		> Description: 
		Computes the conditional mutual information I(X,Y|Z) = H(X,Z)+H(Y,Z)-H(X,Y,Z)-H(Z)
		For more information see Equation. 3.67 in "An introduction to transfer entropy: Information flow in complex systems".
		> Inputs:
		x, y, z : Array signals.
		d : Number of dimensions (or variables)
		bw : bandwidth of the kernel estimator.
		Outputs:
		I : Returns the mutual information between x and y.
	'''

	H_XZ  = KernelEstimatorEntropy(np.vstack([X,Z]).T, 2, bw = bw, norm = norm)
	H_YZ  = KernelEstimatorEntropy(np.vstack([Y,Z]).T, 2, bw = bw, norm = norm)
	H_XYZ = KernelEstimatorEntropy(np.vstack([X,Y,Z]).T, 3, bw = bw, norm = norm)
	H_Z   = KernelEstimatorEntropy(Z, 1, bw = bw, norm = norm)

	return H_XZ.mean() + H_YZ.mean() - H_XYZ.mean() - H_Z.mean()

def KernelEstimatorTE(X, Y, bw = 0.3, delay = 1, norm=True):
	r'''
		> Description: Computes the transfer entropy from the signal X to Y.
		TE(X->Y) = I(X_t:Y_t-1|X_t-1) = H(X_t,X_t-1)+H(Y_t-1,X_t-1)-H(X_t,Y_t-1,X_t-1)-H(X_t-1) 
		For more information see Equation. 4.31 and 4.4 in "An introduction to transfer entropy: Information flow in complex systems".
		> Inputs:
		X, Y: Input signals
		bw: bandwidth of the kernel estimator.
		kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
		delay: Delay applied between x and y, for the delayed transfer entropy
		norm: Sets whether the data will be normalized or not.
		> Outputs:
		TE: Returns the transfer entropy from X to Y.
	'''

	if delay < 1:
		print('Delay must be at leat one for TE computation, setting delay to one')
		delay = 1

	# Applying delays
	x_t   = Y[delay:].copy()
	y_tm1 = X[0:-delay].copy()
	x_tm1 = Y[0:-delay].copy()

	# New length
	N = len(x_t)

	TE = KernelEstimatorConditionalMI(x_t, y_tm1, x_tm1, bw = bw, norm = norm)

	return TE

##################################################################################################
# AUXILIARY FUNCTIONS                                                                            #
##################################################################################################
'''
def KernelDensityEstimator(X, d, bandwidth):
	r'''
		#Computes the KDE for uni and multivariable, i'm using the algorithm by Lizier which uses
		#a java backend
		#Inputs:
		#X : Data matrix, must be Nobservations x Ndimensions
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
'''

def KernelDensityEstimator(X, d, bandwidth):
	from sklearn.neighbors import KernelDensity

	kde = KernelDensity(bandwidth=bandwidth, kernel='tophat')

	if d == 1:
		kde.fit(X[:, np.newaxis])
		p = kde.score_samples(X[:, np.newaxis])

	else:
		kde.fit(X)
		p = kde.score_samples(X)
	
	return np.exp(p)


'''
def KernelDensityEstimator(X, d, bandwidth):

	from scipy.stats import gaussian_kde 

	kde = gaussian_kde(X, bandwidth)

	return kde(X)
'''

'''
def KernelDensityUniVariate(X, kernel, bandwidth):

	from statsmodels.nonparametric.kde import KDEUnivariate

	Nobs = len(X)

	kde = KDEUnivariate(X)
	kde.fit(kernel=kernel, bw=bandwidth, fft=True)

	p = [kde.evaluate(X[i])[0] for i in range(Nobs)]

	return p
'''


