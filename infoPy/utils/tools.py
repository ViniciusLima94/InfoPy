import numpy as np

def normalize_data(X, d):
	r'''
		Normalize each column of the data matrix X
		Inputs:
		X : Data matrix must have size Nobservations x d.
		d : Number of dimensions (or variables)
	'''

	X_norm = np.zeros_like(X)

	if d > 1:

		for i in range(d):
			X_norm[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

		return X_norm

	elif d == 1:
		return ( X - X.mean() ) / X.std()