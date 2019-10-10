import numpy             as np 

def box_kernel(X, grid, r):
	N     = X.shape[0]
	Ngrid = grid.shape[0]

	P     = np.zeros(Ngrid)
	for i in range(Ngrid):
		for j in range(N):
			if np.max(  np.abs( grid[i]-X[j] ) ) / r < 1:
				P[i] += 1 / N
	return P

def boxkernel(x_i, bandwidth):
    def evaluate(x):
        """Evaluate x."""
        if  x<=0: pdf=1
        elif x>0: pdf=0
        return(pdf)
    return(evaluate)

def kde_pdf(data, kernel_func, bandwidth):
    """Generate kernel density estimator over data."""
    kernels = dict()
    n = len(data)
    for d in data:
        kernels[d] = kernel_func(d, bandwidth)
    def evaluate(x):
        """Evaluate `x` using kernels above."""
        pdfs = list()
        for d in data: pdfs.append(kernels[d](x))
        return(sum(pdfs)/n)
    return(evaluate)

