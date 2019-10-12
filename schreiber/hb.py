import sys
sys.path.append('../infoPy')

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from kernel import *
from tools import *

# Read in the data
rawData = pd.read_csv('data.dat', header = None, delimiter = ',')
x = rawData[0].values # Extracts what Matlab does with 2350:3550 argument there.
# Chest vol is second column
y = rawData[1].values
rs = np.array([0.01, 0.016, 0.023, 0.032, 0.047, 0.064, 0.09, 0.12, 0.18, 0.26, 0.36, 0.50, 0.65, 1.0])
MI12 = np.zeros(len(rs))
MI21 = np.zeros(len(rs))
TE12 = np.zeros(len(rs))
TE21 = np.zeros(len(rs))
for i in range( len(rs) ):
	print('r = ' + str(rs[i]))
	#MI12[i] = KernelEstimatorMI(x, y, bw = rs[i], kernel = 'tophat', delay = 1, norm=True) 
	#MI21[i] = KernelEstimatorMI(y, x, bw = rs[i], kernel = 'tophat', delay = 1, norm=True) 
	TE12[i] = KernelEstimatorTE(x, y, bw = rs[i], kernel = 'tophat', norm=True)
	TE21[i] = KernelEstimatorTE(y, x, bw = rs[i], kernel = 'tophat', norm=True)
#plt.semilogx(rs, MI12, '--')
#plt.semilogx(rs, MI21, '--')
plt.semilogx(rs, TE12)
plt.semilogx(rs, TE21, '-.')
plt.legend(['TE(heart->breath)', 'TE(breath->heart)'])
plt.xlim([0.01, 1])
plt.ylim([0, 5])
plt.ylabel('TE')
plt.savefig('figures/hb.pdf', dpi = 600)
plt.close()

