import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

couplings = np.arange(0, 0.052, 0.002)

TEmean = np.zeros(couplings.shape[0])
TEstd  = np.zeros(couplings.shape[0])

for i in range(couplings.shape[0]):
	data  = np.load('data_maps/tent/tent_'+str(i)+'.npy').item()
	TEmean[i] = data['TEmean']
	TEstd[i]  = data['TEstd']

plt.figure()
plt.errorbar(couplings, TEmean, TEstd)
# Data generated with JIDT toolbox for comparation
jidt = pd.read_csv('data_maps/tent/tentMap_jidt.dat', delimiter=',', header=None, names=['c', 'te', 'ste'])
plt.errorbar(jidt['c'], jidt['te'], jidt['ste'])
plt.title('Transfer Entropy for the Tent Map')
plt.ylabel('TE [bits]')
plt.xlabel(r'$\epsilon$')
plt.legend(['InfoPy TE', 'JIDT TE'])
plt.savefig('Figures/TE_TENTMAP.pdf', dpi = 600)