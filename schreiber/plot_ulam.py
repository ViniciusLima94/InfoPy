import numpy as np 
import matplotlib.pyplot as plt

couplings = np.arange(0, 1.02, 0.02)

MI12 = np.zeros(couplings.shape[0])
MI21 = np.zeros(couplings.shape[0])
TE12 = np.zeros(couplings.shape[0])
TE21 = np.zeros(couplings.shape[0])

for i in range(1,couplings.shape[0]+1):
	data = np.load('data_maps/ulam/ulam_kernel_'+str(i)+'.npy').item()
	MI12[i-1] = data['M12']
	MI21[i-1] = data['M21']
	TE12[i-1] = data['TE12']
	TE21[i-1] = data['TE21']

jidt = np.loadtxt('data_maps/ulam/jidt_ulam.txt', delimiter=',')

#plt.plot(couplings, MI12, '--')
#plt.plot(couplings, MI21, '--')
plt.plot(jidt[:,0], jidt[:,1])
plt.plot(jidt[:,0], jidt[:,2])
plt.plot(couplings, TE12)
plt.plot(couplings, TE21)
plt.xlabel('Coupling')
plt.ylabel('TE [bits]')
plt.title('ULAM map TE')
plt.legend(['TEjidt(1->2)', 'TEjidt(2->1)', 'TE(1->2)', 'TE(2->1)'])
plt.savefig('figures/ulam_te_kernel.png', dpi=600)
plt.close()

'''
for i in range(couplings.shape[0]):
	data = np.load('data_maps/ulam/ulam_ksg_'+str(i)+'.npy').item()
	MI12[i] = data['M12']
	MI21[i] = data['M21']
	TE12[i] = data['TE12']
	TE21[i] = data['TE21']

plt.plot(couplings, MI12, '--')
plt.plot(couplings, MI21, '--')
plt.plot(couplings, TE12)
plt.plot(couplings, TE21)
plt.xlabel('Coupling')
plt.ylabel('TE [bits]')
plt.title('KSG')
plt.legend(['MI(1->2)', 'MI(2->1)', 'TE(1->2)', 'TE(2->1)'])
plt.savefig('figures/ulam_te_ksg.pdf', dpi=600)
plt.close()
'''