import numpy as np
import matplotlib.pyplot as plt

data = np.load('fp_data.npy')
fig, ax = plt.subplots()
ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], linestyle='none', label='Free Projection')
#dmrg =  -2.1130823802630223 Holstein
ed = -2.48479635 #SSH
ax.hlines(ed, data[0, 0], data[-1, 0], label='ED')
#ax.set_ylim(-2.5, -2.4)
ax.legend()
ax.set_xlabel('tau')
ax.set_ylabel('E')

plt.savefig("walkers1000_blocks1000_dt5e-5_stab20.png")
plt.show()
