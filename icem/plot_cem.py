import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def read_cem(env_name, name):
    r = []
    for f in os.listdir(f"exp_results/{env_name}/{name}"):
        if f.endswith(".res"):
            rewards = np.loadtxt(f"exp_results/{env_name}/{name}/{f}")
            r.append([*[float(x) for x in f[:-4].split("_")], np.mean(rewards)])
    r = np.array(r)
    return r
#
#results_icem = pickle.load(open('icem_n20.pkl', 'rb'))
#results_fcem = pickle.load(open('fcem_n20.pkl', 'rb'))
#
#icem = np.array([[k1, k2, v] for (k1, k2), v in results_icem['means'].items()])
#fcem = np.array([[k1, k2, v] for (k1, k2), v in results_fcem['means'].items()])
#
#plt.subplot(211)
#plt.scatter(icem[:, 0], icem[:, 1], c=icem[:, 2], label="icem")
#plt.subplot(212)
#plt.scatter(fcem[:, 0], fcem[:, 1], c=fcem[:, 2], label="fcem")
#plt.colorbar()
#plt.show()
env_name = "halfcheetah_running"

fcem = read_cem(env_name, "fcem")
icem = read_cem(env_name, "icem")
fcem_max, fcem_min, fcem_med = np.max(fcem[:, 2]), np.min(fcem[:, 2]), np.median(fcem[:, 2])
icem_max, icem_min, icem_med = np.max(icem[:, 2]), np.min(icem[:, 2]), np.median(icem[:, 2])
print("FCEM MAX:", fcem_max, "FCEM MIN:", fcem_min, "FCEM MED:", fcem_med)
print("ICEM MAX:", icem_max, "ICEM MIN:", icem_min, "ICEM MED:", icem_med)
vmin = min(fcem_min, icem_min)
vmax = max(fcem_max, icem_max)
plt.subplot(121)
plt.scatter(fcem[:, 0], fcem[:, 1], c=fcem[:, 2], label="fcem", vmin=vmin, vmax=vmax)
plt.legend()
plt.subplot(122)
plt.scatter(icem[:, 0], icem[:, 1], c=icem[:, 2], label="icem", vmin=vmin, vmax=vmax)
plt.colorbar()
plt.legend()
plt.show()

a = 0