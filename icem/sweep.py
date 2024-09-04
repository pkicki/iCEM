import asyncio
import pickle 
import os, sys
import subprocess
import numpy as np
from itertools import product

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


ENVBIN = sys.exec_prefix
PYTHON = os.path.join(ENVBIN, "bin", "python")

N = 1
#skip_existing = False
skip_existing = True

#env_name = "humanoid_standup"
env_name = "halfcheetah_running"

#name = "icem"
#noise_betas = np.linspace(0.6, 1.0, 21).tolist()
#init_stds = np.linspace(0.2, 0.6, 5).tolist()
#sweep_values = [noise_betas, init_stds]
#sweep_names = ['noise_beta', 'init_std']

name = "fcem"
cutoff_freqs = np.linspace(6.0, 7.0, 5).tolist()
init_stds = np.linspace(0.5, 1.0, 11).tolist()
sweep_values = [cutoff_freqs, init_stds]
sweep_names = ['cutoff_freq', 'init_std']

results = {"sweep_names": sweep_names,
           "sweep_values": sweep_values,
           "number_of_rollouts": N,
           "means": {},
           "stds": {}}

@background
def run_exp(idx, values):
    filename = f"exp_results/{env_name}/{name}/{'_'.join(str(v) for v in values)}.res"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if skip_existing and os.path.exists(filename):
        return True

    #cmd = f"taskset -c {idx%5} {PYTHON} main.py "
    cmd = f"{PYTHON} main.py "
    cmd += f'./settings/{env_name}/i-cem-blitz.json '
    cmd += f'evaluation_rollouts={N} '

    for i, v in enumerate(values):
        cmd += f'controller_params.action_sampler_params.{sweep_names[i]}={v} '

    result = subprocess.check_output(cmd, shell=True, text=True)
    rewards = [float(x) for x in result.split("\n")[-2].split(" ")]
    fh = open(filename, "a+")
    fh.write("\n".join(str(x) for x in rewards) + "\n")

    #mean_std = [float(x) for x in result.split("\n")[-2].split(" ")]
    #mean, std = mean_std[0], mean_std[1]
    #results["means"][values] = mean
    #results["stds"][values] = std

#for values in product(*sweep_values):
    #cmd_ = cmd
    #for i, v in enumerate(values):
    #    cmd_ += f'controller_params.action_sampler_params.{sweep_names[i]}={v} '

    #result = subprocess.check_output(cmd_, shell=True, text=True)
    #mean_std = [float(x) for x in result.split("\n")[-2].split(" ")]
    #mean, std = mean_std[0], mean_std[1]
    #results["means"][values] = mean
    #results["stds"][values] = std

loop = asyncio.get_event_loop()

looper = asyncio.gather(*[run_exp(i, values) for i, values in enumerate(product(*sweep_values))])
                               
a = loop.run_until_complete(looper)

print(results)
#with open("icem_n20.pkl", "wb") as outfile: 
#with open("fcem_n20.pkl", "wb") as outfile: 
#    pickle.dump(results, outfile)