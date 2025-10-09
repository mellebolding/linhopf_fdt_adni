import os
import numpy as np
from scipy import signal
from src.data_loaders import ADNI_A, ADNI_B, load_data_records
from src.functions_frameworks.functions_FDT_modelfree import _splitSignal, _analysisFdt2, _computeDistanceFromEquilibrium

DL_type = 'DL_A'
TR = 3

if DL_type == 'DL_A': 
    DL = ADNI_A.ADNI_A(normalizeBurden=False)
    NPARCELLS = 379

if DL_type == 'DL_B': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'])
    NPARCELLS = 400  

all_data = []
ts = np.array([])
for group in DL.get_groupLabels():
    all_data.extend(load_data_records.load_group_data(DL, group))

results = []
for i in range(len(all_data)):
    x, dxdt, Fx, eta = _splitSignal(all_data[i]['MRI'][:,:NPARCELLS])
    dt = TR / 1000.
    sigma = np.std(eta)
    C, R, I = _analysisFdt2(x, eta, sigma, dt)
    intI = _computeDistanceFromEquilibrium(I)
    results.append({
        'subject_id': all_data[i]['subject_id'],
        'group': all_data[i]['group'],
        'sigma': sigma,
        'I_norm2': intI,
    })

results_dict = {key: np.array([d[key] for d in results])
        for key in ['subject_id', 'group', 'I_norm2', 'sigma']}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)









    

    # what we want: I and X norm 2. 
    # not sure how to go from what we find here to that, as norm 2 seems to involve noise.
    # after implementation, it is important that this becomes clear and is consistent with the model-based version.