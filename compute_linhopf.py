import os
import numpy as np
import src.data_loaders.ADNI_A as ADNI_A
import src.data_processing.zfilterts as zfilterts
import src.data_loaders.load_data_records as load_data_records
import src.functions_frameworks.LinHopfFit as LinHopfFit
import json
import time
import pandas as pd
from typing import Union

start_time = time.time()

### MAIN INPUT PARAMETERS (other parameters from json file)
NPARCELLS = 21 # max 379
fit_sigma = True
fit_a = True
verbose = False
sigma_ini = 0.45 * np.ones(NPARCELLS)
a_ini = -0.02 * np.ones(NPARCELLS)

with open("hyperparams.json", "r") as f:
    params = json.load(f)

DL = ADNI_A.ADNI_A(normalizeBurden=False)
all_data = []
for group in ['HC', 'MCI', 'AD']:
    print(f"Loading {group}...")
    all_data.extend(load_data_records.load_group_data(DL, group))
TSemp_zsc = zfilterts.prepare_timeseries(all_data, NPARCELLS)
group_data = load_data_records.prepare_group_data(all_data, NPARCELLS)

results = []
group_results = []
groups_fitted = set()

for idx, subj_data in enumerate(all_data):
    
    # Fit group (only once per group)
    current_group = subj_data['group']
    if current_group not in groups_fitted:
        print(f"\nFitting {current_group} group model...")

        single_group_result = LinHopfFit.fit_linhopf(group_data[current_group], None, sigma_ini, a_ini, verbose,params, NPARCELLS)
        single_group_result.update({
            'group': current_group,
            'f_diff': group_data[current_group]['f_diff'][:NPARCELLS],
        })
        group_results.append(single_group_result)
        groups_fitted.add(current_group)

    # Fit individual subject
    single_subject_result = LinHopfFit.fit_linhopf(subj_data, TSemp_zsc[idx], sigma_ini, a_ini, verbose, params, NPARCELLS)
    single_subject_result.update({
        'subject_id': subj_data['subject_id'],
        'group': subj_data['group'],
        'f_diff': subj_data['f_diff'][:NPARCELLS],
    })
    results.append(single_subject_result)
    if (idx + 1) % 10 == 0:
        print(f"Fitted {idx + 1}/{len(all_data)} individual subjects")

end_time = time.time()
print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

#Save results to npz
df = pd.DataFrame(results + group_results)
results_dict = {str(key): value for key, value in df.to_dict(orient="list").items()}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "HOPF_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"linhopf_fit_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)