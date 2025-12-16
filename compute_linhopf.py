"""
This script computes the LinHopf model fitting per subject

INPUTS:
  - fMRI data (through data loader). Contains: [subject_id, group, fMRI (time series), SC, f_diff]
  - Model parameters (from hyperparams.json)

OUTPUTS:
  - Compressed npz file with fitting results per subject: 
      [subject_id, group, f_diff, SC, EC, fitted sigma, fitted a, fit metrics]
"""

import os
import json
import time
import pandas as pd
import numpy as np
import src.data_loaders.ADNI_B as ADNI_B
import src.data_processing.zfilterts as zfilterts
import src.data_loaders.load_data_records as load_data_records
import src.functions_frameworks.LinHopfFit as LinHopfFit


### MAIN INPUT PARAMETERS (other parameters in json file)
DL_type = 'DL_B2' # specific data loader with groups HC, MCI(AB+), AD
filt = True # use filtered data or not
NPARCELLS = 400 # number of parcels to use
fit_sigma = True # whether to fit sigma parameter
fit_a = True # whether to fit a parameter
verbose = False # verbosity of fitting process
sigma_ini = 0.45 * np.ones(NPARCELLS) # initial sigma values
a_ini = -0.02 * np.ones(NPARCELLS) # initial a values

start_time = time.time()

# Load hyperparameters from JSON file
with open("hyperparams.json", "r") as f:
    params = json.load(f)

if DL_type == 'DL_B1':
    DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'],filt=filt)
if DL_type == 'DL_B2':
    DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB+)', 'AD'],filt=filt)
if DL_type == 'DL_B3':
    DL = ADNI_B.ADNI_B_Alt(['HC(AB-)', 'HC(AB+)', 'MCI(AB-)', 'MCI(AB+)', 'AD'],filt=filt)

# Load data for all subjects
all_data = []
for group in DL.get_groupLabels():
    print(f"Loading {group}...")
    all_data.extend(load_data_records.load_group_data(DL, group, SC=None))

# Prepare time series data: z-scoring and detrending
TSemp_zsc = zfilterts.prepare_timeseries(all_data, NPARCELLS)

# Fit LinHopf model for each subject
results = []
for idx, subj_data in enumerate(all_data):

    # fit the model
    single_subject_result = LinHopfFit.fit_linhopf(subj_data, TSemp_zsc[idx], sigma_ini, a_ini, verbose, params, NPARCELLS)

    # adjust length of f_diff and SC if running with less than max parcels
    single_subject_result.update({
        'f_diff': subj_data['f_diff'][:NPARCELLS],
        'SC': subj_data['SC'][:NPARCELLS, :NPARCELLS],
    })

    # collect results and report progress
    results.append(single_subject_result)
    if (idx + 1) % 10 == 0:
        print(f"Fitted {idx + 1}/{len(all_data)} individual subjects")

# report total computation time
end_time = time.time()
print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

#Save results to dataframe and then to compressed npz file
df = pd.DataFrame(results)
results_dict = {str(key): value for key, value in df.to_dict(orient="list").items()}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "HOPF_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"linhopf_fit_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
np.savez_compressed(f"{save_path}/{filename}", **results_dict)