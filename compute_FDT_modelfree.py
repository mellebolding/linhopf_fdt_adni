"""
This script computes FDT metrics using a model-free approach

INPUTS:
  - fMRI data (through data loader). Contains: [subject_id, group, fMRI (time series), SC, f_diff]
OUTPUTS:
  - Compressed npz file with FDT results per subject per node:
      [subject_id, group, I_norm2, X_norm2, sigma]
"""

import os
import numpy as np
from src.data_loaders import ADNI_B, load_data_records
from src.analysis.check_gaussian_noise import plot_eta_histogram, plot_eta_qq
from src.functions_frameworks.functions_FDT_modelfree import _splitSignal, _computeDistanceFromEquilibrium, _analysisFdt1
from src.data_processing.zfilterts import prepare_timeseries

### MAIN INPUT PARAMETERS (other parameters in json file)
DL_type = 'DL_B2' # specific data loader with groups HC, MCI(AB-), MCI(AB+), AD
NPARCELLS = 400 # number of parcels to use
filt = True # use filtered data or not
TR = 3 # repetition time of time series
dt = TR / 1000.

if DL_type == 'DL_B1': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'],filt=filt)
    NPARCELLS = 400  
if DL_type == 'DL_B2': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD'],filt=filt)
    NPARCELLS = 400

# Load data for all subjects
all_data = []
for group in DL.get_groupLabels():
    all_data.extend(load_data_records.load_group_data(DL, group, DL_type))

# Prepare time series data: z-scoring and detrending
TSemp_zsc = prepare_timeseries(all_data, NPARCELLS)

# Compute FDT metrics for each subject
results = []
eta_all = np.zeros((len(all_data), NPARCELLS))
for i in range(len(all_data)):
    print("subject ", i, " of ", len(all_data))
    all_data[i]['MRI'][:,:NPARCELLS] = TSemp_zsc[i,:,:].T

    # Split signal into components
    x, dxdt, Fx, eta = _splitSignal(all_data[i]['MRI'][:,:NPARCELLS])

    # Store eta for Gaussianity check
    eta_all[i, :] = np.mean(eta, axis=0)
    sigma = np.std(eta)

    # Compute I and X model-free
    I_norm_inf, X_norm_inf = _analysisFdt1(x, eta, sigma, dt)

    # compute average distance from eq. (take average over all I values for one subject/parcel)
    intI = _computeDistanceFromEquilibrium(I_norm_inf)
    intX = _computeDistanceFromEquilibrium(X_norm_inf)

    results.append({
        'subject_id': all_data[i]['subject_id'],
        'group': all_data[i]['group'],
        'sigma': sigma,
        'I_norm2': np.abs(intI),
        'X_norm2': np.abs(intX),
    })

# check Gaussianity of noise term eta(t)
eta_avg = np.mean(eta_all, axis=0)  # Average across subjects to get 400 values
plot_eta_qq(eta_avg)
plot_eta_histogram(eta_all)

# Save results to compressed npz file
results_dict = {key: np.array([d[key] for d in results])
        for key in ['subject_id', 'group', 'I_norm2', 'X_norm2', 'sigma']}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)