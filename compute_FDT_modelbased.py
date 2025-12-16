"""
This script computes FDT metrics using the analytical approach using the data from linhopf model

INPUTS:
  - EC (model), a values (model), sigma values (model), f_diff values (data)
OUTPUTS:
  - Compressed npz file with FDT results per subject: I_norm2, X_norm2
"""

import os
import numpy as np
import pandas as pd
from src.functions_frameworks.functions_FDT_analytical import I_norm2, X_norm2

### MAIN INPUT PARAMETERS (other parameters in json file)
DL_type = 'DL_B2' # specific data loader with groups HC, MCI(AB-), MCI(AB+), AD
NPARCELLS = 400 # number of parcels to use
fit_sigma = True # whether sigma values were fitted
fit_a = True # whether a values were fitted

# Load fitted linhopf data
repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "HOPF_DATA")
filename = f"linhopf_fit_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}_filt.npz"
linhopf_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: linhopf_data[k].tolist() for k in linhopf_data.files})

# Compute FDT 
df_fdt = I_norm2(df)
df_fdt = X_norm2(df_fdt)

# Save FDT results
results_dict = {str(key): value for key, value in df_fdt.to_dict(orient="list").items()}
repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"FDT_results_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}_filt.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)