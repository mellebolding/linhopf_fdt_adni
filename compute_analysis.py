import os
import numpy as np
import pandas as pd
from src.data_loaders.ADNI_A import loadBurden
from src.data_loaders.load_data_records import loadProteins

DL_type = 'DL_B'
NPARCELLS = 21  # max 379 for DL_A, max 400 for DL_B
fit_sigma = True
fit_a = True

repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "FDT_DATA")
filename = f"FDT_results_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
linhopf_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: linhopf_data[k].tolist() for k in linhopf_data.files})

df = loadProteins(df, DL_type, 'Amyloid', repo_root) #'Amyloid' or 'Tau'





