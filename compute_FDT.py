import os
import numpy as np
import pandas as pd
from src.functions_frameworks.functions_FDT_norm2 import I_norm2, X_norm2

NPARCELLS = 21  # max 379
fit_sigma = True
fit_a = True

repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "HOPF_DATA")
filename = f"linhopf_fit_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
linhopf_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: linhopf_data[k].tolist() for k in linhopf_data.files})

df = I_norm2(df)
df = X_norm2(df)

results_dict = {str(key): value for key, value in df.to_dict(orient="list").items()}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"FDT_results_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)