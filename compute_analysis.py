import os
import numpy as np
import pandas as pd
from src.data_loaders.load_data_records import loadProteins
from src.analysis.p_values import parcel_comparison_rsn, subject_comparison_rsn
from src.analysis.p_values import parcel_comparison, subject_comparison
import json

DL_type = 'DL_B1'
model_type = 'modelbased'  # 'modelfree' or 'modelbased'
NPARCELLS = 400 # max 379 for DL_A, max 400 for DL_B
fit_sigma = True
fit_a = True


repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "FDT_DATA")
save_path_plot = os.path.join(repo_root, "data", "RESULT_PLOTS")
if model_type == 'modelbased': filename = f"FDT_results_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
else: filename = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"
fdt_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: fdt_data[k].tolist() for k in fdt_data.files})

df = loadProteins(df, DL_type, 'Amyloid', repo_root) #'Amyloid' or 'Tau'
df = loadProteins(df, DL_type, 'Tau', repo_root) #'Amyloid' or 'Tau'

# #print(df.columns)
parcel_comparison(df, 'I_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison(df, 'I_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
parcel_comparison(df, 'X_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison(df, 'X_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
#print(np.max([np.max(arr) for arr in df['ABeta'].values if isinstance(arr, np.ndarray)]))

import json

def add_parcel_info_to_df(df, NPARCELLS, json_data_path):
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    if NPARCELLS==379: 
        parcel_names = 'Parcel_names'
        network_names = 'Networks'
    elif NPARCELLS==400:
        parcel_names = 'Parcel_names_400'
        network_names = 'Networks_400'
    parcel_names_list = [json_data[parcel_names].get(str(i+1), f"Parcel_{i+1}") for i in range(NPARCELLS)]
    
    parcel_idx_to_rsn = {}
    for rsn_name, indices in json_data[network_names].items():
        for idx in indices:
            idx0 = idx - 1  # convert to 0-based index
            if 0 <= idx0 < NPARCELLS:
                parcel_idx_to_rsn[idx0] = rsn_name

    # Create a list of network names for each parcel
    parcel_rsn_list = [
        parcel_idx_to_rsn.get(i, 'Unassigned') 
        for i in range(NPARCELLS)
    ]
    
    # Add to dataframe (same for every subject)
    df['parcel_names'] = [parcel_names_list] * len(df)
    df['parcel_RSNs'] = [parcel_rsn_list] * len(df)
    
    return df

add_parcel_info_to_df(df, NPARCELLS, os.path.join(repo_root, 'hyperparams.json'))

print(df.head())
measure = 'I_norm2'
subject_comparison_rsn(df, measure, 'Vis', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison_rsn(df, measure, 'SomMot', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison_rsn(df, measure, 'DorsAttn', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison_rsn(df, measure, 'SalVentAttn', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison_rsn(df, measure, 'Limbic', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison_rsn(df, measure, 'Cont', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
subject_comparison_rsn(df, measure, 'Def', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)

