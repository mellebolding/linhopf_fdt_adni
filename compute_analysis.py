import os
import numpy as np
import pandas as pd
from src.data_loaders.ADNI_A import loadBurden
from src.data_loaders.load_data_records import loadProteins

DL_type = 'DL_B'
model_type = 'modelfree'  # 'modelfree' or 'modelbased'
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

print(df.columns)

# first the basis plots: parcel wise values and subject wise values
import src.analysis.p_values as p_values
import src.analysis.statannotations_permutation as statannotations_permutation

def parcel_comparison(df, measure, model_type, save_path_plot=save_path_plot):
    if measure in ['Tau', 'Amyloid','ABeta']:
        figure_name = f'{measure} Parcels N{NPARCELLS}'
    elif measure in ['I_norm2', 'X_norm2']:
        if model_type == 'modelfree':
            figure_name = f'{measure} Parcels Modelfree N{NPARCELLS}'
        else:
            figure_name = f'{measure} Parcels Modelbased N{NPARCELLS} sig{fit_sigma} a{fit_a}'

    group_labels = df['group'].dropna().unique()
    
    group_measure = {group: np.nanmean(np.stack(df[df['group'] == group][measure].values), axis=0) for group in group_labels}

    p_values.plotComparisonAcrossLabels2(
        group_measure,
        custom_test=statannotations_permutation.stat_permutation_test,
        columnLables=group_labels,
        graphLabel=figure_name,
        save_path=os.path.join(save_path_plot, figure_name + '.png')
    )

def subject_comparison(df, measure, model_type, save_path_plot=save_path_plot):
    if measure in ['Tau', 'Amyloid','ABeta']:
        figure_name = f'{measure} Subjects N{NPARCELLS}'
    elif measure in ['I_norm2', 'X_norm2']:
        if model_type == 'modelfree':
            figure_name = f'{measure} Subjects Modelfree N{NPARCELLS}'
        else:
            figure_name = f'{measure} Subjects Modelbased N{NPARCELLS} sig{fit_sigma} a{fit_a}'

    group_labels = df['group'].dropna().unique()
    group_measure = {group: np.nanmean(np.stack(df[df['group'] == group][measure].values), axis=1) for group in group_labels}

    p_values.plotComparisonAcrossLabels_ranksum(
        group_measure,
        columnLables=group_labels,
        graphLabel=figure_name,
        y_axis_label=f'Subject {measure}',
        save_path=os.path.join(save_path_plot, figure_name + '.png')
    )

# parcel_comparison(df, 'ABeta', model_type, save_path_plot=save_path_plot)
# subject_comparison(df, 'ABeta', model_type, save_path_plot=save_path_plot)
Xdict = {
    'biomarkers': np.column_stack([
        np.array([np.mean(x) for x in df['ABeta']]),
        np.array([np.mean(x) for x in df['Tau']]),
        np.array(df['group'].values.reshape(-1, 1))
    ]),
    'all_features': np.column_stack([
        np.array([np.mean(x) for x in df['I_norm2']]),
        np.array([np.mean(x) for x in df['ABeta']]),
        np.array([np.mean(x) for x in df['Tau']]),
        np.array(df['group'].values.reshape(-1, 1))
    ])
}

print(Xdict)
print(np.max([np.max(arr) for arr in df['ABeta'].values if isinstance(arr, np.ndarray)]))