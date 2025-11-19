import os
import numpy as np
import src.data_loaders.ADNI_A as ADNI_A
import src.data_loaders.ADNI_B as ADNI_B
import src.data_processing.zfilterts as zfilterts
import src.data_loaders.load_data_records as load_data_records
import src.functions_frameworks.LinHopfFit as LinHopfFit
import json
import time
import pandas as pd
from typing import Union


from typing import Tuple
def calculate_parcel_variance(
    ts_array: np.ndarray, 
    axis: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the variance and standard deviation for each parcel in a 3D time series array.

    The input array is expected to be Z-scored and have the shape (N_SUBJECTS, N_PARCELS, N_STEPS).
    The calculation is performed across the time axis (axis=2 by default).

    Args:
        ts_array: A 3D numpy array of time series data (Subjects, Parcels, Time Steps).
        axis: The axis along which to calculate the variance (usually 2, the time axis).

    Returns:
        A tuple containing:
        - subject_variances: Array of shape (N_SUBJECTS, N_PARCELS) containing the variance 
          per parcel for each subject.
        - group_mean_std_dev: Array of shape (N_PARCELS,) containing the mean standard 
          deviation across all subjects for each parcel.
    """
    # ----------------------------------------------------
    # 1. Variance per Parcel for *Each Subject*
    # ----------------------------------------------------
    # We calculate the variance along the time axis (axis=2).
    # The result has shape (N_SUBJECTS, N_PARCELS).
    subject_variances = np.var(ts_array, axis=axis)

    # ----------------------------------------------------
    # 2. Group Mean Standard Deviation
    # ----------------------------------------------------
    # Standard deviation is often more interpretable than variance.
    subject_std_devs = np.sqrt(subject_variances)
    
    # Calculate the mean standard deviation across all subjects for each parcel.
    group_mean_std_dev = np.mean(subject_std_devs, axis=0)

    # Note: For perfectly z-scored data, both subject_variances and group_mean_std_dev 
    # should be very close to 1.0. Any significant deviation suggests a problem 
    # in the initial z-scoring process or subsequent data trimming/filtering.
    
    return subject_variances, group_mean_std_dev

start_time = time.time()

### MAIN INPUT PARAMETERS (other parameters from json file)
DL_type = 'DL_A'
NPARCELLS = 20 # max 379
fit_sigma = True
fit_a = True
verbose = False
sigma_ini = 0.45 * np.ones(NPARCELLS)
a_ini = -0.02 * np.ones(NPARCELLS)

with open("hyperparams.json", "r") as f:
    params = json.load(f)

DL = ADNI_A.ADNI_A(normalizeBurden=False)
SC_HC_Avg = DL.get_AvgSC_ctrl()

SC_400 = np.pad(
    SC_HC_Avg,
    pad_width=((0, 21), (0, 21)),
    mode='constant',
    constant_values=SC_HC_Avg.mean()
)
if DL_type == 'DL_B1':
    DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'])
if DL_type == 'DL_B2':
    DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD'])
if DL_type == 'DL_B3':
    DL = ADNI_B.ADNI_B_Alt(['HC(AB-)', 'HC(AB+)', 'MCI(AB-)', 'MCI(AB+)', 'AD'])

all_data = []
for group in DL.get_groupLabels():
    print(f"Loading {group}...")
    all_data.extend(load_data_records.load_group_data(DL, group, SC=SC_400))
    print(all_data[0]['MRI'][:,:20])
    sub_var_all, group_std_all = calculate_parcel_variance(np.array([all_data[0]['MRI'][:,:20]]))
    print(np.average(sub_var_all), np.average(group_std_all))
    #print(all_data[0]['MRI'].shape)
    mri_values = all_data[1]['MRI']
    print(f"Max MRI value: {np.max(mri_values)}")
    print(f"Min MRI value: {np.min(mri_values)}")
    print(f"Mean MRI value: {np.mean(mri_values)}")
TSemp_zsc = zfilterts.prepare_timeseries(all_data, NPARCELLS)
group_data = load_data_records.prepare_group_data(all_data, NPARCELLS)
sub_var, group_std = calculate_parcel_variance(TSemp_zsc)
print(np.average(sub_var), np.average(group_std))
results = []
group_results = []
groups_fitted = set()

for idx, subj_data in enumerate(all_data):
    
    current_group = subj_data['group']
    if current_group not in groups_fitted:
        print(f"\nFitting {current_group} group model...")

        single_group_result = LinHopfFit.fit_linhopf(group_data[current_group], None, sigma_ini, a_ini, verbose,params, NPARCELLS)
        single_group_result.update({
            'group': current_group,
            'f_diff': group_data[current_group]['f_diff'][:NPARCELLS],
            'SC': group_data[current_group]['SC'][:NPARCELLS, :NPARCELLS],
        })
        group_results.append(single_group_result)
        groups_fitted.add(current_group)

    single_subject_result = LinHopfFit.fit_linhopf(subj_data, TSemp_zsc[idx], sigma_ini, a_ini, verbose, params, NPARCELLS)
    single_subject_result.update({
        'subject_id': subj_data['subject_id'],
        'group': subj_data['group'],
        'f_diff': subj_data['f_diff'][:NPARCELLS],
        'SC': subj_data['SC'][:NPARCELLS, :NPARCELLS],
    })
    results.append(single_subject_result)
    if (idx + 1) % 10 == 0:
        print(f"Fitted {idx + 1}/{len(all_data)} individual subjects")

end_time = time.time()
print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

#Save results to npz
df = pd.DataFrame(results + group_results)
results_dict = {str(key): value for key, value in df.to_dict(orient="list").items()}
#print(df['FCsim'].values)

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "HOPF_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"linhopf_fit_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)
