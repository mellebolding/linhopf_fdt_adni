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
import matplotlib.pyplot as plt 
from scipy import stats 


# --- FUNCTION TO ISOLATE CORE FITTING LOGIC (UNCHANGED) ---
def run_single_fit(t, params, NPARCELLS, DL_type, SC_400):
    """Encapsulates the data loading, fitting, and result aggregation for one run."""
    
    # --- 1. Load Data Loaders (DL_type specific) ---
    if DL_type == 'DL_A':
        DL = ADNI_A.ADNI_A(normalizeBurden=False)
    elif DL_type == 'DL_B1':
        DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'])
    elif DL_type == 'DL_B2':
        DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD'])
    elif DL_type == 'DL_B3':
        DL = ADNI_B.ADNI_B_Alt(['HC(AB-)', 'HC(AB+)', 'MCI(AB-)', 'MCI(AB+)', 'AD'])
    else:
        raise ValueError(f"Unknown DL_type: {DL_type}")

    # --- 2. Load and Prepare Data ---
    all_data = []
    for group in DL.get_groupLabels():
        DLA = ADNI_A.ADNI_A(normalizeBurden=False)
        SC_HC_Avg = DLA.get_AvgSC_ctrl()
        SC_400 = np.pad(
            SC_HC_Avg,
            pad_width=((0, 21), (0, 21)),
            mode='constant',
            constant_values=SC_HC_Avg.mean()
        )
        all_data.extend(load_data_records.load_group_data(DL, group, DL_type, SC=SC_400))
        
    TSemp_zsc = zfilterts.prepare_timeseries(all_data, NPARCELLS)
    group_data = load_data_records.prepare_group_data(all_data, NPARCELLS)

    # --- 3. Fitting Loop (UNCHANGED) ---
    results = []
    group_results = []
    groups_fitted = set()
    
    params["hopfParamsAdam"]['tau'] = t 
    
    for idx, subj_data in enumerate(all_data):
        current_group = subj_data['group']
        
        # Group Fitting
        if current_group not in groups_fitted:
            single_group_result = LinHopfFit.fit_linhopf(group_data[current_group], None, sigma_ini, a_ini, verbose,params, NPARCELLS)
            single_group_result.update({
                'group': current_group,
                'f_diff': group_data[current_group]['f_diff'][:NPARCELLS],
                'SC': group_data[current_group]['SC'][:NPARCELLS, :NPARCELLS],
            })
            group_results.append(single_group_result)
            groups_fitted.add(current_group)

        # Subject Fitting
        single_subject_result = LinHopfFit.fit_linhopf(subj_data, TSemp_zsc[idx], sigma_ini, a_ini, verbose, params, NPARCELLS)
        single_subject_result.update({
            'subject_id': subj_data['subject_id'],
            'group': subj_data['group'],
            'f_diff': subj_data['f_diff'][:NPARCELLS],
            'SC': subj_data['SC'][:NPARCELLS, :NPARCELLS],
        })
        results.append(single_subject_result)

    # --- 4. Final Calculation (UNCHANGED) ---
    df = pd.DataFrame(results + group_results)
    
    COVtauemp = np.stack(df['COVtauemp'], axis=0)
    COVtausim = np.stack(df['COVtausim'], axis=0)
    
    diagonalsemp = np.diagonal(COVtauemp, axis1=1, axis2=2)
    diagonalssim = np.diagonal(COVtausim, axis1=1, axis2=2)
    
    average_autocorr_emp = np.mean(diagonalsemp)
    average_autocorr_sim = np.mean(diagonalssim)
    
    return average_autocorr_emp, average_autocorr_sim

# --- MAIN EXECUTION ---

start_time = time.time()

# --- INPUT PARAMETERS ---
DL_type = 'DL_B1'
NPARCELLS = 20
fit_sigma = True
fit_a = True
verbose = False
sigma_ini = 0.45 * np.ones(NPARCELLS)
a_ini = -0.02 * np.ones(NPARCELLS)

with open("hyperparams.json", "r") as f:
    params = json.load(f)

tau_values = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
N_REPEATS = 1

# --- DATA STRUCTURE TO STORE RESULTS ---
plot_data = {
    'tau': [], 
    'emp_mean': [], 'emp_ci': [], 
    'sim_mean': [], 'sim_ci': []
}
all_runs_results = {t: {'emp': [], 'sim': []} for t in tau_values}

print(f"Starting simulation for {len(tau_values)} tau values with {N_REPEATS} repeats each...")
print("-" * 50)

# --- NESTED LOOPS FOR REPETITION AND CALCULATION (CI CALCULATION MODIFIED) ---
for t in tau_values:
    print(f"Processing tau = {t}")
    for run in range(N_REPEATS):
        print(f"  Run {run + 1}/{N_REPEATS}")
        
        # --- CALL THE FITTING FUNCTION ---
        try:
            emp_autocorr, sim_autocorr = run_single_fit(t, params, NPARCELLS, DL_type, None)
        except Exception as e:
            print(f"Error during fit for tau={t}, run={run+1}: {e}")
            emp_autocorr, sim_autocorr = np.nan, np.nan 
        
        all_runs_results[t]['emp'].append(emp_autocorr)
        all_runs_results[t]['sim'].append(sim_autocorr)

    # --- CALCULATE STATISTICS FOR PLOTTING (MODIFIED FOR 98% CI) ---
    emp_vals = np.array(all_runs_results[t]['emp'])
    sim_vals = np.array(all_runs_results[t]['sim'])
    
    # Filter out NaNs if any run failed
    emp_vals = emp_vals[~np.isnan(emp_vals)]
    sim_vals = sim_vals[~np.isnan(sim_vals)]

    if len(emp_vals) > 1:
        # Standard Error (SE) = STD / sqrt(N_valid)
        emp_se = np.std(emp_vals, ddof=1) / np.sqrt(len(emp_vals))
        sim_se = np.std(sim_vals, ddof=1) / np.sqrt(len(sim_vals))
        
        # --- CALCULATE 98% CI T-SCORE ---
        # T-score for 98% CI (alpha=0.02, 1 - alpha/2 = 0.99)
        T_SCORE_98 = stats.t.ppf(0.99, len(emp_vals) - 1) 
        
        plot_data['tau'].append(t)
        plot_data['emp_mean'].append(np.mean(emp_vals))
        plot_data['emp_ci'].append(T_SCORE_98 * emp_se)
        plot_data['sim_mean'].append(np.mean(sim_vals))
        plot_data['sim_ci'].append(T_SCORE_98 * sim_se) # Using the larger T-score
    elif len(emp_vals) == 1:
        plot_data['tau'].append(t)
        plot_data['emp_mean'].append(np.mean(emp_vals))
        plot_data['emp_ci'].append(0.0)
        plot_data['sim_mean'].append(np.mean(sim_vals))
        plot_data['sim_ci'].append(0.0)
    else:
        print(f"Skipping tau={t}: No valid runs completed.")

print("-" * 50)
end_time = time.time()
print(f"Total computation time: {end_time - start_time:.2f} seconds")

# --- PLOTTING (MODIFIED FOR 98% CI LABEL AND REMOVED SCALING) ---
if plot_data['tau']:
    
    # Convert lists to NumPy arrays for easy plotting
    tau = np.array(plot_data['tau'])
    emp_mean = np.array(plot_data['emp_mean'])
    sim_mean = np.array(plot_data['sim_mean'])
    sim_ci = np.array(plot_data['sim_ci']) # This now holds the 98% CI width
    
    # Calculate CI bounds for fill_between
    sim_lower = sim_mean - sim_ci
    sim_upper = sim_mean + sim_ci

    plt.figure(figsize=(10, 6))

    # --- Empirical Data Plot (Mean Line Only) ---
    plt.plot(tau, emp_mean, 'o-', label='Empirical Mean', 
             color='darkblue', linewidth=2, zorder=3)
    
    # --- Simulated Data Plot (Line and Shaded CI) ---
    plt.plot(tau, sim_mean, 's-', label='Simulated Mean', 
             color='darkred', linewidth=2, zorder=4)
    
    # Fill between the bounds with 98% CI
    plt.fill_between(tau, sim_lower, sim_upper, 
                     color='red', alpha=0.3, label='98% CI (Simulated)', zorder=1) # Label updated to 98%

    # --- Formatting and Aesthetics ---
    plt.xlabel('$\\tau$ Parameter (Time Delay)', fontsize=14)
    plt.ylabel('Average Covariance Diagonal Value', fontsize=14) 
    
    plt.xticks(tau)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=12, frameon=True) 
    
    plt.tick_params(axis='both', which='major', labelsize=12) 
    
    # Save the plot
    plt.plot()
    plot_filename = 'tau_autocorr_vs_tau_ci.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
else:
    print("No data available to plot.")