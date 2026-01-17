"""
This script performs various analyses and generates plots based on FDT results 
from both model-free and model-based approaches (compute_FDT_modelfree.py and compute_FDT_modelbased.py).
"""

import os
import numpy as np
import pandas as pd
from src.data_loaders.load_data_records import loadProteins
from src.analysis.plot_functions import (
    emp_sim_triangles,
    plot_ceff_matrices,
    add_parcel_info_to_df,
    plot_correlation_histogram,
    brain_map_correlation,
    brain_map_difference,
    brain_map_parcel_average,
    scatter_plot_3d,
    plot_multi_rsn_split_violin,
    plot_violin_groups_with_significance_parcel_mean,
    plot_violin_groups_with_significance,
    aggregate_all_rsns_for_plotting_parcel_avg,
    aggregate_all_rsns_for_plotting,
    subject_comparison,
    plot,
    radarplot,
)


# Data parameters
DL_type = 'DL_B2'
model_type1 = 'modelfree'
model_type2 = 'modelbased'
NPARCELLS = 400
fit_sigma = True
fit_a = True

# Load data
repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "FDT_DATA")
save_path_plot = os.path.join(repo_root, "data", "RESULT_PLOTS")
filename_based = f"FDT_results_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}_filt.npz"
filename_free = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"
fdt_data_based = np.load(os.path.join(save_path, filename_based), allow_pickle=True)
df_based = pd.DataFrame({k: fdt_data_based[k].tolist() for k in fdt_data_based.files})
fdt_data_free = np.load(os.path.join(save_path, filename_free), allow_pickle=True)
df_free = pd.DataFrame({k: fdt_data_free[k].tolist() for k in fdt_data_free.files})

# Add protein data
df_based = loadProteins(df_based, DL_type, 'Amyloid', repo_root,filt=True)
df_based = loadProteins(df_based, DL_type, 'Tau', repo_root,filt=True)
df_free = loadProteins(df_free, DL_type, 'Amyloid', repo_root,filt=True)
df_free = loadProteins(df_free, DL_type, 'Tau', repo_root,filt=True)

# Add parcel names and RSN info
add_parcel_info_to_df(df_based, NPARCELLS, os.path.join(repo_root, 'hyperparams.json'))
add_parcel_info_to_df(df_free, NPARCELLS,  os.path.join(repo_root, 'hyperparams.json'))

df_based = df_based[:-4] # remove last 4 entries with incomplete data
df_based.reset_index(drop=True, inplace=True)

### ANALYSIS PLOTS ###
df = df_free # model-free: df_free; model-based: df_based
measure = 'I_norm2' # 'I_norm2', 'Tau', 'ABeta'
RSNs = ['Vis','SalVentAttn', 'SomMot', 'DorsAttn', 'Limbic', 'Cont', 'Def']

print(df.columns)
print(df.head())
print(df['subject_id'][2], df['subject_id'][3], df['subject_id'][4], df['subject_id'][5])



def plot_I_norm2_by_location(df, measure='I_norm2', save_path=None):
    """
    Creates a box plot of subject-averaged I_norm2 values grouped by recording location.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'subject_id' and the measure column (e.g., 'I_norm2').
    measure : str
        Column name containing the metric values (default: 'I_norm2').
    save_path : str, optional
        Path to save the figure. If None, the plot is shown but not saved.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with subject averages and their locations.
    """
    import matplotlib.pyplot as plt
    
    # Extract location from subject_id (3 digits after 'S_')
    def extract_location(subject_id):
        parts = subject_id.split('_S_')
        if len(parts) == 2:
            return parts[0][:3]  # Get 3 digits after S_
        return None
    
    # Calculate subject average of I_norm2 for each subject
    subject_averages = []
    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        location = extract_location(subject_id)
        # Average over the list of I_norm2 values for this subject
        i_norm2_values = row[measure]
        if isinstance(i_norm2_values, (list, np.ndarray)):
            avg_value = np.mean(i_norm2_values)
        else:
            avg_value = i_norm2_values
        subject_averages.append({
            'subject_id': subject_id,
            'location': location,
            f'{measure}_avg': avg_value
        })
    
    avg_df = pd.DataFrame(subject_averages)
    
    # Sort locations by number of subjects (descending) for better visualization
    location_counts = avg_df['location'].value_counts()
    sorted_locations = location_counts.index.tolist()
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for box plot
    box_data = [avg_df[avg_df['location'] == loc][f'{measure}_avg'].values 
                for loc in sorted_locations]
    
    bp = ax.boxplot(box_data, labels=sorted_locations, patch_artist=True)
    
    # Style the box plot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Recording Location (Site ID)', fontsize=12)
    ax.set_ylabel(f'Subject Average {measure}', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Add count annotations
    for i, loc in enumerate(sorted_locations):
        count = location_counts[loc]
        ax.annotate(f'n={count}', xy=(i+1, ax.get_ylim()[1]), 
                    ha='center', va='bottom', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return avg_df

avg_df = plot_I_norm2_by_location(df, measure='I_norm2')
### rearrange data for subject average RSN plots (needed for Fig. A6.2 and Fig. 4.2.4)
rsn_sub_df = aggregate_all_rsns_for_plotting(
    df=df,
    measure=measure,
    rsn_names_list=RSNs
)

### Fig. 1.0.1: Amyloid-beta and Tau values between groups
subject_comparison(df, measure='ABeta', model_type='modelbased', NPARCELLS=NPARCELLS, fit_sigma=fit_sigma, fit_a=fit_a, save_path_plot=save_path_plot, rsn_name='All')
subject_comparison(df, measure='Tau', model_type='modelbased', NPARCELLS=NPARCELLS, fit_sigma=fit_sigma, fit_a=fit_a, save_path_plot=save_path_plot, rsn_name='All')

### Fig. 2.2.1: Empirical vs Simulated FC/COVtau comparisons
#emp_sim_triangles(DL_type=DL_type, NPARCELLS=NPARCELLS, fit_sigma=fit_sigma, fit_a=fit_a, joint_normalization=True, n_conditions=4)

### Fig. 2.2.2: Ceff matrices per group
#plot_ceff_matrices(df, groups=['HC', 'MCI(AB+)', 'AD'], n_parcels=NPARCELLS)

### Fig. 4.1.1: Violin plots of subject and parcel comparisons
results = plot_violin_groups_with_significance(
    df_data=df,
    measure_col_name=measure,
    group_col_name='group',
    comparisons=[
        ('HC', 'MCI(AB+)'),
        ('HC', 'AD'),
        ('MCI(AB+)', 'AD')
    ],
    measure_display_name='Integral violation [subject average]',
    save_path='./results/plots')
plot_violin_groups_with_significance_parcel_mean(
    df_data=df,
    measure_col_name=measure,
    comparisons=[
        ('HC', 'MCI(AB+)'),
        ('HC', 'AD'),
        ('MCI(AB+)', 'AD')
    ],
    measure_display_name='Integral violation [parcel average]',
    save_path='./results/plots',
    n_permutations=10000
)

### Fig. 4.2.1: 3D scatter plot
scatter_plot_3d(df)

### Fig. 4.2.2: Brain maps of group averages
brain_map_parcel_average(
    df=df,
    measure=measure,
    group='HC',
    cmap='viridis', #tau:inferno, abeta:RdYlBu, int:viridis
    save_path='./plots',
    vmax=0.09, #tau:2.3, int:0.09, abeta:120
    vmin=0.01  #tau:0.7, int:0.01, abeta:-45
)

### Fig. 4.2.3: Brain maps of correlations between measures
corrs, pvals, top = brain_map_correlation(
    df, mode='between_measures',
    measure1='I_norm2', measure2='Tau',
    #group1='AD',
    cmap='cividis')
plot_correlation_histogram(
    corrs, pvals,
    title='I_norm2 vs Tau',
    save_path=os.path.join(save_path_plot, 'corr_hist_i_norm2_tau_ad.png')
)

### Fig. 4.2.4: Violin plots of RSN comparisons for subject and parcel averages
plot_multi_rsn_split_violin(
    df = df,
    plot_df=rsn_sub_df,
    rsn_order=RSNs,
    groups_order=['HC', 'AD'],
    measure_name=measure,
    y_label_override='Integral violation [subject average]',
    test_type='ranksum',
)
plot_ready_df_rsn = aggregate_all_rsns_for_plotting_parcel_avg(
    df=df,
    measure=measure,
    rsn_names_list=RSNs
)
plot_multi_rsn_split_violin(
    df=df,
    plot_df=plot_ready_df_rsn,
    rsn_order=RSNs,
    groups_order=['HC', 'AD'],
    measure_name='Integral Violation',
    y_label_override='Integral violation [parcel average]',
    test_type='permutation',
    n_permutations=10000,
)

### Fig. A3.3: histograms and single subject/group means of model parameters
plot(df_based, group=None, measure="a", mode="both")
plot(df_based, group=None, measure="sigma", mode="both")
plot(df_based, group=None, measure="f_diff", mode="both")

### Fig. A6.1: Brain maps of normalized changes from HC to AD
brain_map_difference(
    df=df,
    measure=measure,
    group1='HC',
    group2='AD',
    cmap='cividis', 
    save_path='./plots',
    vmax=1,
    vmin=0
)

### Fig. A6.2: Radar plots of RSN averages per group
radarplot(rsn_sub_df,RSNs)

### Fit statistics:
corr_fc_values = df_based['losses'].apply(lambda d: d['corr_fc'])
COVtau_corr_values = df_based['losses'].apply(lambda d: d['corr_covtau'])
mse_fc_values = df_based['losses'].apply(lambda d: d['mse_fc'])
mse_COVtau_values = df_based['losses'].apply(lambda d: d['mse_covtau'])
print(f"Correlation FC values:\n{1-corr_fc_values.mean()} ± {corr_fc_values.std()}")
print(f"Correlation COVtau values:\n{1-COVtau_corr_values.mean()} ± {COVtau_corr_values.std()}")
print(f"MSE FC values:\n{mse_fc_values.mean()} ± {mse_fc_values.std()}")
print(f"MSE COVtau values:\n{mse_COVtau_values.mean()} ± {mse_COVtau_values.std()}")