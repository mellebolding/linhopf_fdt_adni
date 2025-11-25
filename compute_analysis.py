import os
import numpy as np
import pandas as pd
from src.data_loaders.load_data_records import loadProteins
from src.analysis.p_values import parcel_comparison_rsn, subject_comparison_rsn
from src.analysis.p_values import parcel_comparison, subject_comparison
import json

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 1. Data Aggregation Function (Modified from subject_comparison_rsn) ---

def aggregate_all_rsns_for_plotting(df, measure: str, rsn_names_list: list
                                    ) -> pd.DataFrame:
    """
    Processes all RSNs and returns a single, long-format DataFrame 
    for multi-RSN plotting.
    """
    if 'parcel_RSNs' not in df.columns or df['parcel_RSNs'].empty:
        raise KeyError("DataFrame must contain 'parcel_RSNs' column for RSN selection.")
             
    all_plot_data = []
    # Get the parcel-to-RSN mapping (consistent across subjects)
    parcel_rsn_map = df['parcel_RSNs'].iloc[0]

    for rsn_name in rsn_names_list:
        # Get RSN-specific parcel indices
        rsn_indices = [i for i, rsn in enumerate(parcel_rsn_map) if rsn == rsn_name]
        
        if not rsn_indices:
            print(f"Warning: No parcels found for RSN: {rsn_name}. Skipping.")
            continue
            
        # Iterate through each subject to calculate the RSN average
        for idx, row in df.iterrows():
            subject_group = row['group']
            
            # --- FIX IS HERE ---
            # 1. Ensure the measure array is a NumPy array
            subject_measure_array = np.array(row[measure])
            
            # 2. Indexing will now work correctly with the list of indices
            if subject_measure_array.ndim == 0:
                subject_measure_array = subject_measure_array.reshape(1) 
                
            rsn_specific_measures = subject_measure_array[rsn_indices]
            
            # Calculate mean across parcels
            mean_measure_for_rsn = np.nanmean(rsn_specific_measures)
            
            all_plot_data.append({
                'Group': subject_group,
                'RSN': rsn_name,
                'Value': mean_measure_for_rsn
            })
            
    return pd.DataFrame(all_plot_data)

# --- 2. Multi-RSN Plotting Function ---

def plot_multi_rsn_split_violin(plot_df: pd.DataFrame, rsn_order: list, groups_order: list, 
                          measure_name: str, y_label_override=None, 
                          palette=None, dpi=300):
    """
    Generates a split violin plot comparison across ALL RSNs (Figure 1a style).
    """
    
    if len(groups_order) != 2:
        raise ValueError("Multi-RSN split violin plot requires exactly two groups.")

    graphLabel = f'Level of hierarchy per subject across RSN'
    y_label = y_label_override if y_label_override else f'Subject {measure_name} Avg'

    # Prepare palette
    if palette is None:
        palette = {
            groups_order[0]: '#8BC34A',  # Light Green
            groups_order[1]: '#1ABC9C'   # Dark Green
        }
    
    fig, ax = plt.subplots(figsize=(10, 6)) # Wide figure for RSNs
    sns.set_context('notebook', font_scale=1.0)
    
    # 1. Create the Split Violin Plot (x=RSN, hue=Group, split=True)
    sns.violinplot(
        x='RSN',
        y='Value',
        hue='Group',
        data=plot_df,
        order=rsn_order,
        hue_order=groups_order,
        palette=palette,
        split=True,             # KEY: This creates the split distribution per RSN
        inner='quartile',       # Adds the black bars (median, quartiles)
        linewidth=0.8,
        ax=ax
    )

    # 2. Add Individual Data Points
    sns.stripplot(
        x='RSN',
        y='Value',
        hue='Group',
        data=plot_df,
        order=rsn_order,
        hue_order=groups_order,
        palette=palette,
        edgecolor='black',
        linewidth=0.7,
        s=3,
        jitter=0.2,
        alpha=0.6,
        ax=ax,
        dodge=True
    )
    ax.tick_params(axis='y', length=0) 
    ax.tick_params(axis='x', length=0)
    
    # Remove the top and right spines (border lines)
    # The bottom spine is needed for the RSN labels, and the left spine for the Y-axis.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Optional: Make the remaining spines (left and bottom) thinner/cleaner
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    # --- Legend and Labels ---
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[0:len(groups_order)], labels_leg[0:len(groups_order)], 
              loc='lower right', title=None, frameon=True)

    ax.set_title(graphLabel, fontsize=14, pad=15)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel('', fontsize=12) # Empty X-label as RSN names are clear

    # --- 3. Statistical Testing and Annotation (Per RSN) ---
    
    # max_y = plot_df['Value'].max()
    # y_range = plot_df['Value'].max() - plot_df['Value'].min()
    max_y = np.nanmax(plot_df['Value'])
    y_range = np.nanmax(plot_df['Value']) - np.nanmin(plot_df['Value'])
    y_increment = y_range * 0.04 if y_range > 0 else 1.0
    yposition = max_y + y_increment 

    for i, rsn in enumerate(rsn_order):
        
        rsn_data_subset = plot_df[plot_df['RSN'] == rsn]
        group1_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[0]]['Value']
        group2_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[1]]['Value']
        
        if not group1_values.empty and not group2_values.empty:
            p_value = stats.ranksums(group1_values, group2_values).pvalue
            
            sig = '***' if p_value < 0.001 else \
                  '**' if p_value < 0.01 else \
                  '*' if p_value < 0.05 else \
                  'n.s.'

            if sig:
                ax.text(i, yposition, sig, 
                        ha='center', va='bottom', color='black', fontsize=12)

    # Adjust Y-axis limit for markers
    ax.set_ylim(top=yposition + y_increment * 1.5)
    
    # Final save
    plt.tight_layout()
    plt.show()
    plt.close(fig)



def permutation_test(group1, group2, n_permutations=10000):
    """
    Perform a permutation test to compare two groups.
    
    Args:
        group1, group2: Arrays of values for each group
        n_permutations: Number of permutations
    
    Returns:
        p_value: Two-tailed p-value
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Combine all data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
    
    perm_diffs = np.array(perm_diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return p_value


def aggregate_all_rsns_for_plotting_parcel_avg(df, measure: str, rsn_names_list: list) -> pd.DataFrame:
    """
    Processes all RSNs and returns parcel-level averages (not subject averages).
    Each row represents a parcel's average across all subjects in a group.
    """
    if 'parcel_RSNs' not in df.columns or df['parcel_RSNs'].empty:
        raise KeyError("DataFrame must contain 'parcel_RSNs' column for RSN selection.")
    
    all_plot_data = []
    parcel_rsn_map = df['parcel_RSNs'].iloc[0]
    
    for rsn_name in rsn_names_list:
        rsn_indices = [i for i, rsn in enumerate(parcel_rsn_map) if rsn == rsn_name]
        
        if not rsn_indices:
            print(f"Warning: No parcels found for RSN: {rsn_name}. Skipping.")
            continue
        
        # Group subjects by their group label
        for group_label in df['group'].unique():
            group_df = df[df['group'] == group_label]
            
            # For each parcel in this RSN
            for parcel_idx in rsn_indices:
                # Collect this parcel's values across all subjects in this group
                parcel_values_across_subjects = []
                
                for idx, row in group_df.iterrows():
                    subject_measure_array = np.array(row[measure])
                    parcel_values_across_subjects.append(subject_measure_array[parcel_idx])
                
                # Average across subjects for this parcel
                parcel_avg = np.nanmean(parcel_values_across_subjects)
                
                all_plot_data.append({
                    'Group': group_label,
                    'RSN': rsn_name,
                    'Value': parcel_avg
                })
    
    return pd.DataFrame(all_plot_data)


def aggregate_models_for_plotting_parcel_avg(model_data_dicts: dict, groups_order: list, 
                                            measure_col_name: str = None) -> pd.DataFrame:
    """
    Aggregates model data at the parcel level (not subject level).
    
    Args:
        model_data_dicts: Dictionary in ONE of two formats:
            Format 1 (dict of dicts):
                {
                    'Model-free': {'HC': [values], 'AD': [values]},
                    'Model-based': {'HC': [values], 'AD': [values]}
                }
            Format 2 (dict of DataFrames):
                {
                    'Model-free': df_free,  # DataFrame with 'group' and measure columns
                    'Model-based': df_based
                }
        groups_order: List of group labels, e.g., ['HC', 'AD']
        measure_col_name: Name of measure column in DataFrames (REQUIRED for Format 2)
    
    Returns:
        Long-format DataFrame with columns: ['Model', 'Group', 'Value']
        where each Value is a parcel average across subjects
    """
    all_plot_data = []
    
    # Check format by examining first value
    first_value = next(iter(model_data_dicts.values()))
    
    # Format 1: dict of dicts with group data
    if isinstance(first_value, dict):
        print("Detected Format 1: dict of dicts")
        for model_name, group_data_dict in model_data_dicts.items():
            for group_label in groups_order:
                if group_label in group_data_dict:
                    values = group_data_dict[group_label]
                    
                    # Each value should be a scalar (parcel average)
                    for value in values:
                        all_plot_data.append({
                            'Model': model_name,
                            'Group': group_label,
                            'Value': value
                        })
    
    # Format 2: dict of DataFrames
    elif hasattr(first_value, 'columns'):
        print("Detected Format 2: dict of DataFrames")
        
        if measure_col_name is None:
            raise ValueError("measure_col_name is required when using DataFrame format. "
                           "Please specify which column to use (e.g., 'I_norm2' or 'X_norm2')")
        
        for model_name, df in model_data_dicts.items():
            if measure_col_name not in df.columns:
                raise ValueError(f"Column '{measure_col_name}' not found in {model_name} DataFrame. "
                               f"Available columns: {df.columns.tolist()}")
            
            measure = measure_col_name
            
            # For each group
            for group_label in groups_order:
                group_df = df[df['group'] == group_label]
                
                if group_df.empty:
                    continue
                
                # Get number of parcels from first subject
                first_subject_measure = np.array(group_df[measure].iloc[0])
                n_parcels = len(first_subject_measure)
                
                # For each parcel
                for parcel_idx in range(n_parcels):
                    # Collect this parcel's values across all subjects in this group
                    parcel_values_across_subjects = []
                    
                    for idx, row in group_df.iterrows():
                        subject_measure_array = np.array(row[measure])
                        parcel_values_across_subjects.append(subject_measure_array[parcel_idx])
                    
                    # Average across subjects for this parcel
                    parcel_avg = np.nanmean(parcel_values_across_subjects)
                    
                    all_plot_data.append({
                        'Model': model_name,
                        'Group': group_label,
                        'Value': parcel_avg
                    })
    else:
        raise ValueError(f"Unknown format for model_data_dicts. First value type: {type(first_value)}")
    
    return pd.DataFrame(all_plot_data)


def plot_multi_model_comparison_violin(model_data_dicts: dict, groups_order: list, 
                                       measure_name: str, y_label_override=None,
                                       palette=None, dpi=300, save_path=None, 
                                       n_permutations=10000, measure_col_name=None,
                                       test_type='permutation'):
    """
    Generates a split violin plot comparison across different models.
    Uses parcel averages and statistical testing.
    
    Args:
        model_data_dicts (dict): Dictionary in one of two formats:
            - Dict of dicts: {'Model-free': {'HC': [...], 'AD': [...]}, ...}
            - Dict of DataFrames: {'Model-free': df_free, 'Model-based': df_based}
        groups_order (list): e.g., ['HC', 'AD']
        measure_name (str): e.g., 'Integral Violation'
        measure_col_name (str): Column name if using DataFrame format
        test_type (str): 'permutation' for parcel-level data (default), 
                        'ranksum' for subject-level data
        n_permutations (int): Number of permutations if test_type='permutation'
    """
    from scipy.stats import mannwhitneyu, ranksums
    
    if len(groups_order) != 2:
        raise ValueError("Comparison plot requires exactly two groups (e.g., HC and AD).")
    
    if test_type not in ['permutation', 'ranksum']:
        raise ValueError("test_type must be 'permutation' or 'ranksum'")
    
    # Convert to parcel-averaged long format
    plot_df = aggregate_models_for_plotting_parcel_avg(model_data_dicts, groups_order, measure_col_name)
    print(plot_df.columns)
    model_names = list(model_data_dicts.keys())
    
    # Prepare palette
    if palette is None:
        palette = {
            groups_order[0]: '#8BC34A',  # Light Green
            groups_order[1]: '#1ABC9C'   # Turquoise
        }
    
    # Figure setup
    fig_width = 6 + 2 * len(model_names)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.set_context('notebook', font_scale=1.0)
    
    graphLabel = f'Level of {measure_name} across Models'
    y_label = y_label_override if y_label_override else f'Parcel {measure_name} Avg'
    
    # 1. Split Violin Plot
    sns.violinplot(
        x='Model',
        y='Value',
        hue='Group',
        data=plot_df,
        order=model_names,
        hue_order=groups_order,
        palette=palette,
        split=True,
        inner='quartile',
        linewidth=0.8,
        ax=ax
    )
    
    # 2. Add Individual Data Points
    sns.stripplot(
        x='Model',
        y='Value',
        hue='Group',
        data=plot_df,
        order=model_names,
        hue_order=groups_order,
        palette=palette,
        edgecolor='black',
        linewidth=0.7,
        s=3,
        jitter=0.2,
        alpha=0.6,
        ax=ax,
        dodge=True
    )
    
    # 3. Aesthetics
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # 4. Legend and Labels
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[0:len(groups_order)], labels_leg[0:len(groups_order)], 
              loc='upper right', title=None, frameon=True)
    
    ax.set_title(graphLabel, fontsize=14, pad=15)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel('', fontsize=12)
    
    # 5. Statistical Testing
    #max_y = plot_df['Value'].max()
    max_y = np.nanmax(plot_df['Value'])
    y_range = np.nanmax(plot_df['Value']) - np.nanmin(plot_df['Value'])
    y_increment = y_range * 0.04 if y_range > 0 else 1.0
    yposition = max_y + y_increment
    
    test_name = "Permutation Test" if test_type == 'permutation' else "Wilcoxon Rank-Sum Test"
    print(f"\n=== {test_name} Results for Models ===")
    
    for i, model in enumerate(model_names):
        model_data_subset = plot_df[plot_df['Model'] == model]
        group1_values = model_data_subset[model_data_subset['Group'] == groups_order[0]]['Value'].values
        group2_values = model_data_subset[model_data_subset['Group'] == groups_order[1]]['Value'].values
        
        if len(group1_values) > 0 and len(group2_values) > 0:
            # Choose test type
            if test_type == 'permutation':
                p_value = permutation_test(group1_values, group2_values, n_permutations=n_permutations)
                print(f"{model}: p={p_value:.4f} (n1={len(group1_values)}, n2={len(group2_values)}, "
                      f"{n_permutations} permutations)")
            else:  # ranksum
                statistic, p_value = ranksums(group1_values, group2_values)
                print(f"{model}: p={p_value:.4f} (statistic={statistic:.4f}, "
                      f"n1={len(group1_values)}, n2={len(group2_values)})")
            
            sig = '***' if p_value < 0.001 else \
                  '**' if p_value < 0.01 else \
                  '*' if p_value < 0.05 else \
                  'n.s.'
            
            if sig != 'n.s.':
                ax.text(i, yposition, sig, ha='center', va='bottom', 
                       color='black', fontsize=12)
    
    ax.set_ylim(top=yposition + y_increment * 1.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model_string = "_".join(model_names).replace(" ", "").replace("(", "").replace(")", "")
        plot_filename = os.path.join(save_path, f"ModelComparison_parcel_avg_{model_string}.png")
        plt.savefig(plot_filename, dpi=dpi)
        print(f"\nPlot saved to {plot_filename}")
    
    plt.show()
    plt.close(fig)


def plot_multi_rsn_split_violin(plot_df: pd.DataFrame, rsn_order: list, groups_order: list, 
                          measure_name: str, y_label_override=None, 
                          palette=None, dpi=300, save_path=None, 
                          n_permutations=10000, test_type='permutation'):
    """
    Generates a split violin plot comparison across ALL RSNs.
    Uses permutation test or rank sum test for statistical comparisons.
    
    Args:
        test_type (str): 'permutation' for parcel-level data (default), 
                        'ranksum' for subject-level data
    """
    from scipy.stats import ranksums
    
    if len(groups_order) != 2:
        raise ValueError("Multi-RSN split violin plot requires exactly two groups.")
    
    if test_type not in ['permutation', 'ranksum']:
        raise ValueError("test_type must be 'permutation' or 'ranksum'")

    graphLabel = f'Level of {measure_name} per parcel across RSN'
    y_label = y_label_override if y_label_override else f'Parcel {measure_name} Avg'

    # Prepare palette
    if palette is None:
        palette = {
            groups_order[0]: '#8BC34A',  # Light Green
            groups_order[1]: '#1ABC9C'   # Dark Green
        }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_context('notebook', font_scale=1.0)
    
    # 1. Create the Split Violin Plot
    sns.violinplot(
        x='RSN',
        y='Value',
        hue='Group',
        data=plot_df,
        order=rsn_order,
        hue_order=groups_order,
        palette=palette,
        split=True,
        inner='quartile',
        linewidth=0.8,
        ax=ax
    )

    # 2. Add Individual Data Points
    sns.stripplot(
        x='RSN',
        y='Value',
        hue='Group',
        data=plot_df,
        order=rsn_order,
        hue_order=groups_order,
        palette=palette,
        edgecolor='black',
        linewidth=0.7,
        s=3,
        jitter=0.2,
        alpha=0.6,
        ax=ax,
        dodge=True
    )
    
    ax.tick_params(axis='y', length=0) 
    ax.tick_params(axis='x', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    
    # Legend and Labels
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[0:len(groups_order)], labels_leg[0:len(groups_order)], 
              loc='lower right', title=None, frameon=True)

    ax.set_title(graphLabel, fontsize=14, pad=15)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel('', fontsize=12)

    # 3. Statistical Testing
    max_y = plot_df['Value'].max()
    y_range = plot_df['Value'].max() - plot_df['Value'].min()
    y_increment = y_range * 0.04 if y_range > 0 else 1.0
    yposition = max_y + y_increment 

    test_name = "Permutation Test" if test_type == 'permutation' else "Wilcoxon Rank-Sum Test"
    print(f"\n=== {test_name} Results for RSNs ===")
    
    for i, rsn in enumerate(rsn_order):
        rsn_data_subset = plot_df[plot_df['RSN'] == rsn]
        group1_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[0]]['Value'].values
        group2_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[1]]['Value'].values
        
        if len(group1_values) > 0 and len(group2_values) > 0:
            # Choose test type
            if test_type == 'permutation':
                p_value = permutation_test(group1_values, group2_values, n_permutations=n_permutations)
                print(f"{rsn}: p={p_value:.4f} (n1={len(group1_values)}, n2={len(group2_values)}, "
                      f"{n_permutations} permutations)")
            else:  # ranksum
                statistic, p_value = ranksums(group1_values, group2_values)
                print(f"{rsn}: p={p_value:.4f} (statistic={statistic:.4f}, "
                      f"n1={len(group1_values)}, n2={len(group2_values)})")
            
            sig = '***' if p_value < 0.001 else \
                  '**' if p_value < 0.01 else \
                  '*' if p_value < 0.05 else \
                  'n.s.'

            if sig != 'n.s.':
                ax.text(i, yposition, sig, 
                        ha='center', va='bottom', color='black', fontsize=12)

    ax.set_ylim(top=yposition + y_increment * 1.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plot_filename = os.path.join(save_path, f"RSN_comparison_parcel_avg_{measure_name}.png")
        plt.savefig(plot_filename, dpi=dpi)
        print(f"\nPlot saved to {plot_filename}")
    
    plt.show()
    plt.close(fig)

def scatter_plot_3d(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import gaussian_kde 

    # --- 0. Setup and Data Aggregation ---
    group_means = pd.DataFrame({
        "ABeta":   df.groupby("group")["ABeta"].apply(lambda x: np.vstack(x).mean(axis=0)),
        "Tau":     df.groupby("group")["Tau"].apply(lambda x: np.vstack(x).mean(axis=0)),
        "I_norm2": df.groupby("group")["I_norm2"].apply(lambda x: np.vstack(x).mean(axis=0)),
    })
    groups = group_means.index.tolist()
    
    color_map = {
        "HC":       '#8BC34A', 
        "MCI(AB-)": "#D6A213",   
        "MCI(AB+)": "#C11C0A",   
        "AD":       '#1ABC9C'    
    }


    # Prepare ranges for density plots
    x_data_all = np.concatenate(group_means["ABeta"].values)
    y_data_all = np.concatenate(group_means["Tau"].values)
    z_data_all = np.concatenate(group_means["I_norm2"].values)
    
    x_min, x_max = x_data_all.min(), x_data_all.max()
    y_min, y_max = y_data_all.min(), y_data_all.max()
    z_min, z_max = z_data_all.min(), z_data_all.max()

    x_lin = np.linspace(x_min, x_max, 200) # ABeta
    y_lin = np.linspace(y_min, y_max, 200) # Tau
    z_lin = np.linspace(z_min, z_max, 200) # I_norm2


    # #######################################################
    # 1. 3D SCATTER PLOT (Tau on X-axis, ABeta on Y-axis)
    # #######################################################
    print("Generating 3D Scatter Plot...")
    fig_3d = plt.figure(figsize=(10, 8)) 
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    
    # Side view adjustment
    ax_3d.view_init(elev=17, azim=42)
    
    for g in groups:
        ab = group_means.loc[g, "ABeta"]
        ta = group_means.loc[g, "Tau"]
        ii = group_means.loc[g, "I_norm2"]
        
        # AXIS SWAP: X=Tau, Y=ABeta, Z=I_norm2
        ax_3d.scatter(
            ta, # X-axis
            ab, # Y-axis
            ii, # Z-axis
            s=18,                       # Slightly larger size
            alpha=.7,                  # Maximize border visibility
            facecolors='none',          # Make the center transparent/hollow
            edgecolors=color_map[g],    # Use group color for the outline
            linewidths=0.8,             # Define the thickness of the outline
            label=g, 
            depthshade=True
        )
    ax_3d.invert_xaxis()
    # Set 3D Axes Labels (reflecting the swap)
    ax_3d.set_ylabel("", labelpad=34)
    ax_3d.set_xlabel("", labelpad=34)
    ax_3d.zaxis.set_rotate_label(False)
    ax_3d.yaxis.set_rotate_label(False)
    ax_3d.xaxis.set_rotate_label(False)

    fig_3d.text(
        0.92, 
        0.3,  # <--- Adjust this 0.2 to move label "to the side", 
        "Tau score", 
        fontsize=12, 
        fontweight='normal',
        horizontalalignment='center'         # Ensures the text is centered on the X-coordinate
    )

    # --- Y-Axis Label ("Integral violation") ---
    # Placed at Y-center.
    # To move it "to the side", we offset it along the X-axis.
    # Changing 'x_max + (dx * 0.2)' moves it further left/right.
    fig_3d.text(
        0.43, # <--- Adjust this 0.2 to move label "to the side"
        0.06 ,
        "Amyloid-beta centiloids",
        fontsize=12, 
        fontweight='normal',
        horizontalalignment='center'         # Ensures the text is centered on the X-coordinate
    )
    
    ax_3d.set_zlabel("", labelpad=85)
    fig_3d.text(
    0.10,                                # X-coordinate (Adjusted to be visually centered over the Z-axis)
    0.80,                                # Y-coordinate (Above the plot area)
    "Integral violation",  # Text to display
    fontsize=12, 
    fontweight='normal',
    horizontalalignment='center'         # Ensures the text is centered on the X-coordinate
)
    
    # Clean up 3D plot aesthetics
    ax_3d.xaxis.pane.set_visible(False)
    ax_3d.yaxis.pane.set_visible(False)
    ax_3d.zaxis.pane.set_visible(False)
    ax_3d.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    ax_3d.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    ax_3d.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    ax_3d.legend(frameon=True) 
    
    plt.tight_layout()
    plt.show()

    # #######################################################
    # 2. ABETA Density Plot
    # #######################################################
    print("Generating ABeta Density Plot...")
    fig_ab, ax_ab = plt.subplots(figsize=(6, 4))
    
    for g in groups:
        ab = group_means.loc[g, "ABeta"]
        kde_x = gaussian_kde(ab)
        density_x = kde_x(x_lin)
        ax_ab.plot(x_lin, density_x, color=color_map[g], lw=2, label=g)

    ax_ab.axhline(0, color='black', linewidth=1.2)
    ax_ab.set_title("Local ABeta SUVR Density")
    ax_ab.set_xlabel("Local ABeta SUVR")
    ax_ab.set_ylabel("Density (KDE)")
    #ax_ab.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # #######################################################
    # 3. TAU Density Plot
    # #######################################################
    print("Generating Tau Density Plot...")
    fig_tau, ax_tau = plt.subplots(figsize=(6, 4))
    
    for g in groups:
        ta = group_means.loc[g, "Tau"]
        kde_y = gaussian_kde(ta)
        density_y = kde_y(y_lin)
        ax_tau.plot(y_lin, density_y, color=color_map[g], lw=2, label=g)

    ax_tau.axhline(0, color='black', linewidth=1.2)
    ax_tau.set_title("Local Tau SUVR Density")
    ax_tau.set_xlabel("Local Tau SUVR")
    ax_tau.set_ylabel("Density (KDE)")
    #ax_tau.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # #######################################################
    # 4. I_NORM2 Density Plot
    # #######################################################
    print("Generating I_norm2 Density Plot...")
    fig_i, ax_i = plt.subplots(figsize=(6, 4))
    
    for g in groups:
        ii = group_means.loc[g, "I_norm2"]
        kde_z = gaussian_kde(ii)
        density_z = kde_z(z_lin)
        ax_i.plot(z_lin, density_z, color=color_map[g], lw=2, label=g)

    ax_i.axhline(0, color='black', linewidth=1.2)
    ax_i.set_title("I Density")
    ax_i.set_xlabel("I ")
    ax_i.set_ylabel("Density (KDE)")
    #ax_i.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    print("\nAll four plots generated successfully.")

def scatter_plot_3d_2(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import gaussian_kde

    # ---------------------------------------------------------
    # 1. Data Prep
    # ---------------------------------------------------------
    group_means = pd.DataFrame({
        "ABeta":   df.groupby("group")["ABeta"].apply(lambda x: np.vstack(x).mean(axis=0)),
        "Tau":     df.groupby("group")["Tau"].apply(lambda x: np.vstack(x).mean(axis=0)),
        "I_norm2": df.groupby("group")["I_norm2"].apply(lambda x: np.vstack(x).mean(axis=0)),
    })
    groups = group_means.index.tolist()
    
    color_map = {
        "HC": "#3AAA35", 
        "MCI(AB-)": "#3478F6", 
        "MCI(AB+)": "#0EC9CC", 
        "AD": "#E32626"
    }

    # Data ranges - MAPPING: X=Tau, Y=ABeta, Z=I_norm2
    x_vals = np.concatenate(group_means["Tau"].values)      # X-axis
    y_vals = np.concatenate(group_means["ABeta"].values)    # Y-axis
    z_vals = np.concatenate(group_means["I_norm2"].values)  # Z-axis

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    
    # Calculate ranges
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min
    
    # Linspaces for smooth density curves
    x_lin = np.linspace(x_min, x_max, 200)
    y_lin = np.linspace(y_min, y_max, 200)
    z_lin = np.linspace(z_min, z_max, 200)

    # ---------------------------------------------------------
    # 2. Figure Setup
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # View angle (matching your reference)
    ax.view_init(elev=17, azim=42)

    # ---------------------------------------------------------
    # 3. Main Scatter Plot
    # ---------------------------------------------------------
    for g in groups:
        ta = group_means.loc[g, "Tau"]
        ab = group_means.loc[g, "ABeta"]
        ii = group_means.loc[g, "I_norm2"]
        
        ax.scatter(
            ta,  # X-axis: Tau
            ab,  # Y-axis: ABeta
            ii,  # Z-axis: I_norm2
            s=18, 
            alpha=0.7, 
            facecolors='none',
            edgecolors=color_map[g],
            linewidths=0.8,
            label=g, 
            depthshade=True
        )

    # Invert X-axis (as in your original)
    ax.invert_xaxis()

    # ---------------------------------------------------------
    # 4. Projection Curves (Fixed Positioning)
    # ---------------------------------------------------------
    
    # Control parameters
    curve_scale = 0.15  # Height of curves as fraction of axis range
    offset = 0.05       # Gap between axis and curve
    
    for g in groups:
        c = color_map[g]

        # --- LEFT PROJECTION: Z-axis (I_norm2 density) ---
        # Position: Along Z-axis, on the LEFT SIDE (Y=y_min, X pushed left)
        ii = group_means.loc[g, "I_norm2"]
        kde_z = gaussian_kde(ii)
        dens_z = kde_z(z_lin)
        dens_z_scaled = (dens_z / dens_z.max()) * dy * curve_scale
        
        # After X-axis inversion: x_max is visually on the left
        # So we offset from x_max to push it further left
        x_proj_left = x_max + dx * offset + dens_z_scaled
        y_proj_left = np.full_like(z_lin, y_min)
        
        ax.plot(x_proj_left, y_proj_left, z_lin, 
               color=c, linewidth=2, alpha=0.8)

        # --- BOTTOM-FRONT PROJECTION: X-axis (Tau density) ---
        # Position: Along X-axis, BELOW floor (Z pushed down, Y=y_min front)
        ta = group_means.loc[g, "Tau"]
        kde_x = gaussian_kde(ta)
        dens_x = kde_x(x_lin)
        dens_x_scaled = (dens_x / dens_x.max()) * dz * curve_scale
        
        y_proj_front = np.full_like(x_lin, y_min)
        z_proj_down = z_min - dz * offset - dens_x_scaled
        
        ax.plot(x_lin, y_proj_front, z_proj_down, 
               color=c, linewidth=2, alpha=0.8)

        # --- BOTTOM-RIGHT PROJECTION: Y-axis (ABeta density) ---
        # Position: Along Y-axis, BELOW floor (Z pushed down, X=x_min right side)
        ab = group_means.loc[g, "ABeta"]
        kde_y = gaussian_kde(ab)
        dens_y = kde_y(y_lin)
        dens_y_scaled = (dens_y / dens_y.max()) * dz * curve_scale
        
        # After inversion, x_min is on the visual right
        x_proj_right = np.full_like(y_lin, x_min)
        z_proj_down_y = z_min - dz * offset - dens_y_scaled
        
        ax.plot(x_proj_right, y_lin, z_proj_down_y, 
               color=c, linewidth=2, alpha=0.8)

    # ---------------------------------------------------------
    # 5. Set Axis Limits (with padding for projections)
    # ---------------------------------------------------------
    
    total_pad = offset + curve_scale
    
    ax.set_xlim(x_min, x_max + dx * total_pad)  # Extra space on left (visual)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min - dz * total_pad, z_max)  # Extra space below

    # ---------------------------------------------------------
    # 6. Labels and Styling
    # ---------------------------------------------------------
    
    # Clear default labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    
    # Disable label rotation
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    
    # Custom text labels (using figure coordinates for precise placement)
    fig.text(0.92, 0.30, "Tau score", 
            fontsize=12, ha='center', rotation=0)
    
    fig.text(0.43, 0.06, "Integral violation", 
            fontsize=12, ha='center', rotation=0)
    
    fig.text(0.10, 0.80, "Amyloid-beta centiloids", 
            fontsize=12, ha='center', rotation=0)
    
    # Clean aesthetics
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    ax.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    ax.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    
    # Add (A) label
    fig.text(0.02, 0.98, '(A)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 7. Separate Density Plots
    # ---------------------------------------------------------
    print("Generating separate density plots...")
    
    fig_dens, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # ABeta density
    for g in groups:
        ab = group_means.loc[g, "ABeta"]
        kde = gaussian_kde(ab)
        density = kde(y_lin)
        axes[0].plot(y_lin, density, color=color_map[g], lw=2, label=g)
    axes[0].axhline(0, color='black', linewidth=1.2)
    axes[0].set_title("Local ABeta SUVR Density")
    axes[0].set_xlabel("Local ABeta SUVR")
    axes[0].set_ylabel("Density (KDE)")
    axes[0].legend(frameon=True)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Tau density
    for g in groups:
        ta = group_means.loc[g, "Tau"]
        kde = gaussian_kde(ta)
        density = kde(x_lin)
        axes[1].plot(x_lin, density, color=color_map[g], lw=2, label=g)
    axes[1].axhline(0, color='black', linewidth=1.2)
    axes[1].set_title("Local Tau SUVR Density")
    axes[1].set_xlabel("Local Tau SUVR")
    axes[1].set_ylabel("Density (KDE)")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # I_norm2 density
    for g in groups:
        ii = group_means.loc[g, "I_norm2"]
        kde = gaussian_kde(ii)
        density = kde(z_lin)
        axes[2].plot(z_lin, density, color=color_map[g], lw=2, label=g)
    axes[2].axhline(0, color='black', linewidth=1.2)
    axes[2].set_title("Integral Violation Density")
    axes[2].set_xlabel("Integral Violation")
    axes[2].set_ylabel("Density (KDE)")
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAll plots generated successfully.")

def radarplot(rsn_means_df: pd.DataFrame, rsn_names: list):
    import plotly.graph_objects as go
    
    # --- 1. Extract and Order Data ---
    
    def get_ordered_means(group_label):
    # Function to extract ordered means for a specific group
        group_data = rsn_means_df.query(f"Group == '{group_label}'")
            
            # 2. Re-aggregate to ensure only one row per RSN
        aggregated_data = group_data.groupby('RSN')['Value'].mean().reset_index()

            # 3. Set index and order the values
        ordered_values = aggregated_data.set_index('RSN').reindex(rsn_names)['Value'].tolist()
        return ordered_values

    ad_values = get_ordered_means('AD')
    hc_values = get_ordered_means('HC')
    # --- 2. Create Plotly Figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=hc_values,
            theta=rsn_names,
            fill='toself',
            name='HC',
            line=dict(color='#8BC34A'),
        ))
    # Add the first radar plot trace for AD
    fig.add_trace(go.Scatterpolar(
        r=ad_values,
        theta=rsn_names,
        fill='toself',
        name='AD',
        line=dict(color='#1ABC9C'),
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        # === Tick spacing ===
        dtick=0.1,          # 4 tick steps → gives 5 ticks (0, .25, .50, .75, 1.0)
        ticklabelstep=1,     # label every tick → 4 "value labels" between min & max

        # === Tick styling ===
        tickfont=dict(size=18, color="black", family="Arial"),

        # === Optional: explicitly set axis range (helps control ticks) ===
        range=[0, 0.9],

        showline=True,
        gridcolor="lightgray",
        ),

        angularaxis=dict(
            tickfont=dict(
                size=18,
                color="black",
                family="Arial"
            ),
        )
    ),
    showlegend=True,
    title="RSN Mean Values Comparison (AD vs HC)",
    legend=dict(
        font=dict(
            size=18,
            color="black",
            family="Arial"
        )
    )
)
    fig.show()

# --- Example of how to call the function ---
# Assuming 'long_df' contains your RSN means (Group, RSN, Value)
# and RSN_LIST contains your RSN names in the desired order:

# radarplot_from_means(long_df, RSN_LIST)

# Example Parameters (Adjust to your actual data)
RSN_ORDER_val = ['Vis','SalVentAttn', 'SomMot', 'DorsAttn', 'Limbic', 'Cont', 'Def']
GROUPS_ORDER_val = ['HC', 'AD']


DL_type = 'DL_B2'
model_type1 = 'modelfree'  # 'modelfree' or 'modelbased'
model_type2 = 'modelbased'
NPARCELLS = 400 # max 379 for DL_A, max 400 for DL_B
fit_sigma = True
fit_a = True


repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "FDT_DATA")
save_path_plot = os.path.join(repo_root, "data", "RESULT_PLOTS")
filename_based = f"FDT_results_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
filename_free = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"
fdt_data_based = np.load(os.path.join(save_path, filename_based), allow_pickle=True)
df_based = pd.DataFrame({k: fdt_data_based[k].tolist() for k in fdt_data_based.files})
fdt_data_free = np.load(os.path.join(save_path, filename_free), allow_pickle=True)
df_free = pd.DataFrame({k: fdt_data_free[k].tolist() for k in fdt_data_free.files})

df_based = loadProteins(df_based, DL_type, 'Amyloid', repo_root) #'Amyloid' or 'Tau'
df_based = loadProteins(df_based, DL_type, 'Tau', repo_root) #'Amyloid' or 'Tau'
df_free = loadProteins(df_free, DL_type, 'Amyloid', repo_root) #'Amyloid' or 'Tau'
df_free = loadProteins(df_free, DL_type, 'Tau', repo_root) #'

df_based['diff'] = [
    (np.array(X) + np.array(I) - 1).tolist()
    for X, I in zip(df_based['X_norm2'], df_based['I_norm2'])
]

# Calculate parcel-wise difference for df_free
df_free['diff'] = [
    (np.array(X) + np.array(I) - 1).tolist()
    for X, I in zip(df_free['X_norm2'], df_free['I_norm2'])
]
print([np.mean(diff) for diff in df_based['diff'][110:115]])
print([np.mean(diff) for diff in df_based['diff'][10:15]])


measure = 'I_norm2'


# #print(df.columns)
#parcel_comparison(df, 'I_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
#subject_comparison(df, 'I_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# parcel_comparison(df, 'X_norm2', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison(df_based, 'ABeta', 'modelbased', NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison(df_based, 'Tau', 'modelbased', NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
#print(np.max([np.max(arr) for arr in df['ABeta'].values if isinstance(arr, np.ndarray)]))
scatter_plot_3d(df_free)
# scatter_plot_3d_2(df_based)


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

add_parcel_info_to_df(df_based, NPARCELLS, os.path.join(repo_root, 'hyperparams.json'))
add_parcel_info_to_df(df_free, NPARCELLS,  os.path.join(repo_root, 'hyperparams.json'))



def aggregate_model_data(df, measure_col, group_col='group'):
    """
    Aggregate model data by group, handling array values.
    Returns a dict: {'HC': [scalar_values], 'AD': [scalar_values]}
    """
    grouped = {}
    
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        values = []
        
        for idx, row in group_df.iterrows():
            measure_val = row[measure_col]
            
            # If it's an array/list, take the mean
            if isinstance(measure_val, (list, np.ndarray)):
                values.append(np.nanmean(measure_val))
            else:
                # If it's already scalar, use it directly
                values.append(measure_val)
        
        grouped[group] = values
    
    return grouped

# Use it:
model_input = {
    'Modelfree': aggregate_model_data(df_free, measure),
    'Modelbased': aggregate_model_data(df_based, measure)
}


plot_multi_model_comparison_violin(
    model_data_dicts=model_input,
    groups_order=GROUPS_ORDER_val,
    measure_name='Integral Violation',
    y_label_override='Integral violation',
    test_type='ranksum',
    # save_path='./plots'  # Uncomment to save
)

df = df_free
plot_ready_df = aggregate_all_rsns_for_plotting(
    df=df,
    measure=measure,
    rsn_names_list=RSN_ORDER_val
)
radarplot(plot_ready_df,RSN_ORDER_val)

plot_multi_rsn_split_violin(
    plot_df=plot_ready_df,
    rsn_order=RSN_ORDER_val,
    groups_order=GROUPS_ORDER_val,
    measure_name=measure,
    y_label_override='Integral violation',
    test_type='ranksum',
)


plot_ready_df_rsn = aggregate_all_rsns_for_plotting_parcel_avg(
    df=df,  # or df_based
    measure=measure,
    rsn_names_list=RSN_ORDER_val
)


plot_multi_rsn_split_violin(
    plot_df=plot_ready_df_rsn,
    rsn_order=RSN_ORDER_val,
    groups_order=GROUPS_ORDER_val,
    measure_name='Integral Violation',
    y_label_override='Integral violation',
    test_type='permutation',
    n_permutations=10000,
)

# 2. Plot for 2 Models
model_input = {
    'Model-free': df_free,
    'Model-based': df_based
}

plot_multi_model_comparison_violin(
    model_data_dicts=model_input,
    groups_order=GROUPS_ORDER_val,
    measure_name='Integral Violation',
    y_label_override='Integral violation',
    save_path='./plots',
    test_type='permutation',
    n_permutations=10000,
    measure_col_name=measure
)

# subject_comparison_rsn(df, measure, 'All', NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'SomMot', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'DorsAttn', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'SalVentAttn', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'Limbic', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'Cont', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'Def', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)

