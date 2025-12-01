import os
from matplotlib.ticker import FixedLocator, MaxNLocator
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

def plot_violin_groups_with_significance(
    df_data,
    measure_col_name: str,
    group_col_name: str = 'group',
    comparisons: list = None,
    measure_display_name: str = None,
    y_label_override: str = None,
    palette: dict = None,
    dpi: int = 300,
    save_path: str = None,
    save_prefix: str = 'GroupViolinComparison'
):
    """
    Single violin plot with one instance per group and pairwise Wilcoxon rank-sum
    significance bars with stars.
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, mannwhitneyu
    import os

    # -------------------------
    # Defaults
    # -------------------------
    if comparisons is None:
        comparisons = [('HC', 'MCI(AB+)'), ('HC', 'AD'), ('MCI(AB+)', 'AD')]

    if measure_display_name is None:
        measure_display_name = measure_col_name

    if palette is None:
        palette = {
            'HC': '#8BC34A',
            'MCI(AB+)': "#c34023",
            'AD': '#1ABC9C'
        }

    # -------------------------
    # Subject-level averaging
    # -------------------------
    
    df = df_data.copy()
    groups_to_keep = ["HC", "MCI(AB+)", "AD"]
    df = df[df['group'].isin(groups_to_keep)].copy() 
    sample_value = df[measure_col_name].iloc[0]

    if isinstance(sample_value, (list, np.ndarray)):
        df[measure_col_name] = df[measure_col_name].apply(
            lambda x: np.nanmean(np.array(x).flatten())
        )

    df = df[[group_col_name, measure_col_name]].dropna()
    df.columns = ['Group', 'Value']
    df['Group'] = df['Group'].astype(str)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna()

    group_order = df['Group'].unique().tolist()

    print("\nGroups used:", group_order)

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_context('notebook', font_scale=1.0)
    group_counts = df.groupby('Group')['Value'].count()
    
    # 2. Create the new labels (e.g., "HC (N=50)")
    new_xlabels = [
        f"{group}\n(N={group_counts[group]})"
        for group in group_order
    ]

    sns.violinplot(
        data=df,
        x='Group',
        y='Value',
        hue='Group',
        order=group_order,
        hue_order=group_order,
        palette=palette,
        inner='quartile',
        linewidth=0.8,
        ax=ax,
        legend=False
    )

    sns.stripplot(
        data=df,
        x='Group',
        y='Value',
        hue='Group',
        order=group_order,
        edgecolor='black',
        s=3,
        alpha=0.6,
        jitter=0.2,
        linewidth=0.7,
        palette=palette,
        dodge=False,
        ax=ax,
        legend=False
    )
    ax.set_xticklabels(new_xlabels, fontsize=14)
    ax.tick_params(axis='y', length=0) 
    ax.tick_params(axis='x', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ylabel = y_label_override if y_label_override else f'{measure_display_name}'
    # from matplotlib.lines import Line2D
    # legend_handles = [
    #     Line2D([0], [0], marker='o', color='w', 
    #            markerfacecolor=palette[group], 
    #            markersize=10, 
    #            markeredgecolor='black', 
    #            markeredgewidth=0.7)
    #     for group in group_order
    # ]
    # legend_labels = group_order # Use the original group names for the legend

    # ax.legend(legend_handles, legend_labels, 
    #           loc='upper right', title=None, frameon=True)

    ax.set_title("", fontsize=14, pad=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('', fontsize=12)

    

    y_max = df['Value'].max()
    y_min = df['Value'].min()
    y_range = y_max - y_min
    base_height = y_max + 0.08 * y_range

    for i, (g1, g2) in enumerate(comparisons):

        vals1 = df[df['Group'] == g1]['Value'].values
        vals2 = df[df['Group'] == g2]['Value'].values

        stat, p_val = ranksums(vals1, vals2)
        u_stat, _ = mannwhitneyu(vals1, vals2, alternative='two-sided')

        n1, n2 = len(vals1), len(vals2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        sig = '***' if p_val < 0.001 else \
              '**' if p_val < 0.01 else \
              '*' if p_val < 0.05 else ''

        x1 = group_order.index(g1)
        x2 = group_order.index(g2)

        y = base_height + i * 0.08 * y_range

        if sig is not '':
            plt.plot([x1, x1, x2, x2],
                 [y, y + 0.02*y_range, y + 0.02*y_range, y],
                 lw=1.2, color='black')

        plt.text((x1 + x2) / 2,
                 y + 0.025*y_range,
                 sig,
                 ha='center',
                 va='bottom',
                 fontsize=13)

    # -------------------------
    # Save
    # -------------------------
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, f"{save_prefix}_{measure_col_name}.png")
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
        print(f"\nSaved to: {fname}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def plot_violin_groups_with_significance_parcel_mean(
    df_data,
    measure_col_name: str,
    group_col_name: str = 'group',
    parcel_col_name: str = 'parcel',   # <<< REQUIRED
    comparisons: list = None,
    measure_display_name: str = None,
    y_label_override: str = None,
    palette: dict = None,
    dpi: int = 300,
    save_path: str = None,
    save_prefix: str = 'GroupViolinComparison_PARCEL_MEAN',
    n_permutations: int = 10000
):
    """
    Parcel-MEAN violin plot with permutation-test significance bars.
    Exactly 1 value per parcel per group (e.g. 400 values per group).
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # -------------------------
    # Defaults
    # -------------------------
    if comparisons is None:
        comparisons = [('HC', 'MCI(AB+)'), ('HC', 'AD'), ('MCI(AB+)', 'AD')]

    if measure_display_name is None:
        measure_display_name = measure_col_name

    if palette is None:
        palette = {
            'HC': '#8BC34A',
            'MCI(AB+)': "#c34023",
            'AD': '#1ABC9C'
        }

    # -------------------------
    # Group filtering
    # -------------------------
    df = df_data.copy()
    groups_to_keep = ["HC", "MCI(AB+)", "AD"]
    df = df[df[group_col_name].isin(groups_to_keep)].copy()

    sample_value = df[measure_col_name].iloc[0]

    if isinstance(sample_value, (list, np.ndarray)):
        # Create parcel index BEFORE exploding
        df = df.copy()
        df['parcel'] = df[measure_col_name].apply(lambda x: list(range(len(x))))
        df = df.explode([measure_col_name, 'parcel'])

    df[measure_col_name] = pd.to_numeric(df[measure_col_name], errors='coerce')
    df = df.dropna(subset=[measure_col_name])

    df_parcel_mean = (
        df
        .groupby([group_col_name, parcel_col_name])[measure_col_name]
        .mean()
        .reset_index()
    )

    df_parcel_mean.columns = ['Group', 'Parcel', 'Value']

    #group_order = df_parcel_mean['Group'].unique().tolist()
    group_order = [g for g in groups_to_keep if g in df[group_col_name].unique()]


    print("\nGroups used (PARCEL MEAN LEVEL):", group_order)
    print("\nParcels per group:")
    print(df_parcel_mean.groupby('Group')['Value'].count())

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_context('notebook', font_scale=1.0)

    new_xlabels = [
        f"{group}\n(N={df_parcel_mean[df_parcel_mean['Group'] == group]['Value'].shape[0]})"
        for group in group_order
    ]

    sns.violinplot(
        data=df_parcel_mean,
        x='Group',
        y='Value',
        hue='Group',
        order=group_order,
        hue_order=group_order,
        palette=palette,
        inner='quartile',
        linewidth=0.8,
        ax=ax,
        legend=False
    )

    sns.stripplot(
        data=df_parcel_mean,
        x='Group',
        y='Value',
        hue='Group',
        order=group_order,
        edgecolor='black',
        s=3,
        alpha=0.6,
        jitter=0.2,
        linewidth=0.7,
        palette=palette,
        dodge=False,
        ax=ax,
        legend=False
    )

    ax.set_xticklabels(new_xlabels, fontsize=14)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ylabel = y_label_override if y_label_override else f'{measure_display_name}'
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('', fontsize=12)
    ax.set_title("", fontsize=14, pad=15)

    y_max = df_parcel_mean['Value'].max()
    y_min = df_parcel_mean['Value'].min()
    y_range = y_max - y_min
    base_height = y_max + 0.08 * y_range

    print("\n=== Permutation Test Results (PARCEL MEANS) ===")

    for i, (g1, g2) in enumerate(comparisons):

        vals1 = df_parcel_mean[df_parcel_mean['Group'] == g1]['Value'].values
        vals2 = df_parcel_mean[df_parcel_mean['Group'] == g2]['Value'].values

        p_val = permutation_test(
            vals1,
            vals2,
            n_permutations=n_permutations
        )

        print(f"{g1} vs {g2}: p={p_val:.5f} "
              f"(n1={len(vals1)}, n2={len(vals2)}, {n_permutations} permutations)")

        sig = '***' if p_val < 0.001 else \
              '**' if p_val < 0.01 else \
              '*' if p_val < 0.05 else ''

        x1 = group_order.index(g1)
        x2 = group_order.index(g2)

        y = base_height + i * 0.08 * y_range

        if sig != '':
            ax.plot(
                [x1, x1, x2, x2],
                [y, y + 0.02*y_range, y + 0.02*y_range, y],
                lw=1.2,
                color='black'
            )

        ax.text(
            (x1 + x2) / 2,
            y + 0.025 * y_range,
            sig,
            ha='center',
            va='bottom',
            fontsize=13
        )

    # -------------------------
    # Save
    # -------------------------
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, f"{save_prefix}_{measure_col_name}.png")
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
        print(f"\nSaved to: {fname}")

    plt.tight_layout()
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
    parcel_rsn_map = df['parcel_RSNs'].iloc[0]

    rsn_order = plot_df['RSN'].unique().tolist()
    rsn_parcel_counts = {}
    for rsn_name in rsn_order:
        count = parcel_rsn_map.count(rsn_name)
        rsn_parcel_counts[rsn_name] = count
    new_xlabels = [
        f"{rsn}\n(N={rsn_parcel_counts[rsn]})"
        for rsn in rsn_order
    ]
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
    ax.set_xticklabels(new_xlabels)
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
    groups_to_keep = ["HC", "MCI(AB+)", "AD"]
    df = df[df['group'].isin(groups_to_keep)].copy() 
    
    group_means = pd.DataFrame({
        "ABeta":   df.groupby("group")["ABeta"].apply(lambda x: np.vstack(x).mean(axis=0)),
        "Tau":     df.groupby("group")["Tau"].apply(lambda x: np.vstack(x).mean(axis=0)),
        "I_norm2": df.groupby("group")["I_norm2"].apply(lambda x: np.vstack(x).mean(axis=0)),
    })
    # group_means = pd.DataFrame({
    #     "ABeta":   df.groupby("group")["ABeta"].apply(lambda x: np.vstack(x)),
    #     "Tau":     df.groupby("group")["Tau"].apply(lambda x: np.vstack(x)),
    #     "I_norm2": df.groupby("group")["I_norm2"].apply(lambda x: np.vstack(x)),
    # })
    groups = ["HC", "MCI(AB+)", "AD"]  # Fixed order for legend
    
    color_map = {
        "HC":       '#8BC34A',   
        "MCI(AB+)": "#c34023",   
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
    
    # Lower viewing angle
    ax_3d.view_init(elev=12, azim=42)
    
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
        0.53, # <--- Adjust this 0.2 to move label "to the side"
        0.06 ,
        "ABeta",
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
    
    # Create legend with fixed order and position inside the figure
    handles, labels = ax_3d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_3d.legend([by_label[g] for g in groups], groups, loc='upper right', frameon=True, bbox_to_anchor=(0.98, 0.98))
    
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
    
    fig.text(0.10, 0.90, "ABeta", 
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
    mci_values = get_ordered_means('MCI(AB+)')
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
    fig.add_trace(go.Scatterpolar(
        r=mci_values,
        theta=rsn_names,
        fill='toself',
        name='MCI(AB+)',
        line=dict(color="#c34023"),
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
            dtick=40,          
            ticklabelstep=1,    

            # === Tick styling (Made Bold) ===
            # Note: We append "Bold" or choose a heavy font variant for reliability
            tickfont=dict(size=18, color="black", family="Arial Bold, Arial Black, sans-serif"),
            
            # === Optional: explicitly set axis range (helps control ticks) ===
            range=[0, 0.07],

            showline=True,
            gridcolor="lightgray",
        ),

        angularaxis=dict(
            tickfont=dict(
                size=22,
                color="black",
                # === Angular Labels Made Bold ===
                family="Arial Bold, Arial Black, sans-serif"
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

def brain_map_parcel_average(df, measure: str, group: str, 
                              vmin=None, vmax=None, 
                              cmap='viridis', 
                              save_path=None, 
                              title_suffix=""):
    """
    Visualizes average parcel values for a specific measure and group on brain surface.
    
    Args:
        df: DataFrame with columns ['group', measure, 'parcel_RSNs'] where measure 
            contains arrays of parcel values
        measure: Column name (e.g., 'I_norm2', 'X_norm2', 'Tau', 'ABeta')
        group: Group label (e.g., 'HC', 'AD', 'MCI(AB+)', 'MCI(AB-)')
        vmin, vmax: Colorbar limits (auto-computed if None)
        cmap: Colormap name
        save_path: Path to save figure (optional)
        title_suffix: Additional text for title
    """
    from nilearn import surface, datasets, plotting
    import nibabel as nib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from nilearn.datasets import load_fsaverage
    from nilearn.surface import SurfaceImage
    from nilearn.datasets import load_fsaverage_data

    group_df = df[df['group'] == group]
    
    if group_df.empty:
        raise ValueError(f"No data found for group: {group}")
    
    # Stack all subjects' parcel arrays and average across subjects
    all_parcels = np.vstack(group_df[measure].values)  # Shape: (n_subjects, n_parcels)
    parcel_averages = np.nanmean(all_parcels, axis=0)  # Shape: (n_parcels,)
    
    NPARCELS = len(parcel_averages)
    print(f"Group: {group}, Measure: {measure}, N_parcels: {NPARCELS}")
    print(f"Value range: [{np.min(parcel_averages):.3f}, {np.max(parcel_averages):.3f}]")
    
    # ---------------------------------------------------------
    # 2. Load Parcellation and Map Values to 3D Volume
    # ---------------------------------------------------------
    nii_path = os.path.join('data/ADNI-B_DATA/N238rev', 
                           'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Parcellation file not found: {nii_path}")
    
    parcel_img = nib.load(nii_path)
    parcel_data = parcel_img.get_fdata()
    
    # Create empty 3D map
    group_map = np.zeros_like(parcel_data)
    
    for i in range(NPARCELS):
    
        nifti_label = i + 1
        average_value = parcel_averages[i]
        group_map[parcel_data == nifti_label] = average_value
    
    group_img = nib.Nifti1Image(group_map, affine=parcel_img.affine)
    
    print("Projecting to surface...")
    fsaverage = datasets.fetch_surf_fsaverage()
    fsaverage_meshes = load_fsaverage()
    surface_image = SurfaceImage.from_volume(
    mesh=fsaverage_meshes["pial"],
    volume_img=group_img,
    )
    curv_sign = load_fsaverage_data(data_type="curvature")
    for hemi, data in curv_sign.data.parts.items():
        curv_sign.data.parts[hemi] = np.sign(data)

    texture_left = surface.vol_to_surf(group_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(group_img, fsaverage.pial_right)

    if vmin is None:
        vmin = np.min(parcel_averages)
    if vmax is None:
        vmax = np.max(parcel_averages)
    
    print(f"Colorbar range: [{vmin:.3f}, {vmax:.3f}]")

    fig = plt.figure(figsize=(20, 5))
    
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes['pial'],  
        hemi='both',
        view='posterior',
        colorbar=False,  # Show colorbar on the last panel
        cmap=cmap,
        bg_map=curv_sign,
        vmin=vmin,
        vmax=vmax,
        axes=ax1,
        darkness=0.5,
        title=f'(Posterior)'
    )
    # Left hemisphere
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage.pial_left, 
        texture_left,
        hemi='left', 
        title=f'{group} - {measure} (Left){title_suffix}',
        view='lateral',
        colorbar=False, 
        cmap=cmap,
        bg_map=fsaverage.sulc_left,
        vmin=vmin, 
        vmax=vmax,
        axes=ax2, 
        darkness=0.5
    )

    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    plotting.plot_surf_stat_map(
    stat_map=surface_image,
    surf_mesh=fsaverage_meshes['pial'],          # Pass the volumetric NIfTI, not textures
    hemi='both',          # Nilearn will handle left + right
    view='anterior',      # or 'posterior'
    colorbar=False,
    cmap=cmap,
    bg_map=curv_sign,  # Nilearn handles right automatically
    vmin=vmin,
    vmax=vmax,
    axes=ax3,
    darkness=0.5,
    title=f'(Anterior)'
    )
    
    # Right hemisphere
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage.pial_right, 
        texture_right,
        hemi='right', 
        title=f'(Right){title_suffix}',
        view='lateral',
        colorbar=False, 
        cmap=cmap,
        bg_map=fsaverage.sulc_right,
        vmin=vmin, 
        vmax=vmax,
        axes=ax4, 
        darkness=0.5
    )

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Position: [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.09, 0.15, 0.86, 0.05]) 
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal') 
    cbar.set_label('Amyloid-beta CL', fontsize=22, labelpad=10, fontweight='bold')
    cbar.locator = MaxNLocator(nbins=5)  
    cbar.ax.tick_params(labelsize=17, labelcolor='black')  # fontweight is not set here
    for tick in cbar.ax.get_xticklabels():  # loop through tick labels
        tick.set_fontweight('bold')
    cbar.update_ticks()

    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig_name = f"brainmap_{group}_{measure}.png"
        full_path = os.path.join(save_path, fig_name)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {full_path}")
    
    plt.show()
    plt.close(fig)

def brain_map_difference(df, measure: str, group1: str, group2: str,
                        vmin=None, vmax=None, 
                        cmap='RdBu_r', 
                        save_path=None):
    """
    Visualizes the difference between two groups (group1 - group2).
    
    Useful for showing AD vs HC differences, for example.
    """
    from nilearn import surface, datasets, plotting
    import nibabel as nib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from nilearn.datasets import load_fsaverage
    from nilearn.surface import SurfaceImage
    from nilearn.datasets import load_fsaverage_data
    
    # Get both groups
    df1 = df[df['group'] == group1]
    df2 = df[df['group'] == group2]
    
    if df1.empty or df2.empty:
        raise ValueError(f"Missing data for {group1} or {group2}")
    
    # Compute averages
    parcels1 = np.nanmean(np.vstack(df1[measure].values), axis=0)
    parcels2 = np.nanmean(np.vstack(df2[measure].values), axis=0)
    
    # Difference
    parcel_diff = np.abs(parcels1 - parcels2)
    max_diff = np.max(parcel_diff)
    parcel_diff = parcel_diff / max_diff
    NPARCELS = len(parcel_diff)
    
    print(f"Difference: {group1} - {group2}")
    print(f"Range: [{np.min(parcel_diff):.3f}, {np.max(parcel_diff):.3f}]")
    
    # Map to volume
    nii_path = os.path.join('data/ADNI-B_DATA/N238rev', 
                           'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    parcel_img = nib.load(nii_path)
    parcel_data = parcel_img.get_fdata()
    
    diff_map = np.zeros_like(parcel_data)
    for i in range(NPARCELS):
    
        nifti_label = i + 1
        average_value = parcel_diff[i]
        diff_map[parcel_data == nifti_label] = average_value
    
    group_img = nib.Nifti1Image(diff_map, affine=parcel_img.affine)
    
    print("Projecting to surface...")
    fsaverage = datasets.fetch_surf_fsaverage()
    fsaverage_meshes = load_fsaverage()
    surface_image = SurfaceImage.from_volume(
    mesh=fsaverage_meshes["pial"],
    volume_img=group_img,
    )
    curv_sign = load_fsaverage_data(data_type="curvature")
    for hemi, data in curv_sign.data.parts.items():
        curv_sign.data.parts[hemi] = np.sign(data)

    texture_left = surface.vol_to_surf(group_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(group_img, fsaverage.pial_right)
    
    # Symmetric colorbar around 0
    if vmin is None:
        vmin = np.min(parcel_diff)
    if vmax is None:
        vmax = np.max(parcel_diff)
    
    # Plot
    fig = plt.figure(figsize=(20, 5))
    
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes['pial'],  
        hemi='both',
        view='posterior',
        colorbar=False,  # Show colorbar on the last panel
        cmap=cmap,
        bg_map=curv_sign,
        vmin=vmin,
        vmax=vmax,
        axes=ax1,
        darkness=0.5,
        title=f'(Posterior)'
    )
    # Left hemisphere
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage.pial_left, 
        texture_left,
        hemi='left', 
        title=f'{group1} - {group2} (left)',
        view='lateral',
        colorbar=False, 
        cmap=cmap,
        bg_map=fsaverage.sulc_left,
        vmin=vmin, 
        vmax=vmax,
        axes=ax2, 
        darkness=0.5
    )

    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    plotting.plot_surf_stat_map(
    stat_map=surface_image,
    surf_mesh=fsaverage_meshes['pial'],          # Pass the volumetric NIfTI, not textures
    hemi='both',          # Nilearn will handle left + right
    view='anterior',      # or 'posterior'
    colorbar=False,
    cmap=cmap,
    bg_map=curv_sign,  # Nilearn handles right automatically
    vmin=vmin,
    vmax=vmax,
    axes=ax3,
    darkness=0.5,
    title=f'(Anterior)'
    )
    
    # Right hemisphere
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage.pial_right, 
        texture_right,
        hemi='right', 
        title=f'{group1} - {group2} (Right)',
        view='lateral',
        colorbar=False, 
        cmap=cmap,
        bg_map=fsaverage.sulc_right,
        vmin=vmin, 
        vmax=vmax,
        axes=ax4, 
        darkness=0.5
    )

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Position: [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.5, 0.29, 0.015, 0.46]) 
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical') 
    cbar.set_label('', fontsize=22, labelpad=10, fontweight='bold')
    #cbar.locator = MaxNLocator(nbins=3)  
    cbar.locator = FixedLocator([0,1]) 
    cbar.ax.tick_params(labelsize=14, labelcolor='black',labelleft=True, labelright=False, left=True, right=False)  # fontweight is not set here
    for tick in cbar.ax.get_yticklabels():  # loop through tick labels
        tick.set_fontweight('bold')
    cbar.update_ticks()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_name = f"brainmap_diff_{group1}_vs_{group2}_{measure}.png"
        full_path = os.path.join(save_path, fig_name)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {full_path}")
    
    plt.show()
    plt.close(fig)

def brain_map_correlation(df, mode='between_groups', 
                          measure1=None, measure2=None,
                          group1=None, group2=None,
                          vmin=-1, vmax=1, 
                          cmap='RdBu_r', 
                          save_path=None,
                          atlas_path=None,
                          show_pvalues=True):
    from scipy.stats import pearsonr
    from nilearn import surface, datasets, plotting
    import nibabel as nib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FixedLocator
    from nilearn.datasets import load_fsaverage
    from nilearn.surface import SurfaceImage
    from nilearn.datasets import load_fsaverage_data
    
    # ==================== INPUT VALIDATION ====================
    if mode not in ['between_groups', 'between_measures']:
        raise ValueError("mode must be 'between_groups' or 'between_measures'")
    
    if mode == 'between_groups':
        if measure1 is None or group1 is None or group2 is None:
            raise ValueError("For 'between_groups' mode, provide: measure1, group1, group2")
        if measure2 is None:
            measure2 = measure1  # Same measure across groups
    
    if mode == 'between_measures':
        if measure1 is None or measure2 is None:
            raise ValueError("For 'between_measures' mode, provide: measure1, measure2")
    
    # Set default atlas path
    if atlas_path is None:
        atlas_path = os.path.join('data/ADNI-B_DATA/N238rev', 
                                  'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    
    # ==================== DATA EXTRACTION ====================
    print(f"\n{'='*60}")
    print(f"Computing Pearson Correlations - Mode: {mode}")
    print(f"{'='*60}")
    
    if mode == 'between_groups':
        # Get both groups
        df1 = df[df['group'] == group1].copy()
        df2 = df[df['group'] == group2].copy()
        
        if df1.empty or df2.empty:
            raise ValueError(f"Missing data for {group1} or {group2}")
        
        # Stack subjects × parcels
        data1 = np.vstack(df1[measure1].values)  # (n_subjects1, n_parcels)
        data2 = np.vstack(df2[measure2].values)  # (n_subjects2, n_parcels)
        
        print(f"Group 1 ({group1}): {data1.shape[0]} subjects")
        print(f"Group 2 ({group2}): {data2.shape[0]} subjects")
        print(f"Measure: {measure1}")
        
        # Need same number of subjects for correlation
        n_min = min(data1.shape[0], data2.shape[0])
        if data1.shape[0] != data2.shape[0]:
            print(f"\nWarning: Unequal group sizes. Using first {n_min} subjects from each.")
            data1 = data1[:n_min, :]
            data2 = data2[:n_min, :]
        
        title_str = f"{measure1}: {group1} vs {group2}"
        
    elif mode == 'between_measures':
        # Filter by group if specified
        if group1 is not None:
            df_filtered = df[df['group'] == group1].copy()
            title_str = f"{measure1} vs {measure2} ({group1})"
        else:
            groups_to_keep = ["HC", "MCI(AB+)", "AD"]
            df_filtered = df[df['group'].isin(groups_to_keep)].copy()
            title_str = f"{measure1} vs {measure2} (All subjects)"
        
        if df_filtered.empty:
            raise ValueError(f"No data available for the specified group")
        
        # Stack subjects × parcels
        data1 = np.vstack(df_filtered[measure1].values)
        data2 = np.vstack(df_filtered[measure2].values)
        
        print(f"Subjects: {data1.shape[0]}")
        print(f"Measures: {measure1} vs {measure2}")
        if group1 is not None:
            print(f"Group: {group1}")
    
    NPARCELS = data1.shape[1]
    print(f"Parcels: {NPARCELS}")
    
    # ==================== COMPUTE CORRELATIONS ====================
    print("\nComputing parcel-wise Pearson correlations...")
    
    correlations = np.zeros(NPARCELS)
    pvalues = np.zeros(NPARCELS)
    
    for p in range(NPARCELS):
        parcel1 = data1[:, p]
        parcel2 = data2[:, p]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(parcel1) | np.isnan(parcel2))
        
        if np.sum(valid_mask) > 3:  # Need at least 3 points for correlation
            r, pval = pearsonr(parcel1[valid_mask], parcel2[valid_mask])
            correlations[p] = r
            pvalues[p] = pval
        else:
            correlations[p] = np.nan
            pvalues[p] = np.nan
    
    # Print statistics
    valid_corrs = correlations[~np.isnan(correlations)]
    print(f"\nCorrelation Statistics:")
    print(f"  Mean: {np.mean(valid_corrs):.3f}")
    print(f"  Std:  {np.std(valid_corrs):.3f}")
    print(f"  Range: [{np.min(valid_corrs):.3f}, {np.max(valid_corrs):.3f}]")
    print(f"  Median: {np.median(valid_corrs):.3f}")
    
    # Count significant correlations
    sig_count = np.sum(pvalues < 0.05)
    sig_bonf = np.sum(pvalues < (0.05 / NPARCELS))  # Bonferroni
    print(f"\nSignificance:")
    print(f"  p < 0.05: {sig_count}/{NPARCELS} parcels ({100*sig_count/NPARCELS:.1f}%)")
    print(f"  p < {0.05/NPARCELS:.2e} (Bonferroni): {sig_bonf}/{NPARCELS} parcels")
    
    if show_pvalues:
        print("\nTop strongest correlations:")
        top_idx = np.argsort(np.abs(correlations))[::-1]
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. Parcel {idx}: r={correlations[idx]:.3f}, p={pvalues[idx]:.4f}")
    
    # ==================== MAP TO VOLUME ====================
    print("\nMapping to brain volume...")
    
    parcel_img = nib.load(atlas_path)
    parcel_data = parcel_img.get_fdata()
    
    corr_map = np.zeros_like(parcel_data)
    
    for i in range(NPARCELS):
        nifti_label = i + 1
        corr_value = correlations[i]
        
        # Set NaN correlations to 0 for visualization
        if np.isnan(corr_value):
            corr_value = 0.0
        
        corr_map[parcel_data == nifti_label] = corr_value
    
    corr_img = nib.Nifti1Image(corr_map, affine=parcel_img.affine)
    
    # ==================== PROJECT TO SURFACE ====================
    print("Projecting to surface...")
    
    fsaverage = datasets.fetch_surf_fsaverage()
    fsaverage_meshes = load_fsaverage()
    
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=corr_img,
    )
    
    curv_sign = load_fsaverage_data(data_type="curvature")
    for hemi, data in curv_sign.data.parts.items():
        curv_sign.data.parts[hemi] = np.sign(data)
    
    texture_left = surface.vol_to_surf(corr_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(corr_img, fsaverage.pial_right)
    
    # ==================== PLOTTING ====================
    print("Creating brain surface plots...")
    
    fig = plt.figure(figsize=(20, 5))
    
    # Posterior view
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes['pial'],
        hemi='both',
        view='posterior',
        colorbar=False,
        cmap=cmap,
        bg_map=curv_sign,
        vmin=vmin,
        vmax=vmax,
        axes=ax1,
        darkness=0.5,
        title='Posterior'
    )
    
    # Left lateral
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,
        texture_left,
        hemi='left',
        title='Left Lateral',
        view='lateral',
        colorbar=False,
        cmap=cmap,
        bg_map=fsaverage.sulc_left,
        vmin=vmin,
        vmax=vmax,
        axes=ax2,
        darkness=0.5
    )
    
    # Anterior view
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes['pial'],
        hemi='both',
        view='anterior',
        colorbar=False,
        cmap=cmap,
        bg_map=curv_sign,
        vmin=vmin,
        vmax=vmax,
        axes=ax3,
        darkness=0.5,
        title='Anterior'
    )
    
    # Right lateral
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage.pial_right,
        texture_right,
        hemi='right',
        title='Right Lateral',
        view='lateral',
        colorbar=False,
        cmap=cmap,
        bg_map=fsaverage.sulc_right,
        vmin=vmin,
        vmax=vmax,
        axes=ax4,
        darkness=0.5
    )
    
    # Colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.08, 0.15, 0.86, 0.05]) 
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal') 
    cbar.set_label('Correlation', fontsize=22, labelpad=10, fontweight='bold')
    # cbar.locator = MaxNLocator(nbins=5)  
    cbar.locator = FixedLocator([-1,0,1]) 
    cbar.ax.tick_params(labelsize=17, labelcolor='black')  # fontweight is not set here
    for tick in cbar.ax.get_xticklabels():  # loop through tick labels
        tick.set_fontweight('bold')
    cbar.update_ticks()
    
    cbar.update_ticks()
    
    # Overall title
    fig.suptitle(f'Parcel-wise Correlations: {title_str}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        if mode == 'between_groups':
            fig_name = f"brainmap_corr_{measure1}_{group1}_vs_{group2}.png"
        else:
            group_str = f"_{group1}" if group1 else "_all"
            fig_name = f"brainmap_corr_{measure1}_vs_{measure2}{group_str}.png"
        
        full_path = os.path.join(save_path, fig_name)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved to: {full_path}")
    
    plt.show()
    plt.close(fig)
    
    return correlations, pvalues, top_idx

def plot_correlation_histogram(correlations, pvalues, title='', save_path=None):
    """
    Plot histogram of correlation values with significance overlay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    valid_corrs = correlations[~np.isnan(correlations)]
    valid_pvals = pvalues[~np.isnan(pvalues)]
    
    # Histogram of correlations
    axes[0].hist(valid_corrs, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='r=0')
    axes[0].axvline(np.mean(valid_corrs), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean={np.mean(valid_corrs):.3f}')
    axes[0].set_xlabel('Correlation (r)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Distribution of Correlations\n{title}', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # P-value histogram
    axes[1].hist(valid_pvals, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    axes[1].set_xlabel('P-value', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of P-values', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    plt.show()
    # plt.close(fig)
def plot(df, group, measure, mode="both"):
    """
    mode:
        "single"     -> plot one random subject from the group
        "group_mean" -> plot the group average only
        "both"       -> plot both figures
    """

    if group is not None:
        df_group = df[df["group"] == group]
    else:
        df_group = df

    if len(df_group) == 0:
        raise ValueError(f"No subjects found for group '{group}'")

    # ============================
    # 1) SINGLE SUBJECT PLOT
    # ============================
    if mode in ["single", "both"]:

        # pick one random subject
        parcel_values = df_group[measure].sample(1).values[0]
        mean_value = np.mean(parcel_values)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            np.arange(len(parcel_values)),
            parcel_values,
            lw=1.2
        )

        ax.axhline(
            mean_value,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f"Mean: {mean_value:.2f}"
        )

        ax.set_xlabel("Parcel index", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"{measure} parameter", fontsize=11, fontweight='bold')

        ax.set_title("", fontsize=12, fontweight='bold', pad=10)

        ax.tick_params(axis='both', labelsize=9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # ✅ remove gap between y-axis and first x value
        ax.margins(x=0)

        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    # ============================
    # 2) GROUP AVERAGE PLOT
    # ============================
    if mode in ["group_mean", "both"]:

    # stack subject-level parcel arrays → (n_subjects, n_parcels)
        all_values = np.vstack(df_group[measure].values)

        subject_means = np.mean(all_values, axis=1) 
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            np.arange(len(subject_means)),
            subject_means,
            s=25,
            alpha=0.8
        )

        ax.set_xlabel("Subject index", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Mean {measure} per subject", fontsize=11, fontweight='bold')
        ax.set_title(f"Subject-wise Means ({group})", fontsize=12, fontweight='bold')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.margins(x=0)

        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(subject_means, bins=30, alpha=0.85)

        ax.set_xlabel(f"Mean {measure} per subject", fontsize=11, fontweight='bold')
        ax.set_ylabel("Number of subjects", fontsize=11, fontweight='bold')
        ax.set_title(f"Distribution of Subject Means ({group})", fontsize=12, fontweight='bold')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tight_layout()
        plt.show()





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

df_based = loadProteins(df_based, DL_type, 'Amyloid', repo_root,filt=False) #'Amyloid' or 'Tau'
df_based = loadProteins(df_based, DL_type, 'Tau', repo_root,filt=False) #'Amyloid' or 'Tau'
df_free = loadProteins(df_free, DL_type, 'Amyloid', repo_root,filt=True) #'Amyloid' or 'Tau'
df_free = loadProteins(df_free, DL_type, 'Tau', repo_root,filt=True) #'

df_based['diff'] = [
    (np.array(X) + np.array(I) - 1).tolist()
    for X, I in zip(df_based['X_norm2'], df_based['I_norm2'])
]
df_free['diff'] = [
    (np.array(X) + np.array(I) - 1).tolist()
    for X, I in zip(df_free['X_norm2'], df_free['I_norm2'])
]

# print([np.mean(diff) for diff in df_based['diff'][110:115]])
# print([np.mean(diff) for diff in df_based['diff'][10:15]])


measure = 'I_norm2'
df = df_based
# plot(df_based, group=None, measure="a", mode="both")
# plot(df_based, group=None, measure="sigma", mode="both")
# plot(df_based, group=None, measure="f_diff", mode="both")



# corrs, pvals, top = brain_map_correlation(
#     df, mode='between_measures',
#     measure1='I_norm2', measure2='Tau',
#     #group1='AD',
#     cmap='cividis',
# )
# corrs, pvals, top = brain_map_correlation(
#     df, mode='between_measures',
#     measure1='I_norm2', measure2='ABeta',
#     #group1='AD',
#     cmap='cividis',
# )
# plot_correlation_histogram(
#     corrs, pvals,
#     title='I_norm2 vs Tau',
#     save_path=os.path.join(save_path_plot, 'corr_hist_i_norm2_tau_ad.png')
# )
# subject_comparison(df_based, 'ABeta', 'modelbased', NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison(df_based, 'Tau', 'modelbased', NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# scatter_plot_3d(df)
# scatter_plot_3d_2(df_based)
# brain_map_parcel_average(
#     df=df,
#     measure='ABeta',
#     group='HC',
#     cmap='plasma', #t:inferno, AB:RdYlBu, i:viridis
#     save_path='./plots',
#     vmax=120, #t:2.3, i:0.09, ab:120
#     vmin=-45  #t:0.7, i:0.01, ab:-45
# )
# brain_map_parcel_average(
#     df=df,
#     measure='ABeta',
#     group='MCI(AB+)',
#     cmap='plasma', #t:inferno, AB:RdYlBu, i:viridis
#     save_path='./plots',
#     vmax=120, #t:2.3, i:0.09, ab:120
#     vmin=-45  #t:0.7, i:0.01, ab:-45
# )
# brain_map_parcel_average(
#     df=df,
#     measure='ABeta',
#     group='AD',
#     cmap='plasma', #t:inferno, AB:RdYlBu, i:viridis
#     save_path='./plots',
#     vmax=120, #t:2.3, i:0.09, ab:120
#     vmin=-45  #t:0.7, i:0.01, ab:-45
# )
# brain_map_difference(
#     df=df,
#     measure='Tau',
#     group1='HC',
#     group2='AD',
#     cmap='cividis', 
#     save_path='./plots',
#     vmax=1,
#     vmin=0
# )
# brain_map_difference(
#     df=df,
#     measure='I_norm2',
#     group1='HC',
#     group2='AD',
#     cmap='cividis', 
#     save_path='./plots',
#     vmax=1,
#     vmin=0
# )


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


# plot_multi_model_comparison_violin(
#     model_data_dicts=model_input,
#     groups_order=GROUPS_ORDER_val,
#     measure_name='Integral Violation',
#     y_label_override='Integral violation',
#     test_type='ranksum',
#     # save_path='./plots'  # Uncomment to save
# )

results = plot_violin_groups_with_significance(
    df_data=df,
    measure_col_name='I_norm2',
    group_col_name='group',
    comparisons=[
        ('HC', 'MCI(AB+)'),
        ('HC', 'AD'),
        ('MCI(AB+)', 'AD')
    ],
    measure_display_name='Integral violation [subject average]',
    save_path='./results/plots'
)
plot_violin_groups_with_significance_parcel_mean(
    df_data=df,
    measure_col_name='I_norm2',
    comparisons=[
        ('HC', 'MCI(AB+)'),
        ('HC', 'AD'),
        ('MCI(AB+)', 'AD')
    ],
    measure_display_name='Integral violation [parcel average]',
    save_path='./results/plots',
    n_permutations=10000
)

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
    y_label_override='Integral violation [subject average]',
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
    y_label_override='Integral violation [parcel average]',
    test_type='permutation',
    n_permutations=10000,
)

# # 2. Plot for 2 Models
# model_input = {
#     'Model-free': df_free,
#     'Model-based': df_based
# }

# plot_multi_model_comparison_violin(
#     model_data_dicts=model_input,
#     groups_order=GROUPS_ORDER_val,
#     measure_name='Integral Violation',
#     y_label_override='Integral violation',
#     save_path='./plots',
#     test_type='permutation',
#     n_permutations=10000,
#     measure_col_name=measure
# )

# subject_comparison_rsn(df, measure, 'All', NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'SomMot', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'DorsAttn', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'SalVentAttn', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'Limbic', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'Cont', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)
# subject_comparison_rsn(df, measure, 'Def', model_type, NPARCELLS, fit_sigma, fit_a, save_path_plot=save_path_plot)

