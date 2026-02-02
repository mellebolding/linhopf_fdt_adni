"""
Compare harmonized model-based FDT data (FDT_data_mb_har.csv) with original model-based data (FDT_data.csv).
Recreates plot 4.1.1: Violin plots of subject averages across groups with significance testing.
Also includes RSN-level analysis plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums, mannwhitneyu

# -------------------------
# Configuration
# -------------------------
repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
save_path_plot = os.path.join(repo_root, "data", "RESULT_PLOTS")
os.makedirs(save_path_plot, exist_ok=True)

NPARCELLS = 400
RSNs = ['Vis', 'SalVentAttn', 'SomMot', 'DorsAttn', 'Limbic', 'Cont', 'Def']

# Group colors
palette = {
    'HC': '#8BC34A',
    'MCI(AB+)': "#c34023",
    'AD': '#1ABC9C'
}

comparisons = [
    ('HC', 'MCI(AB+)'),
    ('HC', 'AD'),
    ('MCI(AB+)', 'AD')
]

groups_to_keep = ["HC", "MCI(AB+)", "AD"]

# -------------------------
# Load Data
# -------------------------
print("Loading data...")

# Load harmonized model-based data
df_har = pd.read_csv(os.path.join(save_path, 'FDT_data_mb_har.csv'))
print(f"Harmonized data shape: {df_har.shape}")
print(f"Harmonized columns: {df_har.columns[:10].tolist()}...")

# Load original model-based data
df_orig = pd.read_csv(os.path.join(save_path, 'FDT_data.csv'))
print(f"Original data shape: {df_orig.shape}")
print(f"Original columns: {df_orig.columns.tolist()}")

# -------------------------
# Process Harmonized Data
# -------------------------
print("\nProcessing harmonized data...")

# Parcel columns are 0-399
parcel_cols = [str(i) for i in range(400)]

# Calculate subject averages for harmonized data
df_har_processed = df_har[['id', 'group', 'site']].copy()
df_har_processed['I_norm2_avg'] = df_har[parcel_cols].mean(axis=1)
df_har_processed = df_har_processed[df_har_processed['group'].isin(groups_to_keep)].copy()
df_har_processed = df_har_processed.rename(columns={'id': 'subject_id'})

print(f"Harmonized data after processing: {df_har_processed.shape}")
print(df_har_processed.head())
print(f"\nHarmonized group counts:\n{df_har_processed['group'].value_counts()}")

# -------------------------
# Process Original Data
# -------------------------
print("\nProcessing original data...")

df_orig_processed = df_orig[['subject_id', 'group', 'I_norm2_modelbased']].copy()
df_orig_processed = df_orig_processed.rename(columns={'I_norm2_modelbased': 'I_norm2_avg'})
df_orig_processed = df_orig_processed[df_orig_processed['group'].isin(groups_to_keep)].copy()
df_orig_processed = df_orig_processed.dropna(subset=['I_norm2_avg'])

print(f"Original data after processing: {df_orig_processed.shape}")
print(df_orig_processed.head())
print(f"\nOriginal group counts:\n{df_orig_processed['group'].value_counts()}")


def plot_violin_with_significance(df, title, filename, palette, comparisons, groups_order=None):
    """
    Creates a violin plot with significance bars for group comparisons.
    """
    df = df.copy()
    df = df[['group', 'I_norm2_avg']].dropna()
    df.columns = ['Group', 'Value']
    df['Group'] = df['Group'].astype(str)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna()
    
    if groups_order is None:
        groups_order = ['HC', 'MCI(AB+)', 'AD']
    
    # Filter to only groups present in data
    groups_order = [g for g in groups_order if g in df['Group'].unique()]
    
    group_counts = df.groupby('Group')['Value'].count()
    new_xlabels = [f"{group}\n(N={group_counts[group]})" for group in groups_order]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_context('notebook', font_scale=1.0)
    
    sns.violinplot(
        data=df,
        x='Group',
        y='Value',
        hue='Group',
        order=groups_order,
        hue_order=groups_order,
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
        order=groups_order,
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
    
    ax.set_title(title, fontsize=15, pad=15)
    ax.set_ylabel('Integral violation [subject average]', fontsize=15)
    ax.set_xlabel('', fontsize=14)
    
    # Add significance bars
    y_max = df['Value'].max()
    y_min = df['Value'].min()
    y_range = y_max - y_min
    base_height = y_max + 0.08 * y_range
    
    print(f"\n{title} - Statistical tests:")
    for i, (g1, g2) in enumerate(comparisons):
        if g1 not in groups_order or g2 not in groups_order:
            continue
            
        vals1 = df[df['Group'] == g1]['Value'].values
        vals2 = df[df['Group'] == g2]['Value'].values
        
        if len(vals1) == 0 or len(vals2) == 0:
            continue
        
        stat, p_val = ranksums(vals1, vals2)
        u_stat, _ = mannwhitneyu(vals1, vals2, alternative='two-sided')
        
        n1, n2 = len(vals1), len(vals2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        sig = '***' if p_val < 0.001 else \
              '**' if p_val < 0.01 else \
              '*' if p_val < 0.05 else ''
        
        x1 = groups_order.index(g1)
        x2 = groups_order.index(g2)
        
        y = base_height + i * 0.08 * y_range
        print(f"  {g1} vs {g2}: p={p_val:.5f}, effect_size={effect_size:.3f} (n1={n1}, n2={n2})")
        
        if sig != '':
            ax.plot([x1, x1, x2, x2],
                    [y, y + 0.02*y_range, y + 0.02*y_range, y],
                    lw=1.2, color='black')
        
        ax.text((x1 + x2) / 2,
                y + 0.025*y_range,
                sig,
                ha='center',
                va='bottom',
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plot, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_side_by_side_comparison(df_orig, df_har, palette, comparisons):
    """
    Creates a side-by-side comparison of original vs harmonized data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    sns.set_context('notebook', font_scale=1.0)
    
    groups_order = ['HC', 'MCI(AB+)', 'AD']
    
    datasets = [
        (df_orig, 'Original Model-Based', axes[0]),
        (df_har, 'Harmonized Model-Based', axes[1])
    ]
    
    for df, title, ax in datasets:
        df = df[['group', 'I_norm2_avg']].copy()
        df.columns = ['Group', 'Value']
        df['Group'] = df['Group'].astype(str)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna()
        
        # Filter to only groups present in data
        available_groups = [g for g in groups_order if g in df['Group'].unique()]
        
        group_counts = df.groupby('Group')['Value'].count()
        new_xlabels = [f"{group}\n(N={group_counts.get(group, 0)})" for group in available_groups]
        
        sns.violinplot(
            data=df,
            x='Group',
            y='Value',
            hue='Group',
            order=available_groups,
            hue_order=available_groups,
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
            order=available_groups,
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
        
        ax.set_xticklabels(new_xlabels, fontsize=12)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('', fontsize=12)
        
        # Add significance bars
        y_max = df['Value'].max()
        y_min = df['Value'].min()
        y_range = y_max - y_min if y_max != y_min else 1
        base_height = y_max + 0.08 * y_range
        
        for i, (g1, g2) in enumerate(comparisons):
            if g1 not in available_groups or g2 not in available_groups:
                continue
                
            vals1 = df[df['Group'] == g1]['Value'].values
            vals2 = df[df['Group'] == g2]['Value'].values
            
            if len(vals1) == 0 or len(vals2) == 0:
                continue
            
            _, p_val = ranksums(vals1, vals2)
            
            sig = '***' if p_val < 0.001 else \
                  '**' if p_val < 0.01 else \
                  '*' if p_val < 0.05 else ''
            
            x1 = available_groups.index(g1)
            x2 = available_groups.index(g2)
            
            y = base_height + i * 0.08 * y_range
            
            if sig != '':
                ax.plot([x1, x1, x2, x2],
                        [y, y + 0.02*y_range, y + 0.02*y_range, y],
                        lw=1.2, color='black')
            
            ax.text((x1 + x2) / 2,
                    y + 0.025*y_range,
                    sig,
                    ha='center',
                    va='bottom',
                    fontsize=12)
    
    axes[0].set_ylabel('Integral violation [subject average]', fontsize=14)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    filename = 'violin_comparison_original_vs_harmonized.png'
    plt.savefig(os.path.join(save_path_plot, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"\nSaved: {filename}")


# -------------------------
# Generate Plots
# -------------------------
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Plot 1: Original model-based data
plot_violin_with_significance(
    df_orig_processed,
    title='Original Model-Based',
    filename='violin_original_modelbased.png',
    palette=palette,
    comparisons=comparisons
)

# Plot 2: Harmonized model-based data
plot_violin_with_significance(
    df_har_processed,
    title='Harmonized Model-Based',
    filename='violin_harmonized_modelbased.png',
    palette=palette,
    comparisons=comparisons
)

# Plot 3: Side-by-side comparison
plot_side_by_side_comparison(
    df_orig_processed,
    df_har_processed,
    palette=palette,
    comparisons=comparisons
)

# -------------------------
# Summary Statistics
# -------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nOriginal Model-Based Data:")
for group in groups_to_keep:
    vals = df_orig_processed[df_orig_processed['group'] == group]['I_norm2_avg']
    if len(vals) > 0:
        print(f"  {group}: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")

print("\nHarmonized Model-Based Data:")
for group in groups_to_keep:
    vals = df_har_processed[df_har_processed['group'] == group]['I_norm2_avg']
    if len(vals) > 0:
        print(f"  {group}: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")

# -------------------------
# Compare distributions
# -------------------------
print("\n" + "="*60)
print("DISTRIBUTION COMPARISON (Original vs Harmonized)")
print("="*60)

# Merge datasets on subject_id
df_merged = df_orig_processed.merge(
    df_har_processed[['subject_id', 'I_norm2_avg']],
    on='subject_id',
    suffixes=('_orig', '_har'),
    how='inner'
)

if len(df_merged) > 0:
    print(f"\nMatched subjects: {len(df_merged)}")
    
    # Correlation between original and harmonized
    corr = np.corrcoef(df_merged['I_norm2_avg_orig'], df_merged['I_norm2_avg_har'])[0, 1]
    print(f"Correlation (orig vs har): r = {corr:.4f}")
    
    # Mean difference
    mean_diff = (df_merged['I_norm2_avg_orig'] - df_merged['I_norm2_avg_har']).mean()
    std_diff = (df_merged['I_norm2_avg_orig'] - df_merged['I_norm2_avg_har']).std()
    print(f"Mean difference (orig - har): {mean_diff:.4f} ± {std_diff:.4f}")
    
    # Create scatter plot of original vs harmonized
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for group in groups_to_keep:
        group_data = df_merged[df_merged['group'] == group]
        ax.scatter(
            group_data['I_norm2_avg_orig'],
            group_data['I_norm2_avg_har'],
            c=palette.get(group, 'gray'),
            label=f"{group} (n={len(group_data)})",
            alpha=0.7,
            s=50
        )
    
    # Add diagonal line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='Identity line')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_xlabel('Original I_norm2 [subject average]', fontsize=12)
    ax.set_ylabel('Harmonized I_norm2 [subject average]', fontsize=12)
    ax.set_title(f'Original vs Harmonized (r = {corr:.3f})', fontsize=14)
    ax.legend(loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plot, 'scatter_original_vs_harmonized.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("\nSaved: scatter_original_vs_harmonized.png")
else:
    print("\nNo matching subjects found between datasets")


# -------------------------
# RSN Analysis Functions
# -------------------------
def get_parcel_rsn_mapping(json_data_path, nparcels=400):
    """Load parcel to RSN mapping from hyperparams.json"""
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    
    network_names = 'Networks_400' if nparcels == 400 else 'Networks'
    
    parcel_idx_to_rsn = {}
    for rsn_name, indices in json_data[network_names].items():
        for idx in indices:
            idx0 = idx - 1  # convert to 0-based index
            if 0 <= idx0 < nparcels:
                parcel_idx_to_rsn[idx0] = rsn_name
    
    parcel_rsn_list = [parcel_idx_to_rsn.get(i, 'Unassigned') for i in range(nparcels)]
    return parcel_rsn_list


def aggregate_rsn_data(df, parcel_cols, parcel_rsn_map, rsn_names_list, groups_to_keep):
    """
    Aggregate parcel data by RSN for each subject.
    Returns a long-format DataFrame with columns: subject_id, Group, RSN, Value
    """
    all_plot_data = []
    
    for rsn_name in rsn_names_list:
        # Get RSN-specific parcel indices
        rsn_indices = [i for i, rsn in enumerate(parcel_rsn_map) if rsn == rsn_name]
        
        if not rsn_indices:
            print(f"Warning: No parcels found for RSN: {rsn_name}. Skipping.")
            continue
        
        rsn_parcel_cols = [parcel_cols[i] for i in rsn_indices]
        
        for idx, row in df.iterrows():
            if row['group'] not in groups_to_keep:
                continue
            
            # Get RSN-specific parcel values and compute mean
            rsn_values = row[rsn_parcel_cols].values.astype(float)
            mean_value = np.nanmean(rsn_values)
            
            all_plot_data.append({
                'subject_id': row.get('subject_id', row.get('id')),
                'Group': row['group'],
                'RSN': rsn_name,
                'Value': mean_value
            })
    
    return pd.DataFrame(all_plot_data)


def permutation_test(group1, group2, n_permutations=10000):
    """Perform a permutation test to compare two groups."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
    
    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return p_value


def plot_rsn_split_violin(plot_df, parcel_rsn_map, rsn_order, groups_order, title, filename,
                          palette=None, test_type='ranksum', n_permutations=10000):
    """
    Generates a split violin plot comparison across RSNs.
    """
    if len(groups_order) != 2:
        raise ValueError("Split violin plot requires exactly two groups.")
    
    if palette is None:
        palette = {
            groups_order[0]: '#8BC34A',
            groups_order[1]: '#1ABC9C'
        }
    
    # Count parcels per RSN
    rsn_parcel_counts = {}
    for rsn_name in rsn_order:
        rsn_parcel_counts[rsn_name] = parcel_rsn_map.count(rsn_name)
    
    new_xlabels = [f"{rsn}\n(N={rsn_parcel_counts[rsn]})" for rsn in rsn_order]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_context('notebook', font_scale=1.0)
    
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
    
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[0:len(groups_order)], labels_leg[0:len(groups_order)],
              loc='upper right', title=None, frameon=True)
    
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel('Integral violation [subject average]', fontsize=12)
    ax.set_xlabel('', fontsize=12)
    
    # Statistical Testing
    max_y = plot_df['Value'].max()
    y_range = plot_df['Value'].max() - plot_df['Value'].min()
    y_increment = y_range * 0.16 if y_range > 0 else 1.0
    yposition = max_y + y_increment
    
    test_name = "Permutation Test" if test_type == 'permutation' else "Wilcoxon Rank-Sum Test"
    print(f"\n=== {title}: {test_name} Results for RSNs ===")
    
    for i, rsn in enumerate(rsn_order):
        rsn_data_subset = plot_df[plot_df['RSN'] == rsn]
        group1_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[0]]['Value'].values
        group2_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[1]]['Value'].values
        
        if len(group1_values) > 0 and len(group2_values) > 0:
            if test_type == 'permutation':
                p_value = permutation_test(group1_values, group2_values, n_permutations=n_permutations)
                print(f"  {rsn}: p={p_value:.4f} (n1={len(group1_values)}, n2={len(group2_values)})")
            else:
                statistic, p_value = ranksums(group1_values, group2_values)
                print(f"  {rsn}: p={p_value:.4f} (n1={len(group1_values)}, n2={len(group2_values)})")
            
            sig = '***' if p_value < 0.001 else \
                  '**' if p_value < 0.01 else \
                  '*' if p_value < 0.05 else ''
            
            if sig:
                ax.text(i, yposition, sig, ha='center', va='bottom', color='black', 
                        fontsize=12, fontweight='bold')
    
    ax.set_ylim(top=yposition + y_increment * 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plot, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_rsn_comparison_side_by_side(rsn_df_orig, rsn_df_har, parcel_rsn_map, rsn_order, groups_order, palette=None):
    """
    Side-by-side RSN comparison of original vs harmonized data.
    """
    if palette is None:
        palette = {
            groups_order[0]: '#8BC34A',
            groups_order[1]: '#1ABC9C'
        }
    
    rsn_parcel_counts = {rsn: parcel_rsn_map.count(rsn) for rsn in rsn_order}
    new_xlabels = [f"{rsn}\n(N={rsn_parcel_counts[rsn]})" for rsn in rsn_order]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    
    datasets = [
        (rsn_df_orig, 'Original Model-Based', axes[0]),
        (rsn_df_har, 'Harmonized Model-Based', axes[1])
    ]
    
    for plot_df, title, ax in datasets:
        sns.violinplot(
            x='RSN', y='Value', hue='Group',
            data=plot_df, order=rsn_order, hue_order=groups_order,
            palette=palette, split=True, inner='quartile',
            linewidth=0.8, ax=ax
        )
        
        sns.stripplot(
            x='RSN', y='Value', hue='Group',
            data=plot_df, order=rsn_order, hue_order=groups_order,
            palette=palette, edgecolor='black', linewidth=0.7,
            s=3, jitter=0.2, alpha=0.6, ax=ax, dodge=True
        )
        
        ax.set_xticklabels(new_xlabels, fontsize=10)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('', fontsize=12)
        
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles[0:len(groups_order)], labels_leg[0:len(groups_order)],
                  loc='upper right', title=None, frameon=True)
        
        # Add significance stars
        max_y = plot_df['Value'].max()
        y_range = plot_df['Value'].max() - plot_df['Value'].min()
        yposition = max_y + y_range * 0.16
        
        for i, rsn in enumerate(rsn_order):
            rsn_data = plot_df[plot_df['RSN'] == rsn]
            g1_vals = rsn_data[rsn_data['Group'] == groups_order[0]]['Value'].values
            g2_vals = rsn_data[rsn_data['Group'] == groups_order[1]]['Value'].values
            
            if len(g1_vals) > 0 and len(g2_vals) > 0:
                _, p_val = ranksums(g1_vals, g2_vals)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                if sig:
                    ax.text(i, yposition, sig, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylim(top=yposition + y_range * 0.2)
    
    axes[0].set_ylabel('Integral violation [subject average]', fontsize=14)
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    filename = 'rsn_violin_comparison_original_vs_harmonized.png'
    plt.savefig(os.path.join(save_path_plot, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"\nSaved: {filename}")


# -------------------------
# RSN Analysis
# -------------------------
print("\n" + "="*60)
print("RSN ANALYSIS")
print("="*60)

# Load parcel to RSN mapping
json_path = os.path.join(repo_root, 'hyperparams.json')
parcel_rsn_map = get_parcel_rsn_mapping(json_path, NPARCELLS)
print(f"Loaded RSN mapping for {NPARCELLS} parcels")

# Parcel columns for harmonized data
parcel_cols = [str(i) for i in range(400)]

# Aggregate RSN data for harmonized dataset
rsn_df_har = aggregate_rsn_data(
    df_har, parcel_cols, parcel_rsn_map, RSNs, groups_to_keep=['HC', 'AD']
)
rsn_df_har = rsn_df_har.rename(columns={'id': 'subject_id'})
print(f"\nHarmonized RSN data shape: {rsn_df_har.shape}")
print(rsn_df_har.head())

# For original data, we need to parse the parcel arrays
# Reload original data and extract parcel values
print("\nProcessing original data for RSN analysis...")
df_orig_full = pd.read_csv(os.path.join(save_path, 'FDT_data.csv'))

# Parse the parcel arrays from string representation
def parse_parcel_array(s):
    """Parse numpy array string representation"""
    if pd.isna(s):
        return np.array([])
    # Clean the string and convert to array
    s = str(s).replace('[', '').replace(']', '').replace('\n', ' ')
    values = [float(x) for x in s.split() if x]
    return np.array(values)

# Create a dataframe with parsed parcel values for original data
orig_parcel_data = []
for idx, row in df_orig_full.iterrows():
    parcel_values = parse_parcel_array(row['I_norm2_parcels_modelbased'])
    if len(parcel_values) == 400:
        orig_parcel_data.append({
            'subject_id': row['subject_id'],
            'group': row['group'],
            **{str(i): parcel_values[i] for i in range(400)}
        })

df_orig_parcels = pd.DataFrame(orig_parcel_data)
print(f"Parsed original parcel data: {df_orig_parcels.shape}")

# Aggregate RSN data for original dataset
rsn_df_orig = aggregate_rsn_data(
    df_orig_parcels, parcel_cols, parcel_rsn_map, RSNs, groups_to_keep=['HC', 'AD']
)
print(f"Original RSN data shape: {rsn_df_orig.shape}")

# -------------------------
# RSN Plots
# -------------------------
print("\n" + "="*60)
print("GENERATING RSN PLOTS")
print("="*60)

# Plot 1: Original RSN violin
plot_rsn_split_violin(
    rsn_df_orig, parcel_rsn_map, RSNs, ['HC', 'AD'],
    title='Original Model-Based: RSN Comparison (HC vs AD)',
    filename='rsn_violin_original_modelbased.png',
    test_type='ranksum'
)

# Plot 2: Harmonized RSN violin
plot_rsn_split_violin(
    rsn_df_har, parcel_rsn_map, RSNs, ['HC', 'AD'],
    title='Harmonized Model-Based: RSN Comparison (HC vs AD)',
    filename='rsn_violin_harmonized_modelbased.png',
    test_type='ranksum'
)

# Plot 3: Side-by-side RSN comparison
plot_rsn_comparison_side_by_side(
    rsn_df_orig, rsn_df_har, parcel_rsn_map, RSNs, ['HC', 'AD']
)

print("\nDone!")
