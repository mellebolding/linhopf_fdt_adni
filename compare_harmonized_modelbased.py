"""
Compare all 4 FDT conditions:
1. Model-based original
2. Model-based harmonized
3. Model-free original
4. Model-free harmonized

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

# Load original data (contains both model-based and model-free)
df_orig = pd.read_csv(os.path.join(save_path, 'FDT_data.csv'))
print(f"Original data shape: {df_orig.shape}")

# Load harmonized model-based data
df_mb_har = pd.read_csv(os.path.join(save_path, 'FDT_data_mb_har.csv'))
print(f"Harmonized model-based data shape: {df_mb_har.shape}")

# Load harmonized model-free data
df_mf_har = pd.read_csv(os.path.join(save_path, 'FDT_data_mf_har.csv'))
print(f"Harmonized model-free data shape: {df_mf_har.shape}")

# -------------------------
# Process All 4 Conditions
# -------------------------
print("\nProcessing data for all 4 conditions...")

parcel_cols = [str(i) for i in range(400)]

# 1. Model-based original
df_mb_orig = df_orig[['subject_id', 'group', 'I_norm2_modelbased']].copy()
df_mb_orig = df_mb_orig.rename(columns={'I_norm2_modelbased': 'I_norm2_avg'})
df_mb_orig = df_mb_orig[df_mb_orig['group'].isin(groups_to_keep)].copy()
df_mb_orig = df_mb_orig.dropna(subset=['I_norm2_avg'])
df_mb_orig['condition'] = 'Model-based Original'
print(f"Model-based original: {len(df_mb_orig)} subjects")

# 2. Model-based harmonized
df_mb_har_processed = df_mb_har[['id', 'group', 'site']].copy()
df_mb_har_processed['I_norm2_avg'] = df_mb_har[parcel_cols].mean(axis=1)
df_mb_har_processed = df_mb_har_processed[df_mb_har_processed['group'].isin(groups_to_keep)].copy()
df_mb_har_processed = df_mb_har_processed.rename(columns={'id': 'subject_id'})
df_mb_har_processed['condition'] = 'Model-based Harmonized'
print(f"Model-based harmonized: {len(df_mb_har_processed)} subjects")

# 3. Model-free original
df_mf_orig = df_orig[['subject_id', 'group', 'I_norm2_modelfree']].copy()
df_mf_orig = df_mf_orig.rename(columns={'I_norm2_modelfree': 'I_norm2_avg'})
df_mf_orig = df_mf_orig[df_mf_orig['group'].isin(groups_to_keep)].copy()
df_mf_orig = df_mf_orig.dropna(subset=['I_norm2_avg'])
df_mf_orig['condition'] = 'Model-free Original'
print(f"Model-free original: {len(df_mf_orig)} subjects")

# 4. Model-free harmonized
df_mf_har_processed = df_mf_har[['id', 'group', 'site']].copy()
df_mf_har_processed['I_norm2_avg'] = df_mf_har[parcel_cols].mean(axis=1)
df_mf_har_processed = df_mf_har_processed[df_mf_har_processed['group'].isin(groups_to_keep)].copy()
df_mf_har_processed = df_mf_har_processed.rename(columns={'id': 'subject_id'})
df_mf_har_processed['condition'] = 'Model-free Harmonized'
print(f"Model-free harmonized: {len(df_mf_har_processed)} subjects")

# -------------------------
# Plotting Functions
# -------------------------

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

    groups_order = [g for g in groups_order if g in df['Group'].unique()]

    group_counts = df.groupby('Group')['Value'].count()
    new_xlabels = [f"{group}\n(N={group_counts[group]})" for group in groups_order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_context('notebook', font_scale=1.0)

    sns.violinplot(
        data=df, x='Group', y='Value', hue='Group',
        order=groups_order, hue_order=groups_order,
        palette=palette, inner='quartile', linewidth=0.8,
        ax=ax, legend=False
    )

    sns.stripplot(
        data=df, x='Group', y='Value', hue='Group',
        order=groups_order, edgecolor='black', s=3, alpha=0.6,
        jitter=0.2, linewidth=0.7, palette=palette, dodge=False,
        ax=ax, legend=False
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

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

        x1 = groups_order.index(g1)
        x2 = groups_order.index(g2)

        y = base_height + i * 0.08 * y_range
        print(f"  {g1} vs {g2}: p={p_val:.5f}, effect_size={effect_size:.3f} (n1={n1}, n2={n2})")

        if sig != '':
            ax.plot([x1, x1, x2, x2], [y, y + 0.02*y_range, y + 0.02*y_range, y],
                    lw=1.2, color='black')

        ax.text((x1 + x2) / 2, y + 0.025*y_range, sig, ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plot, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_four_conditions_comparison(df_mb_orig, df_mb_har, df_mf_orig, df_mf_har, palette, comparisons):
    """
    Creates a 2x2 comparison of all 4 conditions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharey=True)
    sns.set_context('notebook', font_scale=1.0)

    groups_order = ['HC', 'MCI(AB+)', 'AD']

    datasets = [
        (df_mb_orig, 'Model-based Original', axes[0, 0]),
        (df_mb_har, 'Model-based Harmonized', axes[0, 1]),
        (df_mf_orig, 'Model-free Original', axes[1, 0]),
        (df_mf_har, 'Model-free Harmonized', axes[1, 1])
    ]

    # Calculate global y-limits across all 4 conditions for consistent scaling
    all_values = []
    for df, _, _ in datasets:
        vals = pd.to_numeric(df['I_norm2_avg'], errors='coerce').dropna()
        all_values.extend(vals.tolist())
    global_min = min(all_values)
    global_max = max(all_values)
    global_range = global_max - global_min if global_max != global_min else 1
    base_height = global_max + 0.08 * global_range
    y_top = base_height + len(comparisons) * 0.08 * global_range + 0.05 * global_range

    for df, title, ax in datasets:
        df = df[['group', 'I_norm2_avg']].copy()
        df.columns = ['Group', 'Value']
        df['Group'] = df['Group'].astype(str)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna()

        available_groups = [g for g in groups_order if g in df['Group'].unique()]

        group_counts = df.groupby('Group')['Value'].count()
        new_xlabels = [f"{group}\n(N={group_counts.get(group, 0)})" for group in available_groups]

        sns.violinplot(
            data=df, x='Group', y='Value', hue='Group',
            order=available_groups, hue_order=available_groups,
            palette=palette, inner='quartile', linewidth=0.8,
            ax=ax, legend=False
        )

        sns.stripplot(
            data=df, x='Group', y='Value', hue='Group',
            order=available_groups, edgecolor='black', s=3, alpha=0.6,
            jitter=0.2, linewidth=0.7, palette=palette, dodge=False,
            ax=ax, legend=False
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

        # Use global base_height and range for consistent bar placement
        for i, (g1, g2) in enumerate(comparisons):
            if g1 not in available_groups or g2 not in available_groups:
                continue

            vals1 = df[df['Group'] == g1]['Value'].values
            vals2 = df[df['Group'] == g2]['Value'].values

            if len(vals1) == 0 or len(vals2) == 0:
                continue

            _, p_val = ranksums(vals1, vals2)

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

            x1 = available_groups.index(g1)
            x2 = available_groups.index(g2)

            y = base_height + i * 0.08 * global_range

            if sig != '':
                ax.plot([x1, x1, x2, x2], [y, y + 0.02*global_range, y + 0.02*global_range, y],
                        lw=1.2, color='black')

            ax.text((x1 + x2) / 2, y + 0.025*global_range, sig, ha='center', va='bottom', fontsize=12)

        ax.set_ylim(bottom=global_min - 0.05 * global_range, top=y_top)

    axes[0, 0].set_ylabel('Integral violation [subject average]', fontsize=14)
    axes[1, 0].set_ylabel('Integral violation [subject average]', fontsize=14)
    axes[0, 1].set_ylabel('')
    axes[1, 1].set_ylabel('')

    plt.tight_layout()
    filename = 'violin_comparison_all_4_conditions.png'
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

# Plot 1: Model-based original
plot_violin_with_significance(
    df_mb_orig, title='Model-based Original',
    filename='violin_modelbased_original.png',
    palette=palette, comparisons=comparisons
)

# Plot 2: Model-based harmonized
plot_violin_with_significance(
    df_mb_har_processed, title='Model-based Harmonized',
    filename='violin_modelbased_harmonized.png',
    palette=palette, comparisons=comparisons
)

# Plot 3: Model-free original
plot_violin_with_significance(
    df_mf_orig, title='Model-free Original',
    filename='violin_modelfree_original.png',
    palette=palette, comparisons=comparisons
)

# Plot 4: Model-free harmonized
plot_violin_with_significance(
    df_mf_har_processed, title='Model-free Harmonized',
    filename='violin_modelfree_harmonized.png',
    palette=palette, comparisons=comparisons
)

# Plot 5: All 4 conditions side-by-side (2x2)
plot_four_conditions_comparison(
    df_mb_orig, df_mb_har_processed, df_mf_orig, df_mf_har_processed,
    palette=palette, comparisons=comparisons
)

# -------------------------
# Summary Statistics
# -------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

all_conditions = [
    ('Model-based Original', df_mb_orig),
    ('Model-based Harmonized', df_mb_har_processed),
    ('Model-free Original', df_mf_orig),
    ('Model-free Harmonized', df_mf_har_processed)
]

for cond_name, df in all_conditions:
    print(f"\n{cond_name}:")
    for group in groups_to_keep:
        vals = df[df['group'] == group]['I_norm2_avg']
        if len(vals) > 0:
            print(f"  {group}: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")

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
            idx0 = idx - 1
            if 0 <= idx0 < nparcels:
                parcel_idx_to_rsn[idx0] = rsn_name

    parcel_rsn_list = [parcel_idx_to_rsn.get(i, 'Unassigned') for i in range(nparcels)]
    return parcel_rsn_list


def aggregate_rsn_data(df, parcel_cols, parcel_rsn_map, rsn_names_list, groups_to_keep):
    """
    Aggregate parcel data by RSN for each subject.
    """
    all_plot_data = []

    for rsn_name in rsn_names_list:
        rsn_indices = [i for i, rsn in enumerate(parcel_rsn_map) if rsn == rsn_name]

        if not rsn_indices:
            print(f"Warning: No parcels found for RSN: {rsn_name}. Skipping.")
            continue

        rsn_parcel_cols = [parcel_cols[i] for i in rsn_indices]

        for idx, row in df.iterrows():
            if row['group'] not in groups_to_keep:
                continue

            rsn_values = row[rsn_parcel_cols].values.astype(float)
            mean_value = np.nanmean(rsn_values)

            all_plot_data.append({
                'subject_id': row.get('subject_id', row.get('id')),
                'Group': row['group'],
                'RSN': rsn_name,
                'Value': mean_value
            })

    return pd.DataFrame(all_plot_data)


def plot_rsn_split_violin(plot_df, parcel_rsn_map, rsn_order, groups_order, title, filename,
                          palette=None, test_type='ranksum'):
    """
    Generates a split violin plot comparison across RSNs.
    """
    if len(groups_order) != 2:
        raise ValueError("Split violin plot requires exactly two groups.")

    if palette is None:
        palette = {'HC': '#8BC34A', 'AD': '#1ABC9C'}

    rsn_parcel_counts = {rsn: parcel_rsn_map.count(rsn) for rsn in rsn_order}
    new_xlabels = [f"{rsn}\n(N={rsn_parcel_counts[rsn]})" for rsn in rsn_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_context('notebook', font_scale=1.0)

    sns.violinplot(
        x='RSN', y='Value', hue='Group', data=plot_df,
        order=rsn_order, hue_order=groups_order, palette=palette,
        split=True, inner='quartile', linewidth=0.8, ax=ax
    )

    sns.stripplot(
        x='RSN', y='Value', hue='Group', data=plot_df,
        order=rsn_order, hue_order=groups_order, palette=palette,
        edgecolor='black', linewidth=0.7, s=3, jitter=0.2, alpha=0.6,
        ax=ax, dodge=True
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

    max_y = plot_df['Value'].max()
    y_range = plot_df['Value'].max() - plot_df['Value'].min()
    y_increment = y_range * 0.16 if y_range > 0 else 1.0
    yposition = max_y + y_increment

    print(f"\n=== {title}: Wilcoxon Rank-Sum Test Results for RSNs ===")

    for i, rsn in enumerate(rsn_order):
        rsn_data_subset = plot_df[plot_df['RSN'] == rsn]
        group1_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[0]]['Value'].values
        group2_values = rsn_data_subset[rsn_data_subset['Group'] == groups_order[1]]['Value'].values

        if len(group1_values) > 0 and len(group2_values) > 0:
            statistic, p_value = ranksums(group1_values, group2_values)
            print(f"  {rsn}: p={p_value:.4f} (n1={len(group1_values)}, n2={len(group2_values)})")

            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''

            if sig:
                ax.text(i, yposition, sig, ha='center', va='bottom', color='black',
                        fontsize=12, fontweight='bold')

    ax.set_ylim(top=yposition + y_increment * 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plot, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_rsn_four_conditions(rsn_dfs, parcel_rsn_map, rsn_order, groups_order, palette=None):
    """
    2x2 RSN comparison of all 4 conditions.
    """
    if palette is None:
        palette = {'HC': '#8BC34A', 'AD': '#1ABC9C'}

    rsn_parcel_counts = {rsn: parcel_rsn_map.count(rsn) for rsn in rsn_order}
    new_xlabels = [f"{rsn}\n(N={rsn_parcel_counts[rsn]})" for rsn in rsn_order]

    # Calculate global y-limits across all 4 conditions for consistent scaling
    global_min = min(df['Value'].min() for df in rsn_dfs)
    global_max = max(df['Value'].max() for df in rsn_dfs)
    global_range = global_max - global_min
    yposition = global_max + global_range * 0.12
    y_top = yposition + global_range * 0.15

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)

    titles = ['Model-based Original', 'Model-based Harmonized',
              'Model-free Original', 'Model-free Harmonized']
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for plot_df, title, ax in zip(rsn_dfs, titles, axes_flat):
        sns.violinplot(
            x='RSN', y='Value', hue='Group', data=plot_df,
            order=rsn_order, hue_order=groups_order, palette=palette,
            split=True, inner='quartile', linewidth=0.8, ax=ax
        )

        sns.stripplot(
            x='RSN', y='Value', hue='Group', data=plot_df,
            order=rsn_order, hue_order=groups_order, palette=palette,
            edgecolor='black', linewidth=0.7, s=3, jitter=0.2, alpha=0.6,
            ax=ax, dodge=True
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

        # Use global yposition for consistent star placement
        for i, rsn in enumerate(rsn_order):
            rsn_data = plot_df[plot_df['RSN'] == rsn]
            g1_vals = rsn_data[rsn_data['Group'] == groups_order[0]]['Value'].values
            g2_vals = rsn_data[rsn_data['Group'] == groups_order[1]]['Value'].values

            if len(g1_vals) > 0 and len(g2_vals) > 0:
                _, p_val = ranksums(g1_vals, g2_vals)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                if sig:
                    ax.text(i, yposition, sig, ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylim(bottom=global_min - global_range * 0.05, top=y_top)

    axes[0, 0].set_ylabel('Integral violation [subject average]', fontsize=14)
    axes[1, 0].set_ylabel('Integral violation [subject average]', fontsize=14)
    axes[0, 1].set_ylabel('')
    axes[1, 1].set_ylabel('')

    plt.tight_layout()
    filename = 'rsn_violin_comparison_all_4_conditions.png'
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

# Parse original parcel data
def parse_parcel_array(s):
    """Parse numpy array string representation"""
    if pd.isna(s):
        return np.array([])
    s = str(s).replace('[', '').replace(']', '').replace('\n', ' ')
    values = [float(x) for x in s.split() if x]
    return np.array(values)

# Create dataframes with parsed parcel values for original data
print("\nProcessing original data for RSN analysis...")

# Model-based original parcels
orig_mb_parcel_data = []
for idx, row in df_orig.iterrows():
    parcel_values = parse_parcel_array(row['I_norm2_parcels_modelbased'])
    if len(parcel_values) == 400:
        orig_mb_parcel_data.append({
            'subject_id': row['subject_id'],
            'group': row['group'],
            **{str(i): parcel_values[i] for i in range(400)}
        })

df_orig_mb_parcels = pd.DataFrame(orig_mb_parcel_data)
print(f"Model-based original parcel data: {df_orig_mb_parcels.shape}")

# Model-free original parcels
orig_mf_parcel_data = []
for idx, row in df_orig.iterrows():
    parcel_values = parse_parcel_array(row['I_norm2_parcels_modelfree'])
    if len(parcel_values) == 400:
        orig_mf_parcel_data.append({
            'subject_id': row['subject_id'],
            'group': row['group'],
            **{str(i): parcel_values[i] for i in range(400)}
        })

df_orig_mf_parcels = pd.DataFrame(orig_mf_parcel_data)
print(f"Model-free original parcel data: {df_orig_mf_parcels.shape}")

# Aggregate RSN data for all 4 conditions
rsn_groups = ['HC', 'AD']

rsn_df_mb_orig = aggregate_rsn_data(
    df_orig_mb_parcels, parcel_cols, parcel_rsn_map, RSNs, groups_to_keep=rsn_groups
)
print(f"Model-based original RSN data: {rsn_df_mb_orig.shape}")

rsn_df_mb_har = aggregate_rsn_data(
    df_mb_har, parcel_cols, parcel_rsn_map, RSNs, groups_to_keep=rsn_groups
)
rsn_df_mb_har = rsn_df_mb_har.rename(columns={'id': 'subject_id'})
print(f"Model-based harmonized RSN data: {rsn_df_mb_har.shape}")

rsn_df_mf_orig = aggregate_rsn_data(
    df_orig_mf_parcels, parcel_cols, parcel_rsn_map, RSNs, groups_to_keep=rsn_groups
)
print(f"Model-free original RSN data: {rsn_df_mf_orig.shape}")

rsn_df_mf_har = aggregate_rsn_data(
    df_mf_har, parcel_cols, parcel_rsn_map, RSNs, groups_to_keep=rsn_groups
)
rsn_df_mf_har = rsn_df_mf_har.rename(columns={'id': 'subject_id'})
print(f"Model-free harmonized RSN data: {rsn_df_mf_har.shape}")

# -------------------------
# RSN Plots
# -------------------------
print("\n" + "="*60)
print("GENERATING RSN PLOTS")
print("="*60)

# Individual RSN plots
plot_rsn_split_violin(
    rsn_df_mb_orig, parcel_rsn_map, RSNs, ['HC', 'AD'],
    title='Model-based Original: RSN Comparison (HC vs AD)',
    filename='rsn_violin_modelbased_original.png'
)

plot_rsn_split_violin(
    rsn_df_mb_har, parcel_rsn_map, RSNs, ['HC', 'AD'],
    title='Model-based Harmonized: RSN Comparison (HC vs AD)',
    filename='rsn_violin_modelbased_harmonized.png'
)

plot_rsn_split_violin(
    rsn_df_mf_orig, parcel_rsn_map, RSNs, ['HC', 'AD'],
    title='Model-free Original: RSN Comparison (HC vs AD)',
    filename='rsn_violin_modelfree_original.png'
)

plot_rsn_split_violin(
    rsn_df_mf_har, parcel_rsn_map, RSNs, ['HC', 'AD'],
    title='Model-free Harmonized: RSN Comparison (HC vs AD)',
    filename='rsn_violin_modelfree_harmonized.png'
)

# Combined 2x2 RSN plot
plot_rsn_four_conditions(
    [rsn_df_mb_orig, rsn_df_mb_har, rsn_df_mf_orig, rsn_df_mf_har],
    parcel_rsn_map, RSNs, ['HC', 'AD']
)

print("\nDone!")
