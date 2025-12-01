import os
from turtle import pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def emp_sim_triangles(DL_type='DL_A', NPARCELLS=40, fit_sigma=True, fit_a=True, 
                     joint_normalization=False, n_conditions=4):
    repo_root = os.getcwd()
    save_path = os.path.join(repo_root, "data", "HOPF_DATA")
    filename = f"linhopf_fit_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
    linhopf_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
    df = pd.DataFrame({k: linhopf_data[k].tolist() for k in linhopf_data.files})
    s = 11
    empirical_fcs = np.stack(df['FCemp'][s:s+n_conditions], axis=0)
    simulated_fcs = np.stack(df['FCsim'][s:s+n_conditions], axis=0)
    emp_covtau = np.stack(df['COVtauemp'][s:s+n_conditions], axis=0)
    sim_covtau = np.stack(df['COVtausim'][s:s+n_conditions], axis=0)

    # Truncate and set diagonal to zero §FIRST
    for i in range(n_conditions):
        empirical_fcs[i] = empirical_fcs[i][:NPARCELLS, :NPARCELLS]
        simulated_fcs[i] = simulated_fcs[i][:NPARCELLS, :NPARCELLS]
        np.fill_diagonal(empirical_fcs[i], 0.0)
        np.fill_diagonal(simulated_fcs[i], 0.0)
        np.fill_diagonal(emp_covtau[i], 0.0)
        np.fill_diagonal(sim_covtau[i], 0.0)
    
    # Normalization
    if joint_normalization:
        emp_min = empirical_fcs.min()
        emp_max = empirical_fcs.max()
        sim_min = simulated_fcs.min()
        sim_max = simulated_fcs.max()
        emp_cov_min = emp_covtau.min()
        emp_cov_max = emp_covtau.max()
        sim_cov_min = sim_covtau.min()
        sim_cov_max = sim_covtau.max()

        for i in range(n_conditions):
            empirical_fcs[i] = (empirical_fcs[i] - emp_min) / (emp_max - emp_min)
            simulated_fcs[i] = (simulated_fcs[i] - sim_min) / (sim_max - sim_min)
            emp_covtau[i] = (emp_covtau[i] - emp_cov_min) / (emp_cov_max - emp_cov_min)
            sim_covtau[i] = (sim_covtau[i] - sim_cov_min) / (sim_cov_max - sim_cov_min)
            np.fill_diagonal(empirical_fcs[i], 0.0)
            np.fill_diagonal(simulated_fcs[i], 0.0)
            np.fill_diagonal(emp_covtau[i], 0.0)
            np.fill_diagonal(sim_covtau[i], 0.0)
    else:
        for i in range(n_conditions):
            emp_min = empirical_fcs[i].min()
            emp_max = empirical_fcs[i].max()
            sim_min = simulated_fcs[i].min()
            sim_max = simulated_fcs[i].max()
            emp_cov_min = emp_covtau[i].min()
            emp_cov_max = emp_covtau[i].max()
            sim_cov_min = sim_covtau[i].min()
            sim_cov_max = sim_covtau[i].max()
            
            empirical_fcs[i] = (empirical_fcs[i] - emp_min) / (emp_max - emp_min)
            simulated_fcs[i] = (simulated_fcs[i] - sim_min) / (sim_max - sim_min)
            emp_covtau[i] = (emp_covtau[i] - emp_cov_min) / (emp_cov_max - emp_cov_min)
            sim_covtau[i] = (sim_covtau[i] - sim_cov_min) / (sim_cov_max - sim_cov_min)
            np.fill_diagonal(empirical_fcs[i], 0.0)
            np.fill_diagonal(simulated_fcs[i], 0.0)
            np.fill_diagonal(emp_covtau[i], 0.0)
            np.fill_diagonal(sim_covtau[i], 0.0)
    
    # Get upper triangle indices
    k, j = np.triu_indices(NPARCELLS, k=1)
    
    # Compute per-condition correlations for FC (first 2 conditions)
    print("Per-condition FC correlations:")
    fc_condition_corrs = []
    for i in range(2):
        emp_upper = empirical_fcs[i][k, j]
        sim_upper = simulated_fcs[i][k, j]
        corr = np.corrcoef(emp_upper, sim_upper)[0, 1]
        fc_condition_corrs.append(corr)
        print(f"  FC Condition {i+1}: {corr:.3f}")
    
    # Compute per-condition correlations for COVtau (conditions 3-4 -> indices 0-1)
    print("Per-condition COVtau correlations:")
    cov_condition_corrs = []
    for i in range(2):
        emp_upper = emp_covtau[i][k, j]
        sim_upper = sim_covtau[i][k, j]
        corr = np.corrcoef(emp_upper, sim_upper)[0, 1]
        cov_condition_corrs.append(corr)
        print(f"  COVtau Condition {i+1}: {corr:.3f}")
    
    # Overall correlation
    emp_flat = np.concatenate([empirical_fcs[i][k, j] for i in range(2)])
    sim_flat = np.concatenate([simulated_fcs[i][k, j] for i in range(2)])
    cov_emp_flat = np.concatenate([emp_covtau[i][k, j] for i in range(2)])
    cov_sim_flat = np.concatenate([sim_covtau[i][k, j] for i in range(2)])

    corr_fc_overall = np.corrcoef(emp_flat, sim_flat)[0, 1]
    cov_corr_overall = np.corrcoef(cov_emp_flat, cov_sim_flat)[0, 1]
    print(f"\nOverall FC correlation: {corr_fc_overall:.3f}")
    print(f"Overall COVtau correlation: {cov_corr_overall:.3f}")

    NUMBER_OF_AREAS = empirical_fcs.shape[1]
    CONDITION_LABELS = [f"FC HC", f"FC AD", f"COVtau HC", f"COVtau AD"]
    
    # Define the mixed FCs for the plot (top 2: FC, bottom 2: COVtau)
    mixed_matrices = []
    
    # Top row: FC (HC and AD)
    for index in range(2):
        mixed_fc = empirical_fcs[index, :, :].copy()
        for i in range(NUMBER_OF_AREAS):
            for j in range(NUMBER_OF_AREAS):
                if i == j:
                    mixed_fc[i, j] = 0
                elif i > j:
                    mixed_fc[i, j] = simulated_fcs[index, i, j]
        mixed_matrices.append(mixed_fc)
    
    # Bottom row: COVtau (HC and AD)
    for index in range(2):
        mixed_cov = emp_covtau[index, :, :].copy()
        for i in range(NUMBER_OF_AREAS):
            for j in range(NUMBER_OF_AREAS):
                if i == j:
                    mixed_cov[i, j] = 0
                elif i > j:
                    mixed_cov[i, j] = sim_covtau[index, i, j]
        mixed_matrices.append(mixed_cov)

    # Create figure: 2 rows x 3 columns
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.1, wspace=0.1)
    
    # Legend plot spans both rows in first column
    ax_legend = fig.add_subplot(gs[:, 0])
    
    # 4 conditions in 2x2 grid (columns 1 and 2)
    axs_conditions = [
        fig.add_subplot(gs[0, 1]),  # FC HC
        fig.add_subplot(gs[0, 2]),  # FC AD
        fig.add_subplot(gs[1, 1]),  # COVtau HC
        fig.add_subplot(gs[1, 2])   # COVtau AD
    ]

    # Create legend
    square = np.zeros((NPARCELLS, NPARCELLS))
    square[:, :1], square[:, -1:], square[:1, :], square[-1:, :] = 1, 1, 1, 1

    ax_legend.imshow(square, cmap="binary")
    ax_legend.plot([0, NPARCELLS - 1], [0, NPARCELLS - 1], c='k', linewidth=6)
    ax_legend.annotate("Empirical matrix", (2 * NPARCELLS // 3, NPARCELLS // 3), 
                       horizontalalignment='center', verticalalignment='center', 
                       fontsize=16, rotation=315)
    ax_legend.annotate("Simulated matrix", (NPARCELLS // 3, 2 * NPARCELLS // 3), 
                       horizontalalignment='center', verticalalignment='center', 
                       fontsize=16, rotation=315)
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_legend.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=10, width=1.5)
    ax_legend.set_yticks([0, NPARCELLS // 3, 2 * NPARCELLS // 3, NPARCELLS - 1])
    ax_legend.set_yticklabels(['0', f'{NPARCELLS // 3}', f'{2 * NPARCELLS // 3}', 
                               f'{NPARCELLS - 1}'], fontweight='bold')
    ax_legend.set_ylabel("Node", fontsize=14, fontweight='bold')

    # Plot conditions with correlations in titles
    im = None
    vmin, vmax = 0, 1
    all_corrs = fc_condition_corrs + cov_condition_corrs
    
    for index in range(4):
        ax = axs_conditions[index]
        im = ax.imshow(mixed_matrices[index], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{CONDITION_LABELS[index]}", #(r={all_corrs[index]:.3f})", 
                     fontsize=14, fontweight='bold')
        ax.axis("off")

    plt.tight_layout()
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axs_conditions, shrink=0.8, label='Connectivity', 
                       pad=0.02, ticks=[0, 1])
    
    cbar.ax.yaxis.label.set_fontsize(14)
    cbar.ax.yaxis.label.set_fontweight('bold')
    print(DL_type, NPARCELLS, fit_sigma, fit_a, joint_normalization)
    plt.show()

def plot_ceff_matrices(df, groups=['HC', 'MCI(AB+)', 'AD'], n_parcels=400):
    """
    Visualizes Ceff matrices for specified groups.
    
    Args:
        df: DataFrame with columns ['group', 'Ceff']
        groups: List of group labels to plot
        n_parcels: Number of parcels (default: 400)
    """
    
    print(f"\n{'='*60}")
    print(f"Plotting Ceff matrices for: {groups}")
    print(f"{'='*60}")
    
    # Extract and process Ceff matrices for each group
    group_matrices = []
    
    for group in groups:
        group_df = df[df['group'] == group]
        
        if group_df.empty:
            raise ValueError(f"No data found for group: {group}")
        
        # Stack all subjects and average
        all_ceff = np.stack(group_df['Ceff'].values, axis=0)  # (n_subjects, n_parcels, n_parcels)
        avg_ceff = np.nanmean(all_ceff, axis=0)  # (n_parcels, n_parcels)
        
        # Truncate if needed and set diagonal to zero
        avg_ceff = avg_ceff[:n_parcels, :n_parcels]
        np.fill_diagonal(avg_ceff, 0.0)
        
        group_matrices.append(avg_ceff)
        
        print(f"\n{group}:")
        print(f"  Subjects: {len(group_df)}")
        print(f"  Matrix shape: {avg_ceff.shape}")
        print(f"  Value range: [{np.min(avg_ceff):.4f}, {np.max(avg_ceff):.4f}]")
    
    # Normalize all matrices together to [0, 1]
    all_values = np.concatenate([mat.flatten() for mat in group_matrices])
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    
    # Apply normalization
    group_matrices = [(mat - vmin) / (vmax - vmin) for mat in group_matrices]
    
    print(f"\nGlobal value range before normalization: [{vmin:.4f}, {vmax:.4f}]")
    print(f"Normalized to: [0, 1]")
    
    # Create figure with vertical layout
    fig, axes = plt.subplots(len(groups), 1, figsize=(5, 3*len(groups)))
    
    if len(groups) == 1:
        axes = [axes]
    
    # Plot each matrix
    for idx, (ax, group, matrix) in enumerate(zip(axes, groups, group_matrices)):
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title(group, fontsize=16, fontweight='bold')
        #ax.set_xlabel('Node', fontsize=12, fontweight='bold')
        ax.set_ylabel('Node', fontsize=12, fontweight='bold')
        
        # Set ticks
        tick_positions = [0, n_parcels//2, n_parcels-1]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontweight='bold')
        ax.set_yticklabels(tick_positions, fontweight='bold')
    
    plt.suptitle('Effective Connectivity (Ceff) by Group', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    return group_matrices

# emp_sim_triangles(DL_type='DL_B2', NPARCELLS=400, fit_sigma=True, fit_a=True, 
#                   joint_normalization=True, n_conditions=4)


DL_type = 'DL_B2'
NPARCELLS = 400 # max 379
fit_sigma = True
fit_a = True

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "HOPF_DATA")
filename = f"linhopf_fit_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
linhopf_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: linhopf_data[k].tolist() for k in linhopf_data.files})

plot_ceff_matrices(df, groups=['HC', 'MCI(AB+)', 'AD'], n_parcels=NPARCELLS)

all_keys = set(key for d in df['losses'] for key in d.keys())
print(f"All unique keys found in 'losses': {all_keys}")
corr_fc_values = df['losses'].apply(lambda d: d['corr_fc'])
COVtau_corr_values = df['losses'].apply(lambda d: d['corr_covtau'])
mse_fc_values = df['losses'].apply(lambda d: d['mse_fc'])
mse_COVtau_values = df['losses'].apply(lambda d: d['mse_covtau'])
print(f"Correlation FC values:\n{corr_fc_values.mean()} ± {corr_fc_values.std()}")
print(f"Correlation COVtau values:\n{COVtau_corr_values.mean()} ± {COVtau_corr_values.std()}")
print(f"MSE FC values:\n{mse_fc_values.mean()} ± {mse_fc_values.std()}")
print(f"MSE COVtau values:\n{mse_COVtau_values.mean()} ± {mse_COVtau_values.std()}")

