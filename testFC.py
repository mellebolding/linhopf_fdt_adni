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
    
    empirical_fcs = np.stack(df['FCemp'][:n_conditions], axis=0)
    simulated_fcs = np.stack(df['FCsim'][:n_conditions], axis=0)
    #print(df['losses'].values)
    print(df['a'][0])
    print(df['sigma'][0])

    # Truncate and set diagonal to zero §FIRST
    for i in range(n_conditions):
        empirical_fcs[i] = empirical_fcs[i][:NPARCELLS, :NPARCELLS]
        simulated_fcs[i] = simulated_fcs[i][:NPARCELLS, :NPARCELLS]
        np.fill_diagonal(empirical_fcs[i], 0.0)
        np.fill_diagonal(simulated_fcs[i], 0.0)
    
    # Normalization
    if joint_normalization:
        emp_min = empirical_fcs.min()
        emp_max = empirical_fcs.max()
        sim_min = simulated_fcs.min()
        sim_max = simulated_fcs.max()
        
        for i in range(n_conditions):
            empirical_fcs[i] = (empirical_fcs[i] - emp_min) / (emp_max - emp_min)
            simulated_fcs[i] = (simulated_fcs[i] - sim_min) / (sim_max - sim_min)
            np.fill_diagonal(empirical_fcs[i], 0.0)
            np.fill_diagonal(simulated_fcs[i], 0.0)
    else:
        for i in range(n_conditions):
            emp_min = empirical_fcs[i].min()
            emp_max = empirical_fcs[i].max()
            sim_min = simulated_fcs[i].min()
            sim_max = simulated_fcs[i].max()
            
            empirical_fcs[i] = (empirical_fcs[i] - emp_min) / (emp_max - emp_min)
            simulated_fcs[i] = (simulated_fcs[i] - sim_min) / (sim_max - sim_min)
            np.fill_diagonal(empirical_fcs[i], 0.0)
            np.fill_diagonal(simulated_fcs[i], 0.0)
    
    # Get upper triangle indices
    k, j = np.triu_indices(NPARCELLS, k=1)
    
    # Compute per-condition correlations
    print("Per-condition FC correlations:")
    condition_corrs = []
    for i in range(n_conditions):
        emp_upper = empirical_fcs[i][k, j]
        sim_upper = simulated_fcs[i][k, j]
        corr = np.corrcoef(emp_upper, sim_upper)[0, 1]
        condition_corrs.append(corr)
        print(f"  Condition {i+1}: {corr:.3f}")
    
    # Overall correlation (all conditions combined)
    emp_flat = np.concatenate([empirical_fcs[i][k, j] for i in range(n_conditions)])
    sim_flat = np.concatenate([simulated_fcs[i][k, j] for i in range(n_conditions)])
    corr_fc_overall = np.corrcoef(emp_flat, sim_flat)[0, 1]
    print(f"\nOverall FC correlation: {corr_fc_overall:.3f}")
    
    # Plot with shared colorbar
    # fig, axes = plt.subplots(2, n_conditions, figsize=(4*n_conditions, 8))
    
    # # Use [0, 1] range since normalized
    vmin, vmax = 0, 1
    
    # for i in range(n_conditions):
    #     # Empirical
    #     im0 = axes[0, i].imshow(empirical_fcs[i], cmap='viridis', vmin=vmin, vmax=vmax)
    #     axes[0, i].set_title(f'FCemp\nCond {i+1}\nr={condition_corrs[i]:.3f}')
    #     axes[0, i].axis('off')
        
    #     # Simulated
    #     im1 = axes[1, i].imshow(simulated_fcs[i], cmap='viridis', vmin=vmin, vmax=vmax)
    #     axes[1, i].set_title(f'FCsim\nCond {i+1}')
    #     axes[1, i].axis('off')
    
    # # Add colorbar
    # fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.6, label='Normalized Values')
    # norm_type = "Joint" if joint_normalization else "Separate"
    # plt.suptitle(f'Overall correlation: {corr_fc_overall:.3f} ({norm_type} normalization)', 
    #              fontsize=14, y=0.98)
    # plt.tight_layout()
    # plt.show()

    NUMBER_OF_AREAS = empirical_fcs.shape[2]
    CONDITION_KEYS = [f"condition {i+1}" for i in range(n_conditions)]
    CONDITION_LABELS = [f"FC HC (r={condition_corrs[0]:.3f})", f"FC AD (r={condition_corrs[1]:.3f})", f"COVtau HC (r={condition_corrs[2]:.3f})", f"COVtau AD (r={condition_corrs[3]:.3f})", 'Fifth', 'Sixth', 'Seventh', 'Eighth']
    CONDITION_LABELS = [f"{CONDITION_LABELS[i]}" for i in range(n_conditions)]
    
    # Define the mixed FCs for the plot
    mixed_fcs = {}
    for index, condition in enumerate(CONDITION_KEYS):
        mixed_fcs[condition] = empirical_fcs[index, :, :].copy()
        for i in range(NUMBER_OF_AREAS):
            for j in range(NUMBER_OF_AREAS):
                if i == j:
                    mixed_fcs[condition][i, j] = 0
                elif i > j:
                    mixed_fcs[condition][i, j] = simulated_fcs[index, i, j]

    # Create figure based on number of conditions
    if n_conditions == 4:
        # Layout: 2 rows x 3 columns (legend on left center, 4 conditions in 2x2 grid)
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.1, wspace=0.1)
        
        # Legend plot spans both rows in first column
        ax_legend = fig.add_subplot(gs[:, 0])
        
        # 4 conditions in 2x2 grid (columns 1 and 2)
        axs_conditions = [
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2])
        ]
        
    else:  # n_conditions == 5
        # Original 2x3 layout
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        ax_legend = axs[0, 0]
        axs_conditions = [axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]

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

    # Plot conditions
    im = None
    for index, condition in enumerate(CONDITION_KEYS):
        ax = axs_conditions[index]
        im = ax.imshow(mixed_fcs[condition], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(CONDITION_LABELS[index], fontsize=14, fontweight='bold')
        ax.axis("off")

    plt.tight_layout()
    
    # Add colorbar
    if n_conditions == 4:
        cbar = fig.colorbar(im, ax=axs_conditions, shrink=0.8, label='Connectivity', 
                           pad=0.02, ticks=[0, 1])
    else:
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label='Connectivity', 
                           pad=0.02, ticks=[0, 1])
    
    cbar.ax.yaxis.label.set_fontsize(14)
    cbar.ax.yaxis.label.set_fontweight('bold')
    print(DL_type, NPARCELLS, fit_sigma, fit_a, joint_normalization)
    plt.show()

emp_sim_triangles(DL_type='DL_B1', NPARCELLS=20, fit_sigma=True, fit_a=True, 
                 joint_normalization=False, n_conditions=4)

# def connectivity_matrices3():
