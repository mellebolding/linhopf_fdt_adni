# Import libraries
import os
from turtle import pd
import numpy as np
import matplotlib.pyplot as plt

def emp_sim_triangles(DL_type='DL_A', NPARCELLS=379, fit_sigma=True, fit_a=False):
    
    repo_root = os.getcwd() 
    save_path = os.path.join(repo_root, "data", "HOPF_DATA")
    filename = f"linhopf_fit_{DL_type}_N{NPARCELLS}_sig{fit_sigma}_a{fit_a}.npz"
    linhopf_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
    df = pd.DataFrame({k: linhopf_data[k].tolist() for k in linhopf_data.files})

    # Load the data
    empirical_fcs = np.load("replace_for_empirical_fcs_path.npy") # Shape: [Condition, Areas, Areas]
    simulated_fcs = np.load("replace_for_simulated_fcs_path.npy") # Shape: [Condition, Areas, Areas]
    NUMBER_OF_AREAS = empirical_fcs.shape[2]

# Define constants
CONDITION_KEYS = ["condition 1", "condition 2", "condition 3", "condition 4", "condition 5"]
CONDITION_LABELS = ["First condition", "Second condition", "Third condition",
                    "Fourth condition", "Fifth condition"]

# Normalize the data: You can do it with both of them together (concatenate and minmax norm) or separate (like here)
minmax_norm = lambda matrix: (matrix - matrix.min()) / (matrix.max() - matrix.min())
empirical_fcs_normalized = minmax_norm(empirical_fcs)
simulated_fcs_normalized = minmax_norm(simulated_fcs)

# Define the mixed FCs for the plot
mixed_fcs = {}
for index, condition in enumerate(CONDITION_KEYS):
    mixed_fcs[condition] = empirical_fcs[index, :, :]
    for i in range(NUMBER_OF_AREAS):
        for j in range(NUMBER_OF_AREAS):
            if i == j: # Take diagonal out
                mixed_fcs[condition][i, j] = 0
            elif i > j: # Put the data from simulated in lower triangle
                mixed_fcs[condition][i, j] = simulated_fcs[index, i, j]
            else: # for upper triangle, data is already good
                continue

# Get the figure
fig, axs = plt.subplots (2, 3, figsize=(12, 8), sharex=True, sharey=True)

for index, condition in enumerate(CONDITION_KEYS):
    ax = axs.flatten()[i+1]
    ax.imshow(mixed_fcs[condition], cmap="viridis")
    ax.set_title(CONDITION_LABELS[index])
    ax.axis("off")

plt.savefig("../filename.svg")