"""
Produces plots to check Gaussianity of noise term eta(t)
Used in compute_FDT_modelfree.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_eta_histogram(eta, bins=80, title="Noise Gaussianity Check"):
    """
    Histogram + Gaussian fit for eta(t)
    """

    eta_flat = eta.flatten()

    mu, std = norm.fit(eta_flat)

    plt.figure(figsize=(6, 4))
    plt.hist(eta_flat, bins=bins, density=True, alpha=0.7)
    
    x = np.linspace(eta_flat.min(), eta_flat.max(), 500)
    plt.plot(x, norm.pdf(x, mu, std), linewidth=2)

    plt.xlabel("η")
    plt.ylabel("Density")
    plt.title(f"{title}\nμ={mu:.4f}, σ={std:.4f}")
    plt.legend(["Gaussian Fit", "Histogram"])
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "data", "RESULT_PLOTS")
    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, "eta_histogram.png")
    plt.savefig(plot_filename)

    plt.show()

def plot_eta_qq(eta, text_position=(0.05, 0.95), title_suffix=""):
    from scipy.stats import probplot, shapiro
    """
    Q-Q plot for Gaussianity, plotting the Shapiro-Wilk results as text inside the figure.
    
    Args:
        eta (np.array): The noise data (will be flattened).
        text_position (tuple): (x, y) coordinates for the text, in axes fraction (0 to 1).
        title_suffix (str): Optional suffix for the plot title (kept simple, but mainly for context).
    
    Returns:
        tuple: (W_statistic, p_value)
    """
    eta_flat = eta.flatten()
    
    # --- 1. QUANTIFY FIT ---
    # Perform the Shapiro-Wilk test
    W_statistic, p_value = shapiro(eta_flat)

    # Format the results string
    results_text = (
        f"W statistic: {W_statistic:.4f}\n"
        f"p-value: {p_value:.3e}"
    )

    # --- 2. GENERATE PLOT ---
    plt.figure(figsize=(10, 6))
    
    # Generate the Q-Q plot (probplot returns a fig object, which we ignore here)
    probplot(eta_flat, dist="norm", plot=plt)

    # Place the results as text inside the plot
    # The 'transform=plt.gca().transAxes' ensures coordinates are relative to the plot axes (0,0 to 1,1)
    plt.text(
        text_position[0], 
        text_position[1], 
        results_text, 
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
    )

    # Set clear axis labels (as requested previously)
    plt.title(f"Gaussianity Assessment of Noise term  {title_suffix}", fontsize=14)
    plt.xlabel("Theoretical $Z$-scores (normal quantiles)", fontsize=12)
    plt.ylabel("Noise average", fontsize=12)
    plt.legend(["Empirical parcel averages", "Normal line"])
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "data", "RESULT_PLOTS")
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, "eta_qq.png")
    plt.savefig(filename)

    plt.show()