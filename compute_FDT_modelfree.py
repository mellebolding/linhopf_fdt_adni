import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.data_loaders import ADNI_A, ADNI_B, load_data_records
from src.functions_frameworks.functions_FDT_modelfree import _splitSignal, _analysisFdt2, _computeDistanceFromEquilibrium, _analysisFdt3, _analysisFdt4, _analysisFdt1
from src.data_processing.zfilterts import filter_time_series,prepare_timeseries

def detrend_data(data, detrend=True):
    from scipy.signal import detrend as scipy_detrend
    data = np.asarray(data)
    # Apply detrending if requested
    if detrend:
        if data.ndim == 2:
            # [NPARCELS, NTIMES]
            data = scipy_detrend(data, axis=1, type='linear')
        elif data.ndim == 3:
            # [NSUB, NPARCELS, NTIMES]
            # Detrend along the time axis (axis=2) for each parcel of each subject
            data = scipy_detrend(data, axis=2, type='linear')
        else:
            raise ValueError("Input data must be 2D or 3D for detrending.")
    return data
def filter_time_series(ts_array, TR=3.0, low_pass_freq=0.008, high_pass_freq=0.08):
    """
    Preprocessing for model-free FDT, implementing a band-stop filter
    to remove frequencies between low_pass_freq and high_pass_freq.
    
    Parameters
    ----------
    ts_array : ndarray
        Shape (n_parcels, n_timepoints)
    TR : float
        Repetition time in seconds
    low_pass_freq : float
        Lower cutoff frequency for the band-stop filter (Hz)
    high_pass_freq : float
        Upper cutoff frequency for the band-stop filter (Hz)
    """
    from scipy.signal import butter, filtfilt
    
    nyquist = 1.0 / (2 * TR)
    
    # --- 1. Normalize the cutoff frequencies ---
    # The normalized frequencies must be between 0 and 1.
    Wn_low = low_pass_freq / nyquist
    Wn_high = high_pass_freq / nyquist
    
    # The band-stop filter requires a 2-element array for the cutoff
    Wn = [Wn_low, Wn_high]

    # --- 2. Validation ---
    if Wn_low <= 0 or Wn_high >= 1 or Wn_low >= Wn_high:
        raise ValueError(
            f"Invalid normalized frequency range: {[Wn_low, Wn_high]}\n"
            f"Frequencies must be 0 < low < high < {nyquist:.4f} Hz."
        )

    # --- 3. Band-Stop Filter Implementation ---
    # btype='bandstop' removes the frequencies *between* Wn_low and Wn_high.
    b, a = butter(2, Wn, btype='bandstop') 
    
    # Apply the filter using filtfilt for zero-phase distortion
    ts_filtered = filtfilt(b, a, ts_array, axis=-1, padlen=150)
    
    return ts_filtered

from scipy.stats import probplot
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
    plt.title(f"Gaussianity Assessment of Noise Term $\eta(t)$ {title_suffix}", fontsize=14)
    plt.xlabel("Theoretical $Z$-scores (normal quantiles)", fontsize=12)
    plt.ylabel("Noise ($\eta$) average", fontsize=12)
    plt.legend(["Empirical parcel averages", "Normal line"])
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_I_heatmap(I_norm2, title="FDT Violation I(t, t')"):
    """
    Heatmap of I(t, t') with time on both axes.
    """

    n = I_norm2.shape[1]
    print(I_norm2.shape)
    t = np.arange(n)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        I_norm2,
        origin='lower',
        extent=[t[0], t[-1], t[0], t[-1]],
        aspect='auto'
    )

    plt.colorbar(im, label="Normalized FDT Violation  Î")
    plt.xlabel("t (time)")
    plt.ylabel("t' (time)")
    plt.title(title)
    plt.tight_layout()
    plt.show()



DL_type = 'DL_B2'
filt = True
TR = 3

if DL_type == 'DL_A': 
    DL = ADNI_A.ADNI_A(normalizeBurden=False)
    NPARCELLS = 379
if DL_type == 'DL_B1': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'],filt=filt)
    NPARCELLS = 400  
if DL_type == 'DL_B2': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD'],filt=filt)
    NPARCELLS = 400

all_data = []
ts = np.array([])
for group in DL.get_groupLabels():
    all_data.extend(load_data_records.load_group_data(DL, group, DL_type))
    
results = []
eta_all = np.zeros((len(all_data), NPARCELLS))
TSemp_zsc = prepare_timeseries(all_data, NPARCELLS)
for i in range(len(all_data)):
    print("subject ", i, " of ", len(all_data))
    # all_data[i]['MRI'][:,:NPARCELLS] = detrend_data(all_data[i]['MRI'][:,:NPARCELLS])
    #all_data[i]['MRI'][:,:NPARCELLS] = filter_time_series(all_data[i]['MRI'][:,:NPARCELLS])
    all_data[i]['MRI'][:,:NPARCELLS] = TSemp_zsc[i,:,:].T
    x, dxdt, Fx, eta = _splitSignal(all_data[i]['MRI'][:,:NPARCELLS])
    eta_all[i, :] = np.mean(eta, axis=0)
    dt = TR / 1000.
    sigma = np.std(eta)
    I_inf, I_norm_inf, X_norm_inf = _analysisFdt1(x, eta, sigma, dt,return_time_matrix=False)
    # intI = I_norm_inf
    # intX = X_norm_inf
    intI = _computeDistanceFromEquilibrium(I_norm_inf)
    intX = _computeDistanceFromEquilibrium(X_norm_inf)
    results.append({
        'subject_id': all_data[i]['subject_id'],
        'group': all_data[i]['group'],
        'sigma': sigma,
        'I_norm2': np.abs(1-intI),
        'X_norm2': np.abs(intX),
    })

eta_avg = np.mean(eta_all, axis=0)  # Average across subjects to get 400 values
plot_eta_qq(eta_avg)
plot_eta_histogram(eta_all)
# _,_,_,I_full = _analysisFdt1(x, eta, sigma, dt,return_time_matrix=True)
# plot_I_heatmap(I_full[10])

results_dict = {key: np.array([d[key] for d in results])
        for key in ['subject_id', 'group', 'I_norm2', 'X_norm2', 'sigma']}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)









    

    # what we want: I and X norm 2. 
    # not sure how to go from what we find here to that, as norm 2 seems to involve noise.
    # after implementation, it is important that this becomes clear and is consistent with the model-based version.