import numpy as np
from numba import jit, njit, prange
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov

def I_norm2(df,tmax=100.0, ts0=0.0, gconst=1.0):
    I_values = []
    for _, row in df.iterrows():
        EC_matrix = row['Ceff']
        f_diff = np.array(row['f_diff'])
        sigma = np.array(row['sigma'])
        a = np.array(row['a'])

        I = Int_violation_norm2(EC_matrix, f_diff, sigma, a,
                                tmax=tmax, ts0=ts0, gconst=gconst)
        I_values.append(I)

    df['I_norm2'] = pd.Series(I_values, index=df.index)
    return df

def X_norm2(df,tmax=100.0, ts0=0.0, gconst=1.0):
    X_values = []
        
    for _, row in df.iterrows():
        EC_matrix = row['Ceff']
        f_diff = np.array(row['f_diff'])
        sigma = np.array(row['sigma'])
        a = np.array(row['a'])

        X = X_suscep_norm2(EC_matrix, f_diff, sigma, a,
                           tmax=tmax, ts0=ts0, gconst=gconst)
        X_values.append(X)

    df['X_norm2'] = pd.Series(X_values, index=df.index)
    return df

def Int_violation_norm2(EC_matrix, f_diff, sigma, a, **params):

    tmax = params.get('tmax', 100.0) 
    ts0 = params.get('ts0', 0.0)      
    gconst = params.get('gconst', 1.0)  
    
    NPARCELLS = f_diff.shape[0]
    omega = 2 * np.pi * f_diff
    sigma_2 = np.append(sigma, sigma)

    gamma = -construct_matrix_A(a, omega, EC_matrix, gconst)
    D = np.diag(sigma_2**2 * np.ones(2*NPARCELLS))
    V_0 = solve_continuous_lyapunov(gamma, D)

    Inorm2_tmax = Its_norm2_Langevin_ND(gamma, sigma_2, V_0, tmax, ts0)[0:NPARCELLS]

    return Inorm2_tmax

def X_suscep_norm2(EC_matrix, f_diff, sigma, a, **params):

    tmax = params.get('tmax', 5000.0) 
    ts0 = params.get('ts0', 0.0)      
    gconst = params.get('gconst', 1.0)  
    
    NPARCELLS = f_diff.shape[0]
    omega = 2 * np.pi * f_diff
    sigma_2 = np.append(sigma, sigma)

    gamma = -construct_matrix_A(a, omega, EC_matrix, gconst)
    D = np.diag(sigma_2**2 * np.ones(2*NPARCELLS))
    V_0 = solve_continuous_lyapunov(gamma, D)

    intRnorm2 = intRts_norm2_Langevin_ND(gamma, sigma_2, V_0, tmax, ts0)[0:NPARCELLS]

    return intRnorm2

@jit(forceobj=True)
def Its_norm2_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Inorm2 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (C_tt + eps)
    return Inorm2

def intRts_norm2_Langevin_ND(Gamma, sigma, V_0, t, s):
    """
    Normalizes susceptibility using noise amplitude and autocorrelation:
        tilde_chi_i = (sigma_i^2 * chi_i(t,s)) / (2 * C_i(t,t))
    """
    Ctt_diag = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    sigma_squared = sigma**2

    I = np.eye(Gamma.shape[0])
    Gamma_inv = np.linalg.inv(Gamma)
    
    chi = np.diag(Gamma_inv @ (I - exp_Gamma_t_minus_s))  # chi_i(t,s)
    
    # Noise-normalized susceptibility
    intRts_norm2 = (sigma_squared * chi) / (2 * Ctt_diag)
    return intRts_norm2

@jit(forceobj=True)
def Cts_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Exponential matrices using scipy.linalg.expm
    exp_Gamma_t = exp_scaling_squaring(-Gamma * t)
    exp_GammaT_s = exp_scaling_squaring(-Gamma.T * s)
    # First term: (e^{-\Gamma t} V_0 e^{-\Gamma^T s})_{ii}
    term1 = np.diag(exp_Gamma_t @ V_0 @ exp_GammaT_s)
    # Second term: 
    term2 = compute_term2_of_C_precompute(t, s, Gamma, sigma**2)
    C = term1 + term2
    return C

def compute_term2_of_C_precompute(t, s, Gamma, sigma_squared):
    """
    Wrapper to precompute eigenvalues and eigenvectors for a general matrix Gamma.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    P = eigvecs
    P_inv = np.linalg.inv(P)
    return compute_term2_of_C_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared)

@njit
def compute_term2_of_C_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared):
    """
    Optimized computation of the vector B(t, s) using Numba.
    """
    N = len(eigvals)
    
    # eigvals = np.real(eigvals)

    # Ensure sigma_squared is an array
    if np.isscalar(sigma_squared):
        sigma_squared = sigma_squared * np.ones(N)
    
    # Precompute sigma weight terms
    S = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        for m in range(N):
            S[k, m] = np.sum(sigma_squared * P_inv[k, :] * P_inv[m, :])
    
    # Precompute exponentials and integration terms
    integral_term = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        eigval_k = eigvals[k]
        for m in range(N):
            eigval_m = eigvals[m]
            denom = eigval_k + eigval_m
            if np.abs(denom) > 1e-12:  # Avoid division by zero
                integral_term[k, m] = (np.exp(-eigval_k * (t - s)) - np.exp(-(eigval_k * t + eigval_m * s))) / denom
            else:
                integral_term[k, m] = s * np.exp(-(eigval_k * t + eigval_m * s)) # Special case when denom is zero
    
    # Compute B[i]
    term2_of_C = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        for k in range(N):
            for m in range(N):
                term_km = (
                    P[i, k] * P[i, m] *
                    integral_term[k, m] *
                    S[k, m]
                )
                term2_of_C[i] += term_km
    
    return np.real(term2_of_C)  # Return the real part

@jit(nopython=True)
def exp_scaling_squaring(Gamma, m=10):
    """
    Compute exp(Gamma) using the Scaling and Squaring Method.
    
    Parameters:
        Gamma (ndarray): Input square matrix (NxN).
        m (int): Degree of the Pade approximant for the series expansion.
    
    Returns:
        ndarray: Matrix exponential exp(Gamma).
    """
    # Compute the norm of Gamma (1-norm)
    norm_Gamma = np.linalg.norm(Gamma, ord=1)
    
    # Scaling step: Find scaling factor 2^k
    k = max(0, int(np.ceil(np.log2(norm_Gamma))))  # Scale such that norm(Gamma / 2^k) is small
    Gamma_scaled = Gamma / (2**k)
    
    # Compute exp(Gamma_scaled) using a series expansion or Pade approximant
    N = Gamma.shape[0]
    result = np.eye(N)  # Initialize result as the identity matrix
    term = np.eye(N)    # Initialize current term as the identity matrix
    
    for i in range(1, m + 1):
        term = np.dot(term, Gamma_scaled) / i  # Compute the next term in the series
        result += term  # Add to the result
    
    # Squaring step: Square the result k times
    for _ in range(k):
        result = np.dot(result, result)
    
    return result

def construct_matrix_A(a, omega, C, g):
    # Ensure inputs are numpy arrays
    a = np.array(a)
    omega = np.array(omega)
    C = np.array(C)
    # Calculate S
    S = np.sum(C, axis=1)
    # Diagonal matrices
    diag_a_minus_gS = np.diag(a - g * S)
    diag_omega = np.diag(omega)
    # Matrices A_xx, A_yy
    A_xx = diag_a_minus_gS + g * C
    A_yy = A_xx
    # Matrices A_xy, A_yx
    A_xy = diag_omega
    A_yx = -diag_omega
    # Construct the full matrix A
    A = np.block([
        [A_xx, A_xy],
        [A_yx, A_yy]
    ])
    return A