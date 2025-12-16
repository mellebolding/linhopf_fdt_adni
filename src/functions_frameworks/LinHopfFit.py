"""
This script implements the Linear Hopf model fitting using the Adam optimizer.
The main fitting function is called: 'fit' and is part of the accessor function 'fit_linhopf'.

The pipeline of the 'fit' function is as follows:
  - Defining the optimizer and loss function
  - Computing empirical FC, COV, COVtau from time series data
  - Define initial values
  - Fitting loop: (1) compute simulated EC through FC and COVtau, 
                  (2) compute loss and gradients, 
                  (3) update parameters and apply constraints
  - Return fitted parameters and compute final simulated FC and COVtau
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from numba import njit, prange, jit
from typing import Dict, Union

# ==================== Core Numba Functions ====================
@njit
def compute_sigratio_from_cov(COV):
    N = COV.shape[0]
    sigratio = np.zeros((N, N))
    for i in range(N):
        sqrt_ii = np.sqrt(COV[i, i])
        for j in range(N):
            sigratio[i, j] = 1.0 / (sqrt_ii * np.sqrt(COV[j, j]))
    return sigratio

@njit
def corrcov_py_numba(C):
    N = C.shape[0]
    corr = np.empty((N, N))
    stddev = np.empty(N)
    
    for i in range(N):
        stddev[i] = np.sqrt(C[i, i]) if C[i, i] > 0 else 0.0
    
    for i in range(N):
        for j in range(N):
            denom = stddev[i] * stddev[j]
            if denom == 0.0:
                corr[i, j] = 0.0
            else:
                val = C[i, j] / denom
                corr[i, j] = val if np.isfinite(val) else 0.0
    
    return corr

@jit(nopython=True)
def exp_scaling_squaring(Gamma, m=10):
    """Compute exp(Gamma) using the Scaling and Squaring Method."""
    norm_Gamma = np.linalg.norm(Gamma, ord=1)
    k = max(0, int(np.ceil(np.log2(norm_Gamma))))
    Gamma_scaled = Gamma / (2**k)
    
    N = Gamma.shape[0]
    result = np.eye(N)
    term = np.eye(N)
    
    for i in range(1, m + 1):
        term = np.dot(term, Gamma_scaled) / i
        result += term
    
    for _ in range(k):
        result = np.dot(result, result)
    
    return result

@njit
def adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    """Adam optimizer update step"""
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param_new, m_new, v_new

@njit(parallel=True)
def compute_gradients_all(C, FCemp, FCsim, COVtauemp, COVtausim):
    """Compute all gradients at once for efficiency"""
    N = C.shape[0]
    grad_Ceff_FC = np.zeros_like(C)
    grad_Ceff_COVtau = np.zeros_like(C)
    grad_sigma_FC = np.zeros(N)
    grad_sigma_COVtau = np.zeros(N)
    grad_a_FC = np.zeros(N)
    grad_a_COVtau = np.zeros(N)
    
    for i in prange(N):
        for j in range(N):
            fc_diff = FCemp[i, j] - FCsim[i, j]
            covtau_diff = COVtauemp[i, j] - COVtausim[i, j]
            
            if C[i, j] > 0 or j == N - i - 1:
                grad_Ceff_FC[i, j] = -fc_diff
                grad_Ceff_COVtau[i, j] = -covtau_diff
            
            grad_sigma_FC[i] -= fc_diff
            grad_sigma_COVtau[i] -= covtau_diff
            grad_a_FC[i] -= fc_diff
            grad_a_COVtau[i] -= covtau_diff
    
    return (grad_Ceff_FC, grad_Ceff_COVtau, grad_sigma_FC, 
            grad_sigma_COVtau, grad_a_FC, grad_a_COVtau)

# ==================== Loss Manager ====================
class LossManager:
    """Manages different loss functions and their combinations"""
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        """
        Args:
            loss_weights: Dictionary with keys 'mse_fc', 'mse_covtau', 'corr_fc', 'corr_covtau'
        """
        self.weights = loss_weights or {'mse_fc': 0.5, 'mse_covtau': 0.5}
        self.history = {k: [] for k in self.weights.keys()}
        self.history['total'] = []
        
    def compute_loss(self, FCemp, FCsim, COVtauemp=None, COVtausim=None):
        """Compute weighted total loss"""
        losses = {}
        total_loss = 0
        
        # MSE losses
        if 'mse_fc' in self.weights:
            losses['mse_fc'] = np.mean((FCemp - FCsim) ** 2)
            total_loss += self.weights['mse_fc'] * losses['mse_fc']
            
        if 'mse_covtau' in self.weights and COVtauemp is not None:
            losses['mse_covtau'] = np.mean((COVtauemp - COVtausim) ** 2)
            total_loss += self.weights['mse_covtau'] * losses['mse_covtau']
        
        # Correlation losses (1 - correlation)
        i, j = np.triu_indices_from(FCemp, k=1)
        
        if 'corr_fc' in self.weights:
            corr = np.corrcoef(FCemp[i, j], FCsim[i, j])[0, 1]
            losses['corr_fc'] = 1 - corr if np.isfinite(corr) else 1.0
            total_loss += self.weights['corr_fc'] * losses['corr_fc']
            
        if 'corr_covtau' in self.weights and COVtauemp is not None:
            corr = np.corrcoef(COVtauemp[i, j], COVtausim[i, j])[0, 1]
            losses['corr_covtau'] = 1 - corr if np.isfinite(corr) else 1.0
            total_loss += self.weights['corr_covtau'] * losses['corr_covtau']
        
        return total_loss, losses
    
    def update_history(self, losses):
        """Update loss history"""
        for key, value in losses.items():
            if key in self.history:
                self.history[key].append(value)

# ==================== Optimizer Classes ====================
class Optimizer:
    """Base optimizer with multiple algorithm implementations"""
    
    def __init__(self, method='adam', **kwargs):
        self.method = method
        self.params = {
            'lr_Ceff': kwargs.get('lr_Ceff', 1e-4),
            'lr_sigma': kwargs.get('lr_sigma', 1e-4),
            'lr_a': kwargs.get('lr_a', 5e-4),
            'beta1': kwargs.get('beta1', 0.9),
            'beta2': kwargs.get('beta2', 0.999),
            'epsilon': kwargs.get('epsilon', 1e-8),
            'n_particles': kwargs.get('n_particles', 30),
            'w': kwargs.get('w', 0.7),
            'c1': kwargs.get('c1', 1.5),
            'c2': kwargs.get('c2', 1.5),
        }
        self.reset_state()
    
    def reset_state(self):
        """Reset optimizer state"""
        self.state = {
            'm_Ceff': None, 'v_Ceff': None,
            'm_sigma': None, 'v_sigma': None,
            'm_a': None, 'v_a': None,
            't': 0
        }
        
    def step(self, params, grads, model):
        """Perform optimization step based on method"""
        if self.method == 'adam':
            return self._adam_step(params, grads)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def _adam_step(self, params, grads):
        """Adam optimizer step"""
        Ceff, sigma, a = params
        grad_Ceff, grad_sigma, grad_a = grads
        
        self.state['t'] += 1
        t = self.state['t']
        
        # Initialize states if needed
        if self.state['m_Ceff'] is None:
            self.state['m_Ceff'] = np.zeros_like(Ceff)
            self.state['v_Ceff'] = np.zeros_like(Ceff)
            self.state['m_sigma'] = np.zeros_like(sigma)
            self.state['v_sigma'] = np.zeros_like(sigma)
            self.state['m_a'] = np.zeros_like(a)
            self.state['v_a'] = np.zeros_like(a)
        
        # Update each parameter
        Ceff_new, self.state['m_Ceff'], self.state['v_Ceff'] = adam_update(
            Ceff, grad_Ceff, self.state['m_Ceff'], self.state['v_Ceff'],
            self.params['lr_Ceff'], self.params['beta1'], self.params['beta2'],
            self.params['epsilon'], t
        )
        
        sigma_new, self.state['m_sigma'], self.state['v_sigma'] = adam_update(
            sigma, grad_sigma, self.state['m_sigma'], self.state['v_sigma'],
            self.params['lr_sigma'], self.params['beta1'], self.params['beta2'],
            self.params['epsilon'], t
        )
        
        a_new, self.state['m_a'], self.state['v_a'] = adam_update(
            a, grad_a, self.state['m_a'], self.state['v_a'],
            self.params['lr_a'], self.params['beta1'], self.params['beta2'],
            self.params['epsilon'], t
        )
        
        return Ceff_new, sigma_new, a_new


# ==================== Main Linear Hopf Model ====================
class LinearHopfModel:
    """Main Linear Hopf Model class"""
    
    def __init__(self, 
                 C: np.ndarray, 
                 f_diff: np.ndarray,
                 sigma: Union[float, np.ndarray] = 0.5, 
                 a: Union[float, np.ndarray] = -0.02, 
                 **kwargs):
        """
        Initialize Linear Hopf Model
        
        Args:
            C: Structural connectivity matrix
            f_diff: Frequency differences for each node
            TR: Repetition time
            sigma: Noise (scalar or vector)
            a: Bifurcation parameter (scalar or vector)
            g: Global coupling strength
            tau: Time lag for COVtau computation
        """
        self.C = C
        self.f_diff = f_diff
        self.n_parcels = C.shape[0]
        self.params = {
            'competitive_coupling': kwargs.get('competitive_coupling', False),
            'allow_negative': kwargs.get('allow_negative', False),
            'Ceff_norm': kwargs.get('Ceff_norm', True),
            'max_C': kwargs.get('max_C', 1.0),
            'max_iter': kwargs.get('max_iter', 10000),
            'patience': kwargs.get('patience', 4),
            'iter_check': kwargs.get('iter_check', 50),
            'verbose': kwargs.get('verbose', True),
            'loss_weights': kwargs.get('loss_weights', {'mse_fc': 0.25, 'mse_covtau': 0.25, 'corr_fc': 0.25, 'corr_covtau': 0.25}),
            'fit_sigma': kwargs.get('fit_sigma', False),
            'fit_a': kwargs.get('fit_a', False),
            'TR': kwargs.get('TR', 3.0),
            'tau': kwargs.get('tau', 1),
            'g': kwargs.get('g', 1.0),
        }
        
        # Initial values
        self.Ceff = self.params['g'] * C
        self.sigma = np.ones(self.n_parcels) * sigma if np.isscalar(sigma) else np.array(sigma)
        self.a = np.ones(self.n_parcels) * a if np.isscalar(a) else np.array(a)
        
        # Determine what to fit based on input
        self.fit_sigma = not np.isscalar(sigma) or self.params['fit_sigma']
        self.fit_a = not np.isscalar(a) or self.params['fit_a']
        
        # Initialize loss manager and optimizer later
        self.loss_manager = None
        self.optimizer = None
        
        # Store empirical data
        self.FCemp = None
        self.COVemp = None
        self.COVtauemp = None
        
    def setup_optimization(self, optimizer_method='adam', loss_weights=None, **kwargs):
        """Setup optimizer and loss manager"""
        self.loss_manager = LossManager(loss_weights)
        self.optimizer = Optimizer(optimizer_method, **kwargs)
        
    def fit(self, tsdata: np.ndarray):
        """
        Fit the model to empirical data
        
        Args:
            tsdata: Time series data [NSUB, NPARCELS, time] or [NPARCELS, time]
            max_iter: Maximum iterations
            patience: Early stopping patience
            iter_check: Check interval
            verbose: Print progress
        """
        # Ensure loss manager and optimizer are set up
        if self.loss_manager is None:
            self.setup_optimization()
        
        # Compute empirical data
        self._process_empirical_data(tsdata)
        
        # Initialize parameters
        Ceff_current = self.Ceff.copy()
        sigma_current = self.sigma.copy()
        a_current = self.a.copy()
        
        # Tracking
        best_error = np.inf
        best_params = (Ceff_current.copy(), sigma_current.copy(), a_current.copy())
        no_improvement = 0
        
        # Main optimization loop
        for iteration in range(1, self.params['max_iter'] + 1):
            # Forward pass
            FCsim, COVsim, COVsimtotal, A = self._hopf_int(Ceff_current, sigma_current, a_current)
            COVtausim = self._compute_covtau(COVsimtotal, A, COVsim)
            
            # Compute loss and gradients
            if iteration % self.params['iter_check'] == 0:
                error, losses = self.loss_manager.compute_loss(
                    self.FCemp, FCsim, self.COVtauemp, COVtausim
                )
                self.loss_manager.update_history(losses)
                self.loss_manager.history['total'].append(error)
                
                if self.params['verbose']:
                    i, j = np.triu_indices_from(self.FCemp, k=1)
                    corr_fc = np.corrcoef(self.FCemp[i, j], FCsim[i, j])[0, 1]
                    corr_cov = np.corrcoef(self.COVtauemp[i, j], COVtausim[i, j])[0, 1]

                # Early stopping
                if error < best_error:
                    best_error = error
                    best_losses = losses
                    best_params = (Ceff_current.copy(), sigma_current.copy(), a_current.copy())
                    if self.params['verbose']: best_err = (corr_fc,corr_cov)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement > self.params['patience']:
                        if self.params['verbose']:
                            corr_fc, corr_cov = best_err
                            print(f"Early stopping: no improvement for {self.params['patience']} checks")
                            print(f"error={error:.3f}, CorrFC={corr_fc:.3f}, CorrCOV={corr_cov:.3f}")
                            print(f"losses: {best_losses}")
                        break
            
            # Compute gradients
            grads = compute_gradients_all(self.C, self.FCemp, FCsim, 
                                         self.COVtauemp, COVtausim)
            grad_Ceff = grads[0] + grads[1]
            grad_sigma = grads[2] + grads[3] if self.fit_sigma else np.zeros_like(sigma_current)
            grad_a = grads[4] + grads[5] if self.fit_a else np.zeros_like(a_current)
            
            # Optimization step
            Ceff_new, sigma_new, a_new = self.optimizer.step(
                (Ceff_current, sigma_current, a_current),
                (grad_Ceff, grad_sigma, grad_a),
                self
            )
            
            # Apply constraints
            Ceff_new = self._apply_Ceff_constraints(Ceff_new)
            if not self.params['allow_negative']:
                sigma_new = np.maximum(sigma_new, 0.01)
            a_new = np.clip(a_new, -0.5, -0.001)
            
            # Update parameters
            Ceff_current = Ceff_new
            sigma_current = sigma_new
            a_current = a_new
        
        # Store best parameters
        self.Ceff, self.sigma, self.a = best_params
        
        # Final simulation
        self.FCsim, _, _, _ = self._hopf_int(self.Ceff, self.sigma, self.a)
        results = (*best_params, best_losses)
        
        return results
    
    def _process_empirical_data(self, tsdata):
        """Process empirical time series to compute FC, COV, COVtau (can potentially be used for multiple subjects)"""
        if tsdata.ndim == 2:
            tsdata = tsdata[np.newaxis, :, :]
        
        NSUB = tsdata.shape[0]
        FC_all = []
        COV_all = []
        COVtau_all = []
        
        for sub in range(NSUB):
            ts = tsdata[sub, :, 5:-5]  # Remove edges
            
            # FC and COV
            FC = np.corrcoef(ts)
            COV = np.cov(ts)
            FC_all.append(FC)
            COV_all.append(COV)
            
            # COVtau
            COVtau = self._compute_empirical_covtau(ts.T, COV)
            COVtau_all.append(COVtau)
        
        self.FCemp = np.mean(FC_all, axis=0)
        self.COVemp = np.mean(COV_all, axis=0)
        self.COVtauemp = np.mean(COVtau_all, axis=0)
    
    def _compute_empirical_covtau(self, ts, COV):
        """Compute empirical COV(tau)"""
        N = self.n_parcels
        COVtau = np.zeros((N, N))
        sigratio = compute_sigratio_from_cov(COV)
        
        for i in range(N):
            for j in range(N):
                clag = np.correlate(ts[:, i] - ts[:, i].mean(), 
                                  ts[:, j] - ts[:, j].mean(), 'full')
                lags = np.arange(-len(ts) + 1, len(ts))
                valid = (lags >= -self.params['tau']) & (lags <= self.params['tau'])
                clag = clag[valid]
                lags = lags[valid]
                
                idx = np.where(lags == self.params['tau'])[0][0]
                COVtau[i, j] = clag[idx] / len(ts)
        
        COVtau *= sigratio
        return COVtau
    
    def _hopf_int(self, Ceff, sigma, a):
        """Hopf step with diagnostics"""
        N = self.n_parcels
        wo = self.f_diff * (2 * np.pi)
        
        # Build Jacobian
        s = np.sum(Ceff, axis=1)
        Axx = np.diag(a) - np.diag(s) + Ceff
        Ayy = Axx.copy()
        Ayx = np.diag(wo)
        Axy = -Ayx.copy()
        A = np.block([[Axx, Axy], [Ayx, Ayy]])
        
        eigvals = np.linalg.eigvals(A)
        max_real = np.max(eigvals.real)
        # Check if system is stable
        if max_real > -0.001:
            print("WARNING: System is unstable or marginally stable!")
            print("This will cause Sylvester equation to fail")
        
        # Noise covariance
        Qn = np.diag(np.concatenate([sigma**2, sigma**2]))
        
        # Solve Sylvester equation
        try:
            Cvth = solve_sylvester(A, A.T, -Qn)
            
            # Check for numerical issues
            if not np.all(np.isfinite(Cvth)):
                print("ERROR: Non-finite values in Cvth")
                n_nan = np.sum(np.isnan(Cvth))
                n_inf = np.sum(np.isinf(Cvth))
                print(f"NaN count: {n_nan}, Inf count: {n_inf}")
            
            # Check if Cvth is positive definite
            eigvals_cvth = np.linalg.eigvals(Cvth)
            min_eig_cvth = np.min(eigvals_cvth.real)
            # print(f"Cvth min eigenvalue: {min_eig_cvth:.6f}")
            if min_eig_cvth < 0:
                print(f"WARNING: Cvth is not positive definite {np.sum(eigvals_cvth.real < 0)} negative eigenvalues")
            
        except Exception as e:
            print(f"ERROR solving Sylvester equation: {e}")
            # Return dummy values to continue
            Cvth = np.eye(2*N) * 0.01
        
        # Extract FC and COV
        FCth = corrcov_py_numba(Cvth)
        FC = FCth[:N, :N]
        COV = Cvth[:N, :N]

        fc_triu = FC[np.triu_indices_from(FC, k=1)]
        
        if np.std(fc_triu) < 0.01:
            print("PROBLEM: FC is nearly uniform!")
        
        return FC, COV, Cvth, A

    def _compute_covtau(self, COVsimtotal, A, COVsim):
        """Compute COV(tau) for simulation"""
        N = self.n_parcels
        COVtausim = (exp_scaling_squaring((self.params['tau'] * self.params['TR']) * A) @ COVsimtotal)[:N, :N]
        sigratio = compute_sigratio_from_cov(COVsim)
        COVtausim *= sigratio
        return COVtausim
    

    def _apply_Ceff_constraints(self, Ceff):
        """Apply constraints matching the working dataset"""
        Ceff = np.array(Ceff, dtype=float, copy=True)
        N = self.n_parcels

        # Structural mask
        mask = (self.C > 0)
        np.fill_diagonal(mask, False)

        # Zero where no structure
        Ceff[~mask] = 0.0
        
        # Clip individual values (like old dataset: max ~0.013)
        if not self.params['allow_negative']:
            Ceff = np.clip(Ceff, 0.0, 0.2)
        else:
            Ceff = np.clip(Ceff, -0.02, 0.02)

        # Zero diagonal
        np.fill_diagonal(Ceff, 0.0)

        # Normalize row sums
        target_max_rowsum = self.params.get('target_row_sum', 10)
        row_sums = np.sum(np.abs(Ceff), axis=1)
        for i in range(N):
            if row_sums[i] > target_max_rowsum:
                Ceff[i, :] *= (target_max_rowsum / row_sums[i])

        return Ceff

def _compute_empirical_covtau2(ts, COV):
        """Compute empirical COV(tau)"""
        N = COV.shape[0]
        COVtau = np.zeros((N, N))
        sigratio = compute_sigratio_from_cov(COV)
        
        for i in range(N):
            for j in range(N):
                clag = np.correlate(ts[:, i] - ts[:, i].mean(), 
                                  ts[:, j] - ts[:, j].mean(), 'full')
                lags = np.arange(-len(ts) + 1, len(ts))
                valid = (lags >= -1.0) & (lags <= 1.0)
                clag = clag[valid]
                lags = lags[valid]
                
                idx = np.where(lags == 1.0)[0][0]
                COVtau[i, j] = clag[idx] / len(ts)
        
        COVtau *= sigratio
        return COVtau

def fit_linhopf(data, ts_zsc, sigma_ini, a_ini, verbose, params, NPARCELLS):
    """Fit model for one subject"""
    # retrieve parameters and data
    f_diff = data['f_diff'][:NPARCELLS]
    ts_zsc = data['ts'] if ts_zsc is None else ts_zsc
    hopf_params = params['hopfParamsAdam'].copy()
    hopf_params['verbose'] = verbose
    if ts_zsc.ndim == 2:
        ts_zsc = ts_zsc[np.newaxis, :, :]
    ts = ts_zsc[0, :, 5:-5] # remove edges

    # compute initial EC in case of no SC provided (uses FC and COVtau)
    FC = np.corrcoef(ts)
    COV = np.cov(ts)
    COVtau = _compute_empirical_covtau2(ts.T, COV)
    FCCOVtau = 0.5*(FC + COVtau)
    np.fill_diagonal(FCCOVtau, 0.0)
    SC_ini = 0.05*(FCCOVtau - FCCOVtau.min()) / (FCCOVtau.max() - FCCOVtau.min()) # normalize
    np.fill_diagonal(SC_ini, 0.0)

    SC = SC_ini

    # in case SC is provided:
    # SC = data['SC'][:NPARCELLS, :NPARCELLS]

    # initialize model
    model = LinearHopfModel(C=SC, f_diff=f_diff, sigma=sigma_ini, a=a_ini, **hopf_params)

    # choose optimizer method and provide fit parameters
    model.setup_optimization(optimizer_method='adam', **hopf_params)

    Ceff, sigma, a, losses = model.fit(ts_zsc)
    
    # Get all empirical quantities
    FCemp = model.FCemp
    COVemp = model.COVemp
    COVtauemp = model.COVtauemp
    
    # Get all simulated quantities (recompute final state)
    FCsim, COVsim, COVsimtotal, A = model._hopf_int(Ceff, sigma, a)
    COVtausim = model._compute_covtau(COVsimtotal, A, COVsim)
    
    return {
        # Parameters
        'Ceff': Ceff,
        'sigma': sigma,
        'a': a,
        'SC': SC,
        'f_diff': f_diff,
        
        # Empirical quantities
        'FCemp': FCemp,
        'COVemp': COVemp,
        'COVtauemp': COVtauemp,
        
        # Simulated quantities
        'FCsim': FCsim,
        'COVsim': COVsim,
        'COVtausim': COVtausim,
        
        # Jacobian (system dynamics matrix)
        'A': A,
        
        # Loss information
        'losses': losses
    }