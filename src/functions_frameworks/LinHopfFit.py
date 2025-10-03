import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_sylvester
from scipy.signal import correlate
from scipy.optimize import minimize, differential_evolution
from scipy import stats
from numba import njit, prange, jit
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

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
            loss_weights: Dictionary with keys 'mse_fc', 'mse_covtau', 'corr_fc', 'corr_covtau', 'ks'
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
        
        # KS test
        if 'ks' in self.weights:
            ks_stat, _ = stats.ks_2samp(FCemp[i, j], FCsim[i, j])
            losses['ks'] = ks_stat
            total_loss += self.weights['ks'] * losses['ks']
        
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
        elif self.method == 'gradient_descent':
            return self._gd_step(params, grads)
        elif self.method == 'lbfgs':
            return self._lbfgs_step(params, grads, model)
        elif self.method == 'particle_swarm':
            return self._pso_step(params, model)
        elif self.method == 'differential_evolution':
            return self._de_step(params, model)
        elif self.method == 'hybrid_swarm_adam':
            # Use PSO for first half, then switch to Adam
            if self.state['t'] < 500:
                return self._pso_step(params, model)
            else:
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
    
    def _gd_step(self, params, grads):
        """Gradient descent step"""
        Ceff, sigma, a = params
        grad_Ceff, grad_sigma, grad_a = grads
        
        Ceff_new = Ceff - self.params['lr_Ceff'] * grad_Ceff
        sigma_new = sigma - self.params['lr_sigma'] * grad_sigma
        a_new = a - self.params['lr_a'] * grad_a
        
        return Ceff_new, sigma_new, a_new
    
    def _lbfgs_step(self, params, grads, model):
        """L-BFGS step using scipy"""
        Ceff, sigma, a = params
        
        def objective(x):
            # Unpack parameters
            n = model.n_parcels
            Ceff_flat = x[:n*n]
            sigma_vals = x[n*n:n*n+n]
            a_vals = x[n*n+n:]
            
            Ceff_test = Ceff_flat.reshape(n, n)
            # Apply constraints
            Ceff_test = model._apply_Ceff_constraints(Ceff_test)
            
            # Compute loss
            FCsim, COVsim, COVsimtotal, A = model._hopf_int(Ceff_test, sigma_vals, a_vals)
            COVtausim = model._compute_covtau(COVsimtotal, A, COVsim)
            
            loss, _ = model.loss_manager.compute_loss(
                model.FCemp, FCsim, model.COVtauemp, COVtausim
            )
            return loss
        
        # Pack current parameters
        x0 = np.concatenate([Ceff.flatten(), sigma, a])
        
        # Run L-BFGS
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 4})
        
        # Unpack results
        n = model.n_parcels
        Ceff_new = result.x[:n*n].reshape(n, n)
        sigma_new = result.x[n*n:n*n+n]
        a_new = result.x[n*n+n:]
        
        return Ceff_new, sigma_new, a_new
    
    def _pso_step(self, params, model):
        """Particle Swarm Optimization step"""
        if 'particles' not in self.state:
            self._init_pso(params, model)
        
        particles = self.state['particles']
        velocities = self.state['velocities']
        pbest = self.state['pbest']
        pbest_scores = self.state['pbest_scores']
        gbest = self.state['gbest']
        
        # Update each particle
        for i in range(self.params['n_particles']):
            # Update velocity
            r1, r2 = np.random.random(2)
            velocities[i] = (self.params['w'] * velocities[i] +
                           self.params['c1'] * r1 * (pbest[i] - particles[i]) +
                           self.params['c2'] * r2 * (gbest - particles[i]))
            
            # Update position
            particles[i] += velocities[i]
            
            # Evaluate
            Ceff_test, sigma_test, a_test = self._unpack_particle(particles[i], model)
            FCsim, COVsim, COVsimtotal, A = model._hopf_int(Ceff_test, sigma_test, a_test)
            COVtausim = model._compute_covtau(COVsimtotal, A, COVsim)
            
            score, _ = model.loss_manager.compute_loss(
                model.FCemp, FCsim, model.COVtauemp, COVtausim
            )
            
            # Update personal best
            if score < pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
                
                # Update global best
                if score < self.state['gbest_score']:
                    self.state['gbest'] = particles[i].copy()
                    self.state['gbest_score'] = score
        
        # Return global best
        return self._unpack_particle(self.state['gbest'], model)
    
    def _init_pso(self, params, model):
        """Initialize PSO particles"""
        Ceff, sigma, a = params
        n_particles = self.params['n_particles']
        
        # Initialize particles around current solution
        particles = []
        for _ in range(n_particles):
            particle = np.concatenate([
                Ceff.flatten() + np.random.randn(*Ceff.shape).flatten() * 0.01,
                sigma + np.random.randn(*sigma.shape) * 0.01,
                a + np.random.randn(*a.shape) * 0.001
            ])
            particles.append(particle)
        
        self.state['particles'] = np.array(particles)
        self.state['velocities'] = np.random.randn(n_particles, len(particles[0])) * 0.001
        self.state['pbest'] = self.state['particles'].copy()
        self.state['pbest_scores'] = np.inf * np.ones(n_particles)
        self.state['gbest'] = particles[0]
        self.state['gbest_score'] = np.inf
    
    def _unpack_particle(self, particle, model):
        """Unpack particle into parameters"""
        n = model.n_parcels
        Ceff = particle[:n*n].reshape(n, n)
        sigma = particle[n*n:n*n+n]
        a = particle[n*n+n:]
        return Ceff, sigma, a
    
    def _de_step(self, params, model):
        """Differential Evolution step"""
        Ceff, sigma, a = params
        n = model.n_parcels

        def objective(x):
            Ceff_test = x[:n*n].reshape(n, n)
            sigma_test = x[n*n:n*n+n]
            a_test = x[n*n+n:]

            Ceff_test = model._apply_Ceff_constraints(Ceff_test)
            FCsim, COVsim, COVsimtotal, A = model._hopf_int(Ceff_test, sigma_test, a_test)
            COVtausim = model._compute_covtau(COVsimtotal, A, COVsim)

            loss, _ = model.loss_manager.compute_loss(
                model.FCemp, FCsim, model.COVtauemp, COVtausim
            )
            return loss

        # Pack parameters
        x0 = np.concatenate([Ceff.flatten(), sigma, a])

        # Define bounds
        bounds = []
        bounds.extend([(0, 0.5)] * (n*n))       # Ceff bounds
        bounds.extend([(0.01, 0.5)] * n)        # sigma bounds
        bounds.extend([(-0.5, -0.001)] * n)     # a bounds

        # Option 1: Let DE generate its own population (recommended)
        result = differential_evolution(objective, bounds, maxiter=1, workers=1)

        # --- Optional: initialize population around x0 (uncomment if needed) ---
        # D = len(x0)
        # S = max(10, 2*D)
        # init_pop = np.tile(x0, (S, 1)) + 0.01*np.random.randn(S, D)
        # for i, (low, high) in enumerate(bounds):
        #     init_pop[:, i] = np.clip(init_pop[:, i], low, high)
        # result = differential_evolution(objective, bounds, maxiter=1, init=init_pop, workers=1)

        # Unpack results
        Ceff_new = result.x[:n*n].reshape(n, n)
        sigma_new = result.x[n*n:n*n+n]
        a_new = result.x[n*n+n:]

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
            'max_C': kwargs.get('max_C', 0.2),
            'max_iter': kwargs.get('max_iter', 10000),
            'patience': kwargs.get('patience', 4),
            'iter_check': kwargs.get('iter_check', 50),
            'verbose': kwargs.get('verbose', True),
            'loss_weights': kwargs.get('loss_weights', {'mse_fc': 0.5, 'mse_covtau': 0.5}),
            'fit_sigma': kwargs.get('fit_sigma', False),
            'fit_a': kwargs.get('fit_a', False),
            'TR': kwargs.get('TR', 2.0),
            'tau': kwargs.get('tau', 1),
            'g': kwargs.get('g', 1.0),
        }
        
        # Initialize parameters
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
        
        # Process empirical data
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
        """Process empirical time series to compute FC, COV, COVtau"""
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
        """Core Hopf integration"""
        N = self.n_parcels
        wo = self.f_diff * (2 * np.pi)
        
        # Build Jacobian
        s = np.sum(Ceff, axis=1)
        Axx = np.diag(a) - np.diag(s) + Ceff
        Ayy = Axx.copy()
        Ayx = np.diag(wo)
        Axy = -Ayx.copy()
        
        A = np.block([[Axx, Axy], [Ayx, Ayy]])
        
        # Noise covariance
        Qn = np.diag(np.concatenate([sigma**2, sigma**2]))
        
        # Solve Sylvester equation
        Cvth = solve_sylvester(A, A.T, -Qn)
        
        # Extract FC and COV
        FCth = corrcov_py_numba(Cvth)
        FC = FCth[:N, :N]
        COV = Cvth[:N, :N]
        
        return FC, COV, Cvth, A
    
    def _compute_covtau(self, COVsimtotal, A, COVsim):
        """Compute COV(tau) for simulation"""
        N = self.n_parcels
        COVtausim = (exp_scaling_squaring((self.params['tau'] * self.params['TR']) * A) @ COVsimtotal)[:N, :N]
        sigratio = compute_sigratio_from_cov(COVsim)
        COVtausim *= sigratio
        return COVtausim
    
    def _apply_Ceff_constraints(self, Ceff):
        """Apply structural / sign / normalization constraints to effective connectivity.
        
        Rules:
        - If competitive_coupling is False: only allow (non‑negative unless allow_negative=True)
          weights where structural C > 0 (keep diagonal at 0). Elsewhere force 0.
        - If competitive_coupling is True: allow positive and negative values on existing
          structural links (still zero where no structural connection). Rows can be
          mean-centered (excluding diagonal) to encourage competition.
        - Always symmetrize to keep the matrix consistent with undirected SC.
        - If Ceff_norm is True: scale globally so max |Ceff| <= self.max_C.
        """
        Ceff = np.array(Ceff, dtype=float, copy=True)
        N = self.n_parcels

        # Structural mask (allow on structural links only)
        mask = (self.C > 0)
        # Always keep diagonal controllable separately (set to zero later)
        np.fill_diagonal(mask, False)

        if not self.params['competitive_coupling']:
            # Zero where no SC
            Ceff[~mask] = 0.0
            # Clip sign
            if self.params['allow_negative']:
                Ceff = np.clip(Ceff, -self.params['max_C'], self.params['max_C'])
            else:
                Ceff = np.clip(Ceff, 0.0, self.params['max_C'])
        else:
            # Competitive: allow +/- on existing links, zero elsewhere
            Ceff[~mask] = 0.0
            # Optional row mean-centering (excluding diagonal) to promote competition
            for i in range(N):
                row_mask = mask[i]
                if np.any(row_mask):
                    mean_val = Ceff[i, row_mask].mean()
                    Ceff[i, row_mask] -= mean_val
            # Clip
            Ceff = np.clip(Ceff, -self.params['max_C'], self.params['max_C'])

        # Symmetrize (assume undirected SC)
        Ceff = 0.5 * (Ceff + Ceff.T)

        # Enforce zero diagonal
        np.fill_diagonal(Ceff, 0.0)

        # Global normalization if requested
        if self.params['Ceff_norm']:
            max_abs = np.max(np.abs(Ceff))
            if max_abs > self.params['max_C'] and max_abs > 0:
                Ceff *= (self.params['max_C'] / max_abs)

        return Ceff


def fit_linhopf(data, ts_zsc, sigma_ini, a_ini, verbose, params, NPARCELLS):
    """Fit model for one subject"""
    SC = data['SC'][:NPARCELLS, :NPARCELLS]
    f_diff = data['f_diff'][:NPARCELLS]
    ts_zsc = data['ts'] if ts_zsc is None else ts_zsc
    hopf_params = params['hopfParamsAdam'].copy()
    hopf_params['verbose'] = verbose

    model = LinearHopfModel(
        C=SC, f_diff=f_diff, sigma=sigma_ini, a=a_ini, **hopf_params
    )
    model.setup_optimization(optimizer_method='adam', **hopf_params)
    Ceff, sigma, a, losses = model.fit(ts_zsc)
    
    return {
        'Ceff': Ceff,
        'sigma': sigma,
        'a': a,
        'losses': losses,
    }