import numpy as np 
from numba import jit

def simulate_correlated_brownian(Par):

    rho_sto = Par['rho_sto'] 
    nb_X = Par['nb_X'] 
    nb_Tsim = Par['nb_T_sim'] 
    delta_t = Par['delta_t'] 
    
    # Correlation matrix
    corr_matrix = np.array([[1, rho_sto], [rho_sto, 1]])
    
    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated standard normal variables
    Z = np.random.normal(size=(nb_X, nb_Tsim+1, 2))
    
    # Apply Cholesky decomposition to get correlated normal variables
    correlated_Z = np.dot(Z, L.T)
    
    # Scale by sqrt(delta_t) to get correlated Brownian increments
    dW_x = correlated_Z[:, :, 0] * np.sqrt(delta_t)
    dW_v = correlated_Z[:, :, 1] * np.sqrt(delta_t)
    
    return dW_x, dW_v



def initialize_processes(Par):

    nb_X = Par['nb_X'] 
    nb_Tsim = Par['nb_T_sim'] 
    delta_t = Par['delta_t'] 
    
    X_0 = Par['X_0'] 
    v_0 = Par['Y_0']
    kappa = Par['lambda']
    theta = Par['mu']
    zeta = Par['eta']
    cap = Par['cap']

    dW_x, dW_v = simulate_correlated_brownian(Par)
    
    # Stock price
    X_ = np.ones([nb_X, nb_Tsim])*X_0

    # Variance process
    v_ = np.ones([nb_X, nb_Tsim])*v_0 
    v_[:, 0][v_[:, 0] < cap] = cap

    for i in range(1, nb_Tsim):
        v_[:, i] = np.nan_to_num(v_[:, i-1] + kappa * (theta - v_[:, i-1]) * delta_t + zeta * np.sqrt(v_[:, i-1]) * dW_v[:, i], nan=cap)
        v_[:, i][v_[:, i] < cap] = cap

    return X_, v_, dW_x, dW_v



def compute_percentiles(L, nb_X):
    # l_ind = np.random.choice(np.arange(nb_X), size=L, replace=False).astype(int)
    # l_ind.sort()

    # Calculate the percentiles for Z_j
    percentiles = np.arange(1, L+1) * (L / (L+1))

    return percentiles



def compute_beta(X_, v_, t, L, kernel, lambda_par, sigma_f, Par, method="Ridge"):
    """
    Computes beta_hat as well as the distribution, using the ridge regression with regularization parameter lambda_par
    """

    cap = Par['cap']
    nb_X = Par['nb_X']
    
    ## Choosing L << N  
    if L > 0:
        l_ind = compute_percentiles(L, nb_X)
    else:
        l_ind = np.linspace(0, nb_X-1, nb_X).astype(int)
        L = nb_X

    
    ###### Computing $\hat{\beta}$ 

    Z_j = np.percentile(X_[:, t-1:t], l_ind)
    Z_j = Z_j.reshape(L, 1)
    
    # K(k(Zj, X^n)) 
    K = kernel(X_[:, t-1:t], Z_j, sigma_f)
    
    # R(k(Zj, Zl)) 
    R = kernel(Z_j, Z_j, sigma_f)
    
    # G(A(Y_t^n))
    G = v_[:,t]

    ## Computes beta_hat
    if method == "Ridge":
        temp = np.dot(K.T, K) + lambda_par * nb_X * R
        beta_hat = np.linalg.solve(temp, np.dot(K.T, G))
    
        ## Computes the distribution
        m_A_lam = 0 
    
        for j, i in zip(range(L), Z_j): 
            zz = Z_j[j, 0]
            m_A_lam = m_A_lam + beta_hat[j] * kernel(X_[:, t-1:t], np.array([[zz]])) 
        m_A_lam[m_A_lam < cap] = cap

    elif method == "GPR":
        0
    
    return beta_hat, m_A_lam



@jit(nopython=False)
def simulate_diffusion(calib_Par, sim_Par, Par):
    # Update the parameters for the micro simulation
    Par.update(sim_Par)

    
    X_, v_, dW_x, dW_v = initialize_processes(PARAMETERS)
    
    cond = np.zeros([Par['nb_T_sim'], Par['nb_X']]) 
    L = 100
    sigma_f = 0.1
    
    t = 1
    X_[:, t] = X_[:, t-1] +  np.sqrt(Par['Y_0']) * X_[:, t-1] * sigma_dupire((Par['delta_t'], X_[:, t-1])) / np.sqrt(Par['Y_0']) * dW_x[:, t]

    for t in range(2, Par['nb_T_sim']):
        # period 
        beta_hat, m_A_lam = compute_beta(X_, v_, t, L, kernel, lambda_par, sigma_f, Par)
    
        ## Update the memory for the graphs later
        cond[t, :] = m_A_lam.ravel()
        
        # Update step 
        Sig = sigma_dupire((tgrid_sim[t], X_[:, t-1]))
        ## Only forst the initial iterations, given that Dupire's formula isn't great for extra short term 
        
        if np.isnan(Sig).any():
            indices = np.arange(len(Sig))
            # Mask for non-NaN values
            valid_mask = ~np.isnan(Sig)
            # Interpolate the NaN values
            interp_func = interp1d(indices[valid_mask], Sig[valid_mask], kind='linear', fill_value="extrapolate")
            Sig = interp_func(indices)
            
        X_[:, t] = X_[:, t-1] +  np.sqrt(v_[:,t-1]) * X_[:, t-1] * Sig / np.sqrt(m_A_lam.ravel()) * dW_x[:, t]
    
        # Bounds for the interpolation
        X_[:, t][X_[:, t] > Kmax] = Kmax
        X_[:, t][X_[:, t] < Kmin] = Kmin
    
        print(f"\rSteps completed: {t}/{nb_T_sim}", end='', flush=True)

    return X_

def compute_price():
    return 0


def implied_vol_RMSE():
    
    return 0