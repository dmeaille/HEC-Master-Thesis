import numpy as np
from numba import njit
import scipy.stats as st
from scipy.stats import norm

### Hestong with Fourier Pricing
@njit
def Heston(S0, K, T, r, kappa, theta, rho, zeta, v0, opt_type, N = 1_012, z = 24, train=True):
    
    def heston_char(u): 
        t0 = 0.0 ;  q = 0.0
        m = np.log(S0) + (r - q)*(T-t0)
        D = np.sqrt((rho*zeta*1j*u - kappa)**2 + zeta**2*(1j*u + u**2))
        C = (kappa - rho*zeta*1j*u - D) / (kappa - rho*zeta*1j*u + D)
        beta = ((kappa - rho*zeta*1j*u - D)*(1-np.exp(-D*(T-t0)))) / (zeta**2*(1-C*np.exp(-D*(T-t0))))
        alpha = ((kappa*theta)/(zeta**2))*((kappa - rho*zeta*1j*u - D)*(T-t0) - 2*np.log((1-C*np.exp(-D*(T-t0))/(1-C))))
        return np.exp(1j*u*m + alpha + beta*v0)

    # # Parameters for the Function to make sure the approximations are correct.
    c1 = np.log(S0) + r*T - .5*theta*T
    c2 = theta/(8*kappa**3)*(-zeta**2*np.exp(-2*kappa*T) + 4*zeta*np.exp(-kappa*T)*(zeta-2*kappa*rho) 
        + 2*kappa*T*(4*kappa**2 + zeta**2 - 4*kappa*zeta*rho) + zeta*(8*kappa*rho - 3*zeta))
    a = c1 - z*np.sqrt(np.abs(c2))
    b = c1 + z*np.sqrt(np.abs(c2))
    
    h       = lambda n : (n*np.pi) / (b-a) 
    g_n     = lambda n : (np.exp(a) - (K/h(n))*np.sin(h(n)*(a - np.log(K))) - K*np.cos(h(n)*(a - np.log(K)))) / (1 + h(n)**2)
    g0      = K*(np.log(K) - a - 1) + np.exp(a)
    
    F = g0 
    for n in range(1, N+1):
        h_n = h(n)
        F += 2*heston_char(h_n) * np.exp(-1j*a*h_n) * g_n(n)

    F = np.exp(-r*T)/(b-a) * np.real(F)
    F = F if opt_type == -1 else F + S0 - K*np.exp(-r*T)
    return F if F > 0 else 0


HS = np.vectorize(Heston)




def heston(Par, train=True):
    S0=Par['S0']
    r=Par['r']
    v0=Par['v0']      
    kappa=Par['kappa']
    theta=Par['theta'] 
    zeta=Par['zeta']
    rho=Par['rho']
    opt_type=Par['opt_type']
    N=Par['N']
    z=Par['z']

    if train:
        K=Par['K_train']
        T=Par['T_train']
    else:
        K=Par['K_sim']
        T=Par['T_sim']

    return HS(S0, K, T, r, kappa, theta, rho, zeta, v0, opt_type, N, z)



### Black and Scholes pricing



def bs(Par, train=True):

    opt_type=Par['opt_type']
    S0=Par['S0']
    r=Par['r']
    sigma=Par['sigma']
    div=0

    if train:
        K=Par['K_train']
        T=Par['T_train']
    else:
        K=Par['K_sim']
        T=Par['T_sim']
    
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if opt_type == 1:
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return price