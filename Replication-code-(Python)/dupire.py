import numpy as np 

def dupire(prices, t, k, K):
    """ 
    Computing the local volatility grid for the Dupire Volatility 
    Adjustments in the end to extrapolate over the areas that couldn't be computed
    """
    
    dC_dT = np.gradient(prices, t, axis=0)  # Time derivative
    dC_dT = np.where(dC_dT < 0, 1e-10, dC_dT) # can't be 0 given Dupire formula

    
    dC_dK = np.gradient(prices, k, axis=1)  # First strike derivative
    d2C_dK2 = np.gradient(dC_dK, k, axis=1) # Second strike derivative

    local_vol = np.sqrt((dC_dT) / (0.5 * K**2 * d2C_dK2))

    return local_vol