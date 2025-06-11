import numpy as np 
from pricing import bs

## With Newton-Raphson

def compute_implied_vol_gradient(price, cp, s, k, r, t, div=0.0, guess=0.5):
    """
    Finds the implied volatility using the Gradient Descent Algorithm.

    Args:
        price: Market price of the option
        cp: Option type (+1 for call, -1 for put)
        s: Current stock price (array or scalar)
        k: Strike price
        r: Risk-free rate
        t: Time to maturity
        div: Dividend yield (default 0.0)

    Returns:
        v_imp: Implied volatility
        price_imp: Option price using the found v_imp
    """

    # Upper bound: to avoid explosion (nan or inf) for very short-term maturity and very low strikes
    # may be adjusted upward if needed
    max_V = 4
    # Initial guess
    v_imp = np.ones_like(k) * guess  # Medium Guess (25% vol)
    h = 1e-9
    
    err = 1
    
    while err > 1e-6:

        # new test points
        price_imp = bs(cp, s, k, r, t, v_imp, div)[0]
        price_imp_h = bs(cp, s, k, r, t, v_imp+h, div)[0]
        
        # f_x
        f_x = price_imp - price 
        # f_prime 
        f_prime = (price_imp_h - price - f_x)/h

        # updating the error: how far away from 0 in the worst case
        err = np.max(np.max(np.abs(f_x)))

        # updating the guess
        v_imp = v_imp - f_x/f_prime

        v_imp[v_imp > max_V] = max_V

        print(err)

    return v_imp




## With Bisection

def compute_implied_vol(price, Par, train=True):
    """
    Finds the implied volatility using the Bisection Algorithm.

    Args:
        price: Market price of the option
        cp: Option type (+1 for call, -1 for put)
        s: Current stock price (array or scalar)
        k: Strike price
        rf: Risk-free rate
        t: Time to maturity
        div: Dividend yield (default 0.0)

    Returns:
        v_imp: Implied volatility
        price_imp: Option price using the found v_imp
    """
    max_iter = 100 
    it = 0

    if train:
        k = Par['K_train']
    else:
        k = Par['K_sim']
        
    # Initial bounds
    v_do = np.ones_like(k) * 0  # Lower bound (0% vol)
    v_up = np.ones_like(k) * 10   # Upper bound (300% vol)

    diff = 1
    
    while diff > 1e-8 and it < max_iter:
        it += 1
        
        # new test points
        v_imp = (v_do + v_up)/2
        Par['sigma'] = v_imp
        
        price_imp = bs(Par, train)

        # where we are too high
        mask = price_imp > price 

        # updating the bounds
        v_up[mask] = v_imp[mask]
        v_do[~mask] = v_imp[~mask]

        # updating the error
        diff = np.max(np.abs(price_imp - price))

     # Final estimate
    v_imp = (v_do + v_up) / 2

    return v_imp 