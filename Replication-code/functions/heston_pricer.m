function F = heston_pricer(par, train)
    % Heston Fourier algorithm to generate a set of data that features a
    % volatility skew. 
    % 
    % Taken from: 
    % https://github.com/ithakis/Pricing-Options-with-Black-Scholes/blob/main/Pricing%20Options%20with%20Fourier.ipynb

    if nargin < 2
        train = true;
    end

    if train
        K = par.K_train;
        T = par.T_train;
    else
        K = par.K_sim;
        T = par.T_sim;
    end
    

    % Fourier inversion bounds
    c1 = log(par.S0) + par.r*T - 0.5*par.theta*T;

    c2 = par.theta / (8*par.kappa^3) * ( ...
        -par.zeta^2 * exp(-2*par.kappa*T) ...
        + 4*par.zeta*exp(-par.kappa*T)*(par.zeta - 2*par.kappa*par.rho) ...
        + 2*par.kappa*T*(4*par.kappa^2 + par.zeta^2 - 4*par.kappa*par.zeta*par.rho) ...
        + par.zeta*(8*par.kappa*par.rho - 3*par.zeta) );

    a = c1 - par.z * sqrt(abs(c2));
    b = c1 + par.z * sqrt(abs(c2));

    h = @(n) (n * pi) ./ (b - a);
    g_n = @(n) (exp(a) - (K ./ h(n)) .* sin(h(n) .* (a - log(K))) ...
        - K .* cos(h(n) .* (a - log(K)))) ./ (1 + h(n).^2);
    g0 = K .* (log(K) - a - 1) + exp(a);

    F = g0;
    for n = 1:par.N
        hn = h(n);
        F = F + 2 * heston_char(hn, a, T, par.S0, K, par.r, par.kappa, par.theta, par.rho, par.zeta, par.v0, par.opt_type, par.N, par.z) .* exp(-1i .* a .* hn) .* g_n(n);
    end

    F = exp(-par.r * T) ./ (b - a) .* real(F);

    if par.opt_type == 1
        F = F + par.S0 - K .* exp(-par.r * T);  % For call
    end
    
    mask = F < 0;
    F(mask) = 0;
end
