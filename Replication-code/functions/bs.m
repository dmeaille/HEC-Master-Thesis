function price = bs(par, train)
    % Vectorized Black-Scholes Formula, based on the parameters input in
    % the par structure. 

    % Determine K and T based on the train flag
    if train
        K = par.K_train;
        T = par.T_train;
    else
        K = par.K_sim;
        T = par.T_sim;
    end

    % Calculate d1 and d2
    d1 = (log(par.S0 ./ K) + (par.r + 0.5 * par.sigma.^2) .* T) ./ (par.sigma .* sqrt(T));
    d2 = d1 - par.sigma .* sqrt(T);

    % Calculate the option price
    if par.opt_type == 1
        price = par.S0 * normcdf(d1) - K .* exp(-par.r * T) .* normcdf(d2);
    else % put option
        price = K .* exp(-par.r * T) .* normcdf(-d2) - par.S0 * normcdf(-d1);
    end
end
