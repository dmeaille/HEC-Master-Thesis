function [X_, v_, dW_x, dW_v] = initialize_diffusion(par)
    % Function that initializes the diffusion, generating the correlated
    % Brownian motions for the stock and for the volatility, as well as
    % computing the Monte-Carlo simulation of the volatility. 

    % Get Brownian increments
    [dW_x, dW_v] = simulate_brownian(par);
    
    % Stock price initialization
    X_ = ones(par.nb_X, par.nb_T_sim) * par.X_0;
    
    % Variance process initialization
    v_ = ones(par.nb_X, par.nb_T_sim) * par.Y_0;
    v_(:, 1) = max(v_(:, 1), par.cap); % Apply cap to initial values
    
    % Simulate the variance process using SDE discretization
    for i = 2:par.nb_T_sim
        v_(:, i) = v_(:, i-1) + par.lambda * (par.mu - v_(:, i-1)) * par.delta_t + ...
                   par.eta * sqrt(v_(:, i-1)) .* dW_v(:, i);
        
        % Replace NaN values with cap and enforce minimum value
        v_(:, i) = max(v_(:, i), par.cap);
        
        % Handle any NaN values (equivalent to np.nan_to_num)
        v_(isnan(v_(:, i)), i) = par.cap;
    end
end