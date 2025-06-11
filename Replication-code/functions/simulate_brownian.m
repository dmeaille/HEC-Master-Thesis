function [dW_x, dW_v] = simulate_brownian(par)
    % Generate two correlated Brownian motions, using the Cholesky
    % decomposition. 
    
    % Correlation matrix
    corr_matrix = [1, par.rho_sto; par.rho_sto, 1];
    
    % Cholesky decomposition
    L = chol(corr_matrix, 'lower');
    
    % Generate uncorrelated standard normal variables
    Z = randn(par.nb_X, par.nb_T_sim+1, 2);
    
    % Apply Cholesky decomposition to get correlated normal variables
    % Need to reshape for matrix multiplication in MATLAB
    Z_reshaped = reshape(Z, par.nb_X * (par.nb_T_sim+1), 2);
    correlated_Z_reshaped = Z_reshaped * L';
    correlated_Z = reshape(correlated_Z_reshaped, par.nb_X, par.nb_T_sim+1, 2);
    
    % Scale by sqrt(delta_t) to get correlated Brownian increments
    dW_x = correlated_Z(:, :, 1) * sqrt(par.delta_t);
    dW_v = correlated_Z(:, :, 2) * sqrt(par.delta_t);
end