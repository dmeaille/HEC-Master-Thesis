function [X_, cond] = simulate_diffusion(sigma_dupire, sigma_f, lambda_par, par, grid, method)


[X_, v_, dW_x, ~] = initialize_diffusion(par);

if method == "Ridge"
    kernel = @(x, y, z) gaussian_kernel(x, y, z);
elseif method == "GPR"
    kernel = @(x, y, w, z) sq_exp_kernel(x, y, w, z);
end

cond = zeros(par.nb_T_sim, par.nb_X);
L = 100;


% First step (t=2 in MATLAB due to 1-based indexing)
t = 2;
X_(:, t) = X_(:, t-1) + sqrt(par.Y_0) .* X_(:, t-1) .* ...
           sigma_dupire(par.delta_t * ones(par.nb_X, 1), X_(:, t-1)) ./ sqrt(par.Y_0) .* dW_x(:, t);

% Loop through remaining time steps
for t = 3:par.nb_T_sim
    % Compute beta and conditional expectation
    if method == "Ridge"
        [~, m_A_lam] = compute_beta(X_, v_, t, L, kernel, lambda_par, sigma_f, par);
    elseif method == "GPR"
        [~, m_A_lam] = GPR_pred(X_, v_, t, L, kernel, lambda_par, l_f, sigma_f, par);
        disp(0)
    end
    
    % Update the memory for the graphs later
    cond(t, :) = m_A_lam(:);
    
    % Update step - evaluate Dupire's local vol
    Sig = sigma_dupire(grid.tgrid_sim(t) * ones(par.nb_X, 1), X_(:, t-1));
    
    % Handle NaN values in Sig with interpolation
    if any(isnan(Sig))
        indices = 1:length(Sig);
        valid_mask = ~isnan(Sig);
        
        % Create interpolation function for valid points
        if sum(valid_mask) > 1  % Need at least 2 points for interpolation
            Sig_valid = interp1(indices(valid_mask), Sig(valid_mask), indices, 'linear', 'extrap');
            Sig = Sig_valid;
        else
            % If too few valid points, use a default value
            Sig(isnan(Sig)) = mean(Sig(~isnan(Sig)));
        end
        Sig = Sig';
    end
    
    % Update stock prices
    X_(:, t) = X_(:, t-1) + sqrt(v_(:, t-1)) .* X_(:, t-1) .* ...
               Sig ./ sqrt(m_A_lam(:)) .* dW_x(:, t);
    
    % Apply bounds for the interpolation
    X_(:, t) = min(max(X_(:, t), grid.Kmin), grid.Kmax);
    
    % Display progress
    % fprintf('\rSteps completed: %d/%d', t, par.nb_T_sim);
end

end