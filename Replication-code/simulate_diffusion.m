function [X, cond, v_] = simulate_diffusion(sigma_dupire, sigma_f, lambda_par, par, grid, method)

% do not modify
internal = "pointwise"; 

[X_, v_, dW_x, ~] = initialize_diffusion(par);

% To store the estimated conditional expectation
cond = zeros(par.nb_X, par.nb_T_sim);

% Parameter for the size of the training sample 
L = 200;


% First step (t=2 in MATLAB due to 1-based indexing)
t = 2;
X_(:, t) = X_(:, t-1) + sqrt(par.Y_0) .* X_(:, t-1) .* ...
           sigma_dupire(par.delta_t * ones(par.nb_X, 1), X_(:, t-1)) ./ sqrt(par.Y_0) .* dW_x(:, t);



if method == "Ridge" % In case the original Ridge regression is selected
    kernel = @(x, y, z) gaussian_kernel(x, y, z);


    % Loop through remaining time steps
    for t = 3:par.nb_T_sim
        % Compute beta and conditional expectation 
        [~, m_A_lam] = compute_ridge(X_, v_, t, L, kernel, lambda_par, sigma_f, par);
        
        % Update the memory for the graphs later
        cond(:, t) = m_A_lam(:);
        
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
        fprintf('\rSteps completed: %d/%d', t, par.nb_T_sim);
    end

    X = X_; 




elseif method == "GPR" & internal == "pointwise" % In case the new GPR version is selected

    cond = zeros(par.nb_X, par.nb_T_sim);
    Up_X = X_;
    Lo_X = X_;
    
    Var_x = ones(par.nb_X, par.nb_T_sim);
    
    % Loop through remaining time steps
    for t = 3:par.nb_T_sim
        % Compute beta and conditional expectation
        [m_A_lam, ~, uncertainty] = GPR_pred(X_, v_, t, L, par);
        
        % Update the memory for the graphs later
        cond(:, t) = m_A_lam(:);
        Var_x(:, t) = mean(X_(:, t))^2 .*  1.96 .* sqrt(uncertainty(:));

        % Update step - evaluate Dupire's local vol
        Sig = sigma_dupire(grid.tgrid_sim(t) * ones(par.nb_X, 1), X_(:, t-1));
                
        % Handle NaN values in Sig with interpolation
        Sig = check_nan_sig(Sig);

        % Update stock prices
        X_(:, t) = X_(:, t-1) + sqrt(v_(:, t-1)) .* X_(:, t-1) .* ...
                   Sig ./ sqrt(m_A_lam(:)) .* dW_x(:, t);
        pp = X_(:, t-1) + sqrt(v_(:, t-1)) .* X_(:, t-1) .* ...
                   Sig .* dW_x(:, t) .* ( 1./ sqrt(m_A_lam(:)) + 1.96*sqrt(uncertainty) );
        mm = X_(:, t-1) + sqrt(v_(:, t-1)) .* X_(:, t-1) .* ...
                   Sig .* dW_x(:, t) .* ( 1./ sqrt(m_A_lam(:)) - 1.96*sqrt(uncertainty) );
        Up_X(:, t) = max(pp, mm);

        Lo_X(:, t) = min(pp, mm);
   
         

        % Apply bounds for the interpolation
        X_(:, t) = min(max(X_(:, t), grid.Kmin), grid.Kmax);
        Up_X(:, t) = min(max(Up_X(:, t), grid.Kmin), grid.Kmax);
        Lo_X(:, t) = min(max(Lo_X(:, t), grid.Kmin), grid.Kmax);
        
        % Display progress
        fprintf('\rSteps completed: %d/%d', t, par.nb_T_sim);
    end

    X = zeros([size(X_), 3]);
    X(:, :, 1) = Lo_X; X(:, :, 2) = X_; X(:, :, 3) = Up_X; 


end

end