function [mu_s, cov_s, uncertainty] = GPR_pred(X_, v_, t, L, par)
    % Computes beta_hat as well as the distribution, using the ridge regression 
    % with regularization parameter lambda_par


    %% Choosing L << N
    if L > 0
        l_ind = compute_percentiles(L); % This can be removed from the loop for very small gains
    else
        l_ind = linspace(0, par.nb_X-1, par.nb_X);
        l_ind = floor(l_ind); % Convert to integers (equivalent to astype(int))
        L = par.nb_X;
    end

    %% Creatingthe sub arrays for the training 
    % We need to do it very carefully, as we must make sure to match the
    % percentiles data from X with their counterpart from Y
        
    X_test = X_(:,t-1);
    cutoff_values = prctile(X_(:, t-1), l_ind);
    
    % Preallocate output
    X_train = zeros(1, L);
    Y_train = zeros(1, L);
    
    for i = 1:L
        % Find the index of the value in X closest to the percentile cutoff
        [~, idx] = min(abs(X_(:, t-1) - cutoff_values(i)));
        X_train(i) = X_(idx, t-1);
        Y_train(i) = v_(idx, t-1);
    end

    [mu_s, cov_s, uncertainty] = posterior_GPR(X_test, X_train, Y_train);
    
    % Making last minues adjustments to ensure numerical stability
    idx = find(mu_s < par.cap);
    uncertainty(idx) = 0; 
    mu_s(idx) = par.cap; 
    
    

   
end