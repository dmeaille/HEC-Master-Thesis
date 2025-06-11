function [beta_hat, m_A_lam] = compute_ridge(X_, v_, t, L, kernel, lambda_par, sigma_f, par)
    % Computes beta_hat as well as the distribution, using the Ridge regression 
    % with regularization parameter lambda_par


    %% Choosing L << N
    if L > 0
        l_ind = compute_percentiles(L);
    else
        l_ind = linspace(0, par.nb_X-1, par.nb_X);
        l_ind = floor(l_ind); % Convert to integers (equivalent to astype(int))
        L = par.nb_X;
    end
    
    %% Computing $\hat{\beta}$
    % We are selecting the percentiles as the training data, for speed
    % purposes
    Z_j = prctile(X_(:, t-1), l_ind); 
    Z_j = reshape(Z_j, L, 1);
    
    % K(k(Zj, X^n))
    K = kernel(Z_j, X_(:, t-1), sigma_f);
    
    % R(k(Zj, Zl))
    R = kernel(Z_j, Z_j, sigma_f);
    
    % G(A(Y_t^n)) 
    G = v_(:, t); 
    
    %% Computes beta_hat
    
    temp = K' * K + lambda_par * par.nb_X * R;
    beta_hat = temp \ (K' * G);
    
    %% Computes the distribution
    m_A_lam = 0;
    
    for j = 1:L
        zz = Z_j(j);
        m_A_lam = m_A_lam + beta_hat(j) * kernel(X_(:, t-1), reshape(zz, 1, 1), sigma_f);
    end
    m_A_lam(m_A_lam < par.cap) = par.cap;
   
end