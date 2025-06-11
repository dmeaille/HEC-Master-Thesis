function [mu_pred, cov_s, uncertainty] = posterior_GPR(X_test, X_train, Y_train)
    % Computes the sufficient statistics of the GP posterior predictive distribution
    % from m training data X_train and Y_train and n new inputs X_test.
    %
    % Args:
    % X_test: New input locations.
    % X_train: Training locations.
    % Y_train: Training targets.
    % l: Kernel length parameter.
    % sigma_f: Kernel vertical variation parameter.
    % sigma_y: Noise parameter.
    %
    % Returns:
    % Posterior mean vector and covariance matrix.
 
    Xtrain = X_train(:);
    n = size(Xtrain, 1);
    Ytrain = Y_train(:);
    Xtest = X_test(:);
    
    
    % Step 1: Train GPR with fitrgp to get the parameters
    gprMdl = fitrgp(Xtrain, Ytrain, ...
    'KernelFunction', 'matern32', ...
    'BasisFunction', 'none', ...
    'Optimizer','quasinewton', ...
    'FitMethod', 'exact', ...
    'PredictMethod', 'exact');
    
    % Step 2: Extract model params
    theta = gprMdl.KernelInformation.KernelParameters;
    % disp(theta);
    lengthScale = theta(1);
    sigmaF = theta(2);
    sigmaN = gprMdl.Sigma;
    
    % Step 3: Implement Matern 3/2 kernel manually
    K_ = matern_kernel(Xtrain, Xtrain, lengthScale, sigmaF) + sigmaN^2 * eye(n);
    Ks_ = matern_kernel(Xtrain, Xtest, lengthScale, sigmaF);
    Kss_ = matern_kernel(Xtest, Xtest, lengthScale, sigmaF);
    
    % Predictive mean and variance
    L = chol(K_ + 1e-10*eye(n),'lower'); % Numerical stability
    alpha = L'\(L\Ytrain);
    mu_pred = Ks_' * alpha;
    mu_pred(mu_pred < 1e-3) = 1e-3; 
    
    v = L\Ks_;
    cov_s = Kss_ - v'*v;
    var_pred = diag(cov_s); % Predictive variance

    uncertainty = 1/4 * var_pred./min(mu_pred.^3, 1e-2); % final uncertainty 



end

