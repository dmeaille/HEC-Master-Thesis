function K = matern_kernel(X, Y, l_f, sigma_f)
    % Computes the Matern kernel matrix between input points X and Y.
    %
    % Parameters:
    %   X: Input vector
    %   Y: Input vector
    %   l_f: Length scale parameter that determines the correlation distance.
    %   sigma_f: Signal standard deviation parameter that scales the kernel.
    %
    % Returns:
    %   K: Kernel matrix of size n x m containing the covariance between points in X and Y.

    % Compute pairwise Euclidean distances between points in X and Y
    dist = sqrt(pdist2(X(:), Y(:)).^2);

    % Compute the Matern kernel with nu = 3/2
    K = sigma_f^2 * (1 + sqrt(3)*dist/l_f) .* exp(-sqrt(3)*dist/l_f);

end