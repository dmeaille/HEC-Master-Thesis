function K = gaussian_kernel(X, Y, sigma)
    % Computes the Gaussian kernel matrix between input points X and Y.
    %
    % Parameters:
    %   X: Input vector or array of size n, representing n points.
    %   Y: Input vector or array of size m, representing m points.
    %   sigma: Bandwidth parameter that determines the width of the Gaussian kernel.
    %
    % Returns:
    %   K: Kernel matrix of size n x m containing the Gaussian similarities between points in X and Y.

    % Ensure X and Y are used to create 2D grid matrices for pairwise computation
    [X_grid, Y_grid] = meshgrid(X, Y);

    % Compute the Gaussian kernel
    K = exp(-(X_grid - Y_grid).^2/(2*sigma^2));

end