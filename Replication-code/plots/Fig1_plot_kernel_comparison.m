function Fig1_plot_kernel_comparison(X_train, Y_train, X_test)
    % Replicates Figure 1: Comparison of Kernels

    X_train = X_train(:);
    Y_train = Y_train(:);

    % Fitting the matern kernel
    gprMdl = fitrgp(X_train, Y_train, ...
    'KernelFunction', 'matern32', ...
    'BasisFunction', 'none', ...
    'FitMethod', 'exact', ...
    'PredictMethod', 'exact');

    mu_matern = predict(gprMdl, X_test);


    % Fitting the squared exponential kernel
    gprMdl = fitrgp(X_train, Y_train, ...
    'KernelFunction', 'squaredexponential', ...
    'BasisFunction', 'none', ...
    'FitMethod', 'exact', ...
    'PredictMethod', 'exact');

    mu_squared = predict(gprMdl, X_test);

    scatter(X_test, mu_matern, ".");
    hold on; 
    scatter(X_test, mu_squared, "."); 
    legend('Matern Kernel', 'Squared Exponential Kernel')
    title("Comparison of Matérn 3/2 vs Squared Exponential Kernel")

end




