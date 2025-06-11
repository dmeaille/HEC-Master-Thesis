clear all;
warning('off','all')

% Replicate the original base results from the paper: 
% 
% Bayer, C., Belomestny, D., Butkovsky, O., & Schoenmakers, J. (2024). 
% A reproducing kernel Hilbert space approach to singular local stochastic 
% volatility McKean–Vlasov models. Finance and Stochastics, 28(4), 1147-1178.
% 
% using Ridge regression and Gaussian Kernel. 


%% Loading the grids and par

Tmax = 10.0; 
delta_t = 0.01;

grid = create_grids(Tmax, delta_t);
par = parameters(Tmax, delta_t, grid);

% Set it to 1 to get pedagogical plots, such as the evolution of the
% distribution and the prices surface
additional_plots = 1;

sigma_f = 0.5;
lambda_par = 1e-9;

%% Generating the prices and the dupire grid 

train = true; 
prices = heston_pricer(par, train);
% prices = bs(par, train);


local_vol = dupire(prices, par);

if additional_plots
    surf(par.T_train, par.K_train, local_vol)
    xlabel("Time")
    ylabel("Strike")
    title("Dupire Local Volatility")
end

sigma_dupire = griddedInterpolant({par.tgrid_train, par.kgrid_train}, local_vol, 'linear');

[X_, v_, dW_x, dW_v] = initialize_diffusion(par);



%% Computing the diffusion


kernel = @(x, y, z) gaussian_kernel(x, y, z);

cond = zeros(par.nb_T_sim, par.nb_X);
L = 10;


% First step (t=2 in MATLAB due to 1-based indexing)
t = 2;
X_(:, t) = X_(:, t-1) + sqrt(par.Y_0) .* X_(:, t-1) .* ...
           sigma_dupire(par.delta_t * ones(par.nb_X, 1), X_(:, t-1)) ./ sqrt(par.Y_0) .* dW_x(:, t);

% Loop through remaining time steps
for t = 3:par.nb_T_sim
    % Compute beta and conditional expectation
    [beta_hat, m_A_lam] = compute_ridge(X_, v_, t, L, kernel, lambda_par, sigma_f, par);
    
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
    fprintf('\rSteps completed: %d/%d', t, par.nb_T_sim);
end
fprintf('\n'); % Add newline at the end



%% Plot of the diffusion of the 50s first particles

if additional_plots
% Assuming X_ is a matrix where each row is a process and each column is a time step
mean_ = mean(X_, 1); % Calculate mean along the first dimension (rows)
std_ = std(X_, 1);   % Calculate standard deviation along the first dimension (rows)

confidence_level = 0.95;
confidence_interval = std_ * 1.96; % Approximate z-score for 95% confidence

figure;
hold on;
for i = 1:50
    plot(X_(i,:), 'Color', [0.53, 0.81, 0.92], 'LineWidth', 1.5, 'HandleVisibility', 'off'); % skyblue color
end

title("Diffusion of " + string(50) + " processes");
ylim([0, 3]);

% Plot mean and confidence interval
plot(mean_, 'Color', 'red', 'LineWidth', 2, 'DisplayName', 'Mean');
fill_between = fill([1:par.nb_T_sim, par.nb_T_sim:-1:1], [mean_ - confidence_interval, fliplr(mean_ + confidence_interval)], 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '95% Confidence Interval');
hold off;
xlabel("Time steps")
legend("show")
end


%% Compute the implied prices

% Initialize the prices_LSV matrix with zeros
prices_LSV = zeros(grid.nb_T_sim, grid.nb_K_sim);

% Loop over each element in kgrid_sim
for i = 1:length(grid.kgrid_sim)
    k = grid.kgrid_sim(i);
    % Compute max(X_t - k_i, 0) for each element
    payoffs = max(X_ - k, 0);
    % Sum over all simulations (nb_X) and store the result
    prices_LSV(:, i) = 1/par.nb_X * sum(payoffs, 1);
end



%% Plotting the prices surface

if additional_plots
figure;
surf(grid.T_sim, grid.K_sim, prices_LSV)
ylim([0,2])
xlabel('Time'); 
ylabel('Strike');
title('Prices from the diffusion'); 
end


%% Computing the volatility surface implied by the simulation

vimp_LSV = implied_vol(prices_LSV, par, false);
vimp_model = implied_vol(prices, par, true);



%% Plot the smile comparisons 


figure;
set(gcf, 'Position', [100, 100, 800, 300]); % Adjust figure size

% Set the overall title
sgtitle(sprintf("Implied Volatility for %d particles", par.nb_X), 'FontSize', 16);

first_mat = 1;
second_mat = 4;
third_mat = 10;

% Find the indices for the closest maturity values
[~, idx1_sim] = min(abs(grid.tgrid_sim - first_mat));
[~, idx1_train] = min(abs(grid.tgrid_train - first_mat));

[~, idx2_sim] = min(abs(grid.tgrid_sim - second_mat));
[~, idx2_train] = min(abs(grid.tgrid_train - second_mat));

[~, idx3_sim] = min(abs(grid.tgrid_sim - third_mat));
[~, idx3_train] = min(abs(grid.tgrid_train - third_mat));

% First subplot
subplot(1, 3, 1);
plot(grid.kgrid_sim, vimp_LSV(idx1_sim, :), 'DisplayName', 'LSV');
hold on;
plot(grid.kgrid_train, vimp_model(idx1_train, :), 'DisplayName', 'True Model');
title(sprintf("Implied Volatility - %d yr", first_mat));
xlabel("Strike");
ylabel("Implied vol.");
xlim([0.5, 2.5]);
ylim([0.1, 0.6]);
hold off;

% Second subplot
subplot(1, 3, 2);
plot(grid.kgrid_sim, vimp_LSV(idx2_sim, :), 'DisplayName', 'LSV');
hold on;
plot(grid.kgrid_train, vimp_model(idx2_train, :), 'DisplayName', 'True Model');
title(sprintf("Implied Volatility - %d yr", second_mat));
xlabel("Strike");
ylabel("Implied vol.");
xlim([0.5, 2.5]);
ylim([0.1, 0.3]);
hold off;

% Third subplot
subplot(1, 3, 3);
plot(grid.kgrid_sim, vimp_LSV(idx3_sim, :), 'DisplayName', 'LSV');
hold on;
plot(grid.kgrid_train, vimp_model(idx3_train, :), 'DisplayName', 'True Model');
title(sprintf("Implied Volatility - %d yr", third_mat));
xlabel("Strike");
ylabel("Implied vol.");
xlim([0.5, 2.5]);
ylim([0.1, 0.3]);
legend('show');
hold off;

% Adjust layout to prevent overlap
sgtitle(sprintf("Implied Volatility for %d particles", par.nb_X));