clear all; close all; clc;
warning('off','all')

addpath("functions/")
addpath("plots/")

%% Simulation settings


Tmax = 10.0; 
delta_t = 0.04;

grid = create_grids(Tmax, delta_t);
par = parameters(Tmax, delta_t, grid);

% First with Heston pricer
prices = heston_pricer(par, true);

% Creating the grid to compute the local volatility
local_vol = dupire(prices, par);
sigma_dupire = griddedInterpolant({par.tgrid_train, par.kgrid_train}, local_vol, 'linear');

% Parameter for the Ridge case
sigma_f = 0.5;

% Normalization constant
lambda_par = 1e-9; 

% Computing the target volatility surface
vimp_model = implied_vol(prices, par, true);
% vimp_mod(:,:,1) = vimp_model;    % for later
% vimp_mod(:,:,2) = vimp_model;

prices_bs = bs(par, true);
vimp_model_bs = implied_vol(prices_bs, par, true);
vimp_mod_heston_bs(:,:,1) = vimp_model;
vimp_mod_heston_bs(:,:,2) = vimp_model_bs;


%% Simulating the diffusion

% Simulates the diffusion with Heston and GPR
[X_, ~, v_] = simulate_diffusion(sigma_dupire, sigma_f, lambda_par, par, grid, "GPR");
xx = X_;

% Computes the prices from the McKean-Vlasov Monte-Carlo
prices_LSV_heston = compute_prices_from_diffusion(X_, par, grid);

% Computes the implied volatility surfaces from the diffusion and from the
% model (BS or Heston)
vimp_LSV_gpr_heston = implied_vol(prices_LSV_heston, par, false);



% Same but with Ridge 
[X_, ~, ~] = simulate_diffusion(sigma_dupire, sigma_f, lambda_par, par, grid, "Ridge");
prices_LSV_ridge = compute_prices_from_diffusion(X_, par, grid);
vimp_LSV_gpr_ridge = implied_vol(prices_LSV_ridge, par, false);


% Same but with Black Scholes and GPR
prices = bs(par, true);
local_vol = dupire(prices, par);
sigma_dupire = griddedInterpolant({par.tgrid_train, par.kgrid_train}, local_vol, 'linear');
[X_, ~, ~] = simulate_diffusion(sigma_dupire, sigma_f, lambda_par, par, grid, "GPR");
prices_LSV_bs = compute_prices_from_diffusion(X_, par, grid);
vimp_LSV_gpr_bs = implied_vol(prices_LSV_bs, par, false);



%% Figure 1: Comparison of Kernels

l_ind = compute_percentiles(100);

mid = 2; 
xx = xx(:, mid, 2); 
yy = v_(:, mid);

% Taking the percentiles values of our training X
cutoff_values = prctile(xx, l_ind);

X_train = zeros(1, 100);
Y_train = zeros(1, 100);
X_test = xx;

for i = 1:100
    % Find the index of the value in X closest to the percentile cutoff
    [~, idx] = min(abs(xx - cutoff_values(i)));
    X_train(i) = xx(idx);
    Y_train(i) = yy(idx);
end


Fig1_plot_kernel_comparison(X_train, Y_train, X_test)


%% Figure 2: Comparison of performance to replicate the prices from Heston vs Black-Scholes

% Heston
vimp_LS(:,:,1) = vimp_LSV_gpr_heston(:, :, 2); % taking the mean prediction
% Black-Scholes
vimp_LS(:,:,2) = vimp_LSV_gpr_bs(:, :, 2); % taking the mean prediction

Fig2_plotSV_heston_bs(vimp_LS, vimp_mod_heston_bs, par, grid)


%% Figure 3: Comparison between Ridge and GPR methods for Smile calibration 

% Ridge
vimp_LS(:,:,1) = vimp_LSV_gpr_ridge; 
% GPR
vimp_LS(:,:,2) = vimp_LSV_gpr_heston(:, :, 2); % taking the mean prediction


Fig3_plotSV_ridge_gpr(vimp_LS, vimp_model, par, grid)


%% Figure 4: Point wise Lower and Upper bounds with GPR

Fig4_plotSV_bounds(vimp_LSV_gpr_heston, vimp_model, par, grid)


