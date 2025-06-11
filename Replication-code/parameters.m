function par = parameters(Tmax, delta_t, grid)

% Define the parameters using dot notation
par.S0 = 1.0;             % Initial Spot
par.r = 0.0;              % Fixed interest rates
par.v0 = 0.0045;          % Initial volatility
par.sigma = 0.3;          % Constant volatility (for Black Scholes)
par.kappa = 1.19;         % Speed of the mean reversion
par.theta = 0.07023;      % Long term mean
par.zeta = 0.8;          % Volatility of volatility
par.rho = -0.83;          % Correlation between 2 random variables
par.opt_type = 1;         % 1 for call
par.N = 1012;             % Number of simulations
par.z = 24;               % Additional parameter

% LSV Model parameters
par.X_0 = 1.0;           % Initial value for X
par.Y_0 = 0.025;         % Initial volatility @ 12%
par.mu = 0.025;          % Mean of the process
par.lambda = 1.2;        % Lambda parameter
par.eta = 0.9;           % Eta parameter
par.rho_sto = -0.9;      % Stochastic correlation
par.cap = 1e-3;          % Cap parameter

% Simulation parameters
par.nb_X = 1e4;          % Number of X simulations
par.Tmax = Tmax;          % Maximum time
par.delta_t = delta_t;    % Time step
par.nb_T_sim = floor(grid.Tmax / grid.delta_t);  % Number of time simulations

% Grids
par.tgrid_train = grid.tgrid_train; % Training time grid
par.kgrid_train = grid.kgrid_train; % Training k grid
par.K_train = grid.K_train;     % Training K
par.T_train = grid.T_train;     % Training T

par.K_sim = grid.K_sim;         % Simulation K
par.T_sim = grid.T_sim;         % Simulation T

% Derivative steps
par.dt_train = mean(diff(grid.tgrid_train));
par.dk_train = mean(diff(grid.kgrid_train));
par.dt_sim = mean(diff(grid.tgrid_sim));
par.dk_sim = mean(diff(grid.kgrid_sim));

end
