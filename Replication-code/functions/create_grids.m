function grid = create_grids(Tmax, delta_t)
    % Generates the training and simulation grids, for a give maximal
    % maturity and time step. 

    
    % Define parameters
    grid.Tmax = Tmax;                          % Maximum maturity
    grid.delta_t = delta_t;
    grid.nb_T_sim = floor(grid.Tmax / grid.delta_t);    % Number of time steps for simulation
    grid.nb_T_train = 30;                    % Number of time steps for training
    
    grid.Kmax = 3.1;
    grid.Kmin = 0.01;
    grid.nb_K_sim = 200;                      % Number of K steps for simulation
    grid.nb_K_train = 50;                    % Number of K steps for training
    
    % Create grids for training
    grid.kgrid_train = linspace(grid.Kmin, grid.Kmax, grid.nb_K_train);
    grid.tgrid_train = linspace(grid.delta_t, grid.Tmax, grid.nb_T_train);
    [grid.K_train, grid.T_train] = meshgrid(grid.kgrid_train, grid.tgrid_train);
    
    % Create grids for simulation
    grid.kgrid_sim = linspace(grid.Kmin, grid.Kmax, grid.nb_K_sim);
    grid.tgrid_sim = linspace(grid.delta_t, grid.Tmax, grid.nb_T_sim);
    [grid.K_sim, grid.T_sim] = meshgrid(grid.kgrid_sim, grid.tgrid_sim);


end