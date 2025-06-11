function [VV_imp, price_imp] = implied_vol(price, par, train)
    % Finds the implied volatility using the Bisection Algorithm.
    %
    % Args:
    %   price: Market price of the option
    %   par: Parameter structure containing:
    %       K_train: Training strike price
    %       K_sim: Simulation strike price
    %       (other parameters needed for bs function)
    %   train: Boolean flag to determine if training or simulating
    %
    % Returns:
    %   v_imp: Implied volatility
    %   price_imp: Option price using the found v_imp

    max_iter = 100;
    

    nn = size(price, 3);

    if train
        k = par.K_train;
    else
        k = par.K_sim;
    end

    VV_imp = zeros([size(k), nn]); 
    
    for j = 1:nn
        % Initialize max iteration 
        it = 0;
        % Initial bounds
        pp = squeeze(price(:,:,j));
        v_do = zeros(size(k));  % Lower bound (0% vol)
        v_up = ones(size(k));  % Upper bound (100% vol)
    
        diff = 1;
    
        while (diff > 1e-8) * (it < max_iter)
            it = it + 1;
    
            % New test points
            v_imp = (v_do + v_up) / 2;
            par.sigma = v_imp;
    
            price_imp = bs(par, train);
    
            % Where we are too high
            mask = price_imp > pp;
    
            % Updating the bounds
            v_up(mask) = v_imp(mask);
            v_do(~mask) = v_imp(~mask);
    
            % Updating the error
            diff = max(max(abs(price_imp - price)));
        end
      
    
        % Final estimate
        VV_imp(:,:,j) = (v_do + v_up) / 2;

    end
end
