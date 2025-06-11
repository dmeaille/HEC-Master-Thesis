function prices_LSV = compute_prices_from_diffusion(X_, par, grid)

nn = size(X_, 3); 

% Initialize the prices_LSV matrix with zeros
prices_LSV = zeros(grid.nb_T_sim, grid.nb_K_sim, nn);

for j = 1:nn
    % Loop over each element in kgrid_sim
    for i = 1:length(grid.kgrid_sim)
        k = grid.kgrid_sim(i);
        % Compute max(X_t - k_i, 0) for each element
        payoffs = max(X_(:,:,j) - k, 0);
        % Sum over all simulations (nb_X) and store the result
        prices_LSV(:, i, j) = 1/par.nb_X * sum(payoffs, 1);
    end
end

end