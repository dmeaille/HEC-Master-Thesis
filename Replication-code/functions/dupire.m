function local_vol = dupire(prices, par)
    % Computing the local volatility grid for the Dupire Volatility 
    % Adjustments in the end to extrapolate over the areas that couldn't be computed

    grad = gradient(prices, par.tgrid_train, par.kgrid_train, 2);
    dC_dT = grad{1};
    dC_dK = grad{2};
    dC_dT(dC_dT < 0) = 1e-10;

    grad_2 = gradient(dC_dK, par.tgrid_train, par.kgrid_train, 2);
    d2C_dK2 = grad_2{2};

    local_vol = sqrt((dC_dT) ./ (0.5 * par.K_train.^2 .* d2C_dK2));
    local_vol(d2C_dK2 < 0) = NaN;
    local_vol(local_vol == Inf) = NaN;
    % local_vol = min(local_vol, 1);

end


