
function val = heston_char(u, a, T, S0, K, r, kappa, theta, rho, zeta, v0, opt_type, N, z)
    t0 = 0.0;
    q = 0.0;
    m = log(S0) + (r - q) * (T - t0);

    D = sqrt((rho * zeta * 1i * u - kappa).^2 + ...
             zeta^2 * (1i * u + u.^2));

    C = (kappa - rho * zeta * 1i * u - D) ./ ...
        (kappa - rho * zeta * 1i * u + D);

    beta = ((kappa - rho * zeta * 1i * u - D) .* ...
           (1 - exp(-D .* (T - t0)))) ./ ...
           (zeta^2 .* (1 - C .* exp(-D .* (T - t0))));

    alpha = (kappa * theta / zeta^2) .* ...
        ( (kappa - rho * zeta * 1i * u - D) .* (T - t0) - ...
        2 * log( (1 - C .*  exp(-D .*(T - t0))  ./ (1 - C)) ) ...
        );


    val = exp(1i .* u .* m + alpha + beta * v0); 
end