function Fig3_plotSV_ridge_gpr(vimp_LSV, vimp_model, par, grid)
    % Replicates Figure 3: Comparison between Ridge and GPR methods for Smile calibration



    set(gcf, 'Position', [100, 100, 800, 300]); % Adjust figure size
    
    
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
    
    
    % Create a figure to hold all subplots
    
    
    n_slices = size(vimp_LSV, 3); % Number of 3rd-dim slices in vimp_LSV
    maturities = [first_mat, second_mat, third_mat];
    idx_sim = {idx1_sim, idx2_sim, idx3_sim};
    idx_train = {idx1_train, idx2_train, idx3_train};
    
    figure;
    
    for row = 1:n_slices
        for col = 1:3
            subplot(n_slices, 3, (row - 1) * 3 + col);
            
            plot(grid.kgrid_sim, vimp_LSV(idx_sim{col}, :, row), 'DisplayName', 'LSV');
            hold on;
            plot(grid.kgrid_train, vimp_model(idx_train{col}, :), 'DisplayName', 'True Model');
            hold off;
            if row == 1
            title(sprintf("Implied Vol - %d yr", maturities(col)));
            end
            xlabel("Strike");
            ylabel("Implied vol.");
            xlim([0.5, 2]);
            ylim([0, 0.6]);
            
            
            if col == 1
                if row == 1
                ylabel(sprintf("Ridge \nImplied vol.")); % Label rows
                elseif row == 2
                    ylabel(sprintf("GPR \nImplied vol.")); 
                end
            
            end
            
    
            
            if row == 1 && col == 3
                legend('show'); % Show legend only once to avoid clutter
            end
        end
    end
    
    sgtitle(sprintf("Implied Volatility for %d particles", par.nb_X));





end