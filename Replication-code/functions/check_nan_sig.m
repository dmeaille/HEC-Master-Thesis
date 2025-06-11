function Sig = check_nan_sig(Sig)
    % Checks for NaN values in the input array and replaces them using interpolation or a default value.
    %
    % Parameters:
    %   Sig: Input array that may contain NaN values.
    %
    % Returns:
    %   Sig: Array with NaN values replaced by interpolated values or a default value.


    % Check if there are any NaN values in the input array
    if any(isnan(Sig))
        indices = 1:length(Sig);
    
        % Create a logical mask for valid (non-NaN) elements
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


end