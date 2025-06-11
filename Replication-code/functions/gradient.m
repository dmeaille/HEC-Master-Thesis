function outvals = gradient(f, varargin)
    % f: input array (n-dimensional)
    % varargin: can be spacing (scalar or vector per axis) and optionally axis
    % edge_order: defaults to 1
    % Adaptation of python numpy gradient function, for comparability with
    % pre-existing pythong code that I developed 

    narginchk(1, Inf);
    edge_order = 1;
    
    % Check if last argument is edge_order
    if isnumeric(varargin{end}) && isscalar(varargin{end})
        edge_order = varargin{end};
        varargin(end) = [];
    end

    f = double(f);
    N = ndims(f);

    % Determine axes and spacing
    if isempty(varargin)
        axes = 1:N;
        dx = repmat({1.0}, 1, N);
    elseif isscalar(varargin{1}) && isscalar(varargin)
        axes = 1:N;
        dx = repmat({varargin{1}}, 1, N);
    elseif isscalar(varargin) && isvector(varargin{1}) && length(varargin{1}) == N
        axes = 1:N;
        dx = varargin{1};
        dx = cellfun(@(d) double(d), num2cell(dx), 'UniformOutput', false);
    else
        axes = 1:length(varargin);
        dx = varargin;
        dx = cellfun(@(d) double(d), dx, 'UniformOutput', false);
    end

    if edge_order > 2
        error('Only edge_order = 1 or 2 is supported.');
    end

    outvals = cell(1, length(axes));

    for k = 1:length(axes)
        axis = axes(k);
        sz = size(f);
        n = sz(axis);
        out = zeros(sz);
        d = dx{k};

        % Slicing
        idx_all = repmat({':'}, 1, N);
        slice1 = idx_all; slice2 = idx_all;
        slice3 = idx_all; slice4 = idx_all;

        if isscalar(d)
            % Uniform spacing
            h = d;
            % Central difference
            slice1{axis} = 2:n-1;
            slice2{axis} = 3:n;
            slice3{axis} = 1:n-2;
            out(slice1{:}) = (f(slice2{:}) - f(slice3{:})) / (2*h);

            % First-order edges
            if edge_order == 1
                % Forward at start
                slice1{axis} = 1; slice2{axis} = 2; slice3{axis} = 1;
                out(slice1{:}) = (f(slice2{:}) - f(slice3{:})) / h;

                % Backward at end
                slice1{axis} = n; slice2{axis} = n; slice3{axis} = n-1;
                out(slice1{:}) = (f(slice2{:}) - f(slice3{:})) / h;
            else
                % Second-order at start
                slice1{axis} = 1; slice2{axis} = 1;
                slice3{axis} = 2; slice4{axis} = 3;
                out(slice1{:}) = (-1.5*f(slice2{:}) + 2*f(slice3{:}) - 0.5*f(slice4{:})) / h;

                % Second-order at end
                slice1{axis} = n; slice2{axis} = n-2;
                slice3{axis} = n-1; slice4{axis} = n;
                out(slice1{:}) = (0.5*f(slice2{:}) - 2*f(slice3{:}) + 1.5*f(slice4{:})) / h;
            end
        else
            % Non-uniform spacing
            if length(d) ~= n
                error('Length of spacing vector must match size of dimension.');
            end

            d = d(:);
            dx1 = d(2:end-1) - d(1:end-2);
            dx2 = d(3:end) - d(2:end-1);
            a = -dx2 ./ (dx1 .* (dx1 + dx2));
            b = (dx2 - dx1) ./ (dx1 .* dx2);
            c = dx1 ./ (dx2 .* (dx1 + dx2));

            % Broadcasted central difference
            for i = 2:n-1
                idx = idx_all;
                idx{axis} = i;
                idxm1 = idx_all; idxm1{axis} = i-1;
                idxp1 = idx_all; idxp1{axis} = i+1;

                out(idx{:}) = a(i-1)*f(idxm1{:}) + b(i-1)*f(idx{:}) + c(i-1)*f(idxp1{:});
            end

            % Edges
            if edge_order == 1
                idx0 = idx_all; idx1 = idx_all;
                idx0{axis} = 1; idx1{axis} = 2;
                out(idx0{:}) = (f(idx1{:}) - f(idx0{:})) / (d(2) - d(1));

                idxn = idx_all; idxnm1 = idx_all;
                idxn{axis} = n; idxnm1{axis} = n-1;
                out(idxn{:}) = (f(idxn{:}) - f(idxnm1{:})) / (d(n) - d(n-1));
            else
                % Start
                dx1 = d(2) - d(1);
                dx2 = d(3) - d(2);
                a = -(2*dx1 + dx2)/(dx1*(dx1 + dx2));
                b = (dx1 + dx2)/(dx1*dx2);
                c = -dx1/(dx2*(dx1 + dx2));

                idx1 = idx_all; idx2 = idx_all; idx3 = idx_all;
                idx1{axis} = 1; idx2{axis} = 2; idx3{axis} = 3;
                out(idx1{:}) = a*f(idx1{:}) + b*f(idx2{:}) + c*f(idx3{:});

                % End
                dx1 = d(n-1) - d(n-2);
                dx2 = d(n) - d(n-1);
                a = dx2 / (dx1*(dx1 + dx2));
                b = -(dx2 + dx1)/(dx1*dx2);
                c = (2*dx2 + dx1)/(dx2*(dx1 + dx2));

                idx1{axis} = n-2; idx2{axis} = n-1; idx3{axis} = n;
                out(idx3{:}) = a*f(idx1{:}) + b*f(idx2{:}) + c*f(idx3{:});
            end
        end

        outvals{k} = out;
    end

    if length(outvals) == 1
        outvals = outvals{1};
    end
end
