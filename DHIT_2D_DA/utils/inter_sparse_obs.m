function [u_guess, v_guess] = inter_sparse_obs(u, v, Setups, gap)
% Extract the grids and values for the extended domain
% please make sure Nx and Ny are multiples of gap
    % Transpose the output to match our data orientation
    XU       = Setups.Parameters.XU';
    YU       = Setups.Parameters.YU';
    XV       = Setups.Parameters.XV';
    YV       = Setups.Parameters.YV';
    Lx       = Setups.Parameters.Lx;
    Ly       = Setups.Parameters.Ly;
    [Nx, Ny] = size(XU);

    XU_ext = XU([1:gap:Nx,1], [1:gap:Ny,1]);
    YU_ext = YU([1:gap:Nx,1], [1:gap:Ny,1]);
    u_ext  =  u([1:gap:Nx,1], [1:gap:Ny,1]);
    
    XV_ext = XV([1:gap:Nx,1], [1:gap:Ny,1]);
    YV_ext = YV([1:gap:Nx,1], [1:gap:Ny,1]);
    v_ext  =  v([1:gap:Nx,1], [1:gap:Ny,1]);
    
    % Extend grids and data for periodic conditions
    XU_ext(end,:) = XU_ext(end,:) + Lx;
    YU_ext(:,end) = YU_ext(:,end) + Ly;
    XV_ext(end,:) = XV_ext(end,:) + Lx;
    YV_ext(:,end) = YV_ext(:,end) + Ly;

    % Perform interpolation with the periodic extensions
    u_guess = interp2(XU_ext', YU_ext', u_ext', XU', YU', 'linear')';
    v_guess = interp2(XV_ext', YV_ext', v_ext', XV', YV', 'linear');
    
    [u_guess, v_guess] = divergence_free_projection(u_guess, v_guess, Setups);
end