function [U, V, R, cost] = F_simulation(u0, v0, r0, Setups)
    % Solve forward 2D Navier-Stokes equation using initial condition u0,
    % v0, r0, with Setups specifying the parameteres
    % The input variables seems to be unused, but they are really used in
    % the script Solver_initialization.
    
    Solver_F_initialization;

    %% Main loop
    for steps = 1:Nt - 1    
        if(steps > 1)
            alpha = alpha_record; beta = alpha - 1;
        end
    
        % EXPLICIT DIFFUSION
        if(random_forcing)
            u_diff = - real(ifft2(fft2(u).*(nv * (Modified_k2_ori.^n_nv) + mu * (Modified_k2_ori.^n_mu))));
            v_diff = - real(ifft2(fft2(v).*(nv * (Modified_k2_ori.^n_nv) + mu * (Modified_k2_ori.^n_mu))));
        else
            u_diff = nu*((circshift(u,[1,0]) + circshift(u,[-1,0]) - 2*u)/dx/dx + (circshift(u,[0,1]) + circshift(u,[0,-1]) - 2*u)/dy/dy);
            v_diff = nu*((circshift(v,[1,0]) + circshift(v,[-1,0]) - 2*v)/dx/dx + (circshift(v,[0,1]) + circshift(v,[0,-1]) - 2*v)/dy/dy);
        end
        if(r_flag) 
            r_diff = kp*((circshift(r,[1,0]) + circshift(r,[-1,0]) - 2*r)/dx/dx + (circshift(r,[0,1]) + circshift(r,[0,-1]) - 2*r)/dy/dy);
        end

        % ADVECTION TERMS
        uv = shifthalf(u,[0,-1]).*shifthalf(v,[-1,0]);
    
        duu_dx = circdiff(shifthalf(u,[-1,0]).^2,[1,0])/dx;
        duv_dy = circdiff(uv,[0,1])/dy;
        duv_dx = circdiff(uv,[1,0])/dx;
        dvv_dy = circdiff(shifthalf(v,[0,-1]).^2,[0,1])/dy;
    
        adv_u = -duu_dx - duv_dy;
        adv_v = -duv_dx - dvv_dy;
    
        % RHS
        rhs_u = adv_u + 0.5*u_diff;
        rhs_v = adv_v + 0.5*v_diff;
        if(r_flag)
            rhs_v = rhs_v - shifthalf(r,[0,-1])*g;
            rhs_r = -circdiff(shifthalf(r,[-1,0]).*u,[1,0])/dx - circdiff((shifthalf(r,[0,-1])).*v,[0,1])/dy - shifthalf(v,[0,1])*dRHOdy;
            rhs_r = rhs_r + 0.5*r_diff;
        end

        if(random_forcing)
            [u_forcing, v_forcing] = createVelocityForcing(Lx, Ly, Nx, Ny, epsilon_f, forcing_wavenumber, forcing_bandwidth, kmin, kmax);
            rhs_u = rhs_u + u_forcing / sqrt(dt);
            rhs_v = rhs_v + v_forcing / sqrt(dt);
        end
        
        % INVERT DIFFUSION WITH ADAMS-BASHFORTH FOR RHS
        if(random_forcing)
            ustar = real(ifft2(fft2(u + dt*((alpha * rhs_u - beta * rhs_u_old)))./(1 + 0.5*dt*nv*(Modified_k2_ori.^n_nv) + 0.5*dt*mu*(Modified_k2_ori.^n_mu))));
            vstar = real(ifft2(fft2(v + dt*((alpha * rhs_v - beta * rhs_v_old)))./(1 + 0.5*dt*nv*(Modified_k2_ori.^n_nv) + 0.5*dt*mu*(Modified_k2_ori.^n_mu))));
        else
            ustar = real(ifft2(fft2(u + dt*((alpha * rhs_u - beta * rhs_u_old)))./(1 + 0.5*dt*nu*Modified_k2_ori)));
            vstar = real(ifft2(fft2(v + dt*((alpha * rhs_v - beta * rhs_v_old)))./(1 + 0.5*dt*nu*Modified_k2_ori)));
        end
        rhs_u_old = rhs_u;
        rhs_v_old = rhs_v;

        if(r_flag)
            if exist('damp_r','var')
                if(damp_r == 1)
                    rhs_r = rhs_r - r.*damping;
                end
            end
            r = real(ifft2(fft2(r + dt*((alpha * rhs_r - beta * rhs_r_old)))./(1+0.5*dt*kp*Modified_k2_ori)));
            rhs_r_old = rhs_r;
        end
        

    
        % PROJECTION
        div = circdiff(ustar, [1,0])/dx + circdiff(vstar, [0,1])/dy;
            % solve Poisson equation using modified wave numbers for central difference
        p = real(ifft2(fft2(div)./Modified_k2))/dt;
    
            % pressue gradient
        dpdx = circdiff(p, [-1, 0])/dx;
        dpdy = circdiff(p, [0, -1])/dy;
        
            % projection step
        u = ustar - dt * dpdx;
        v = vstar - dt * dpdy;
        
        omega = circdiff(u,[0,-1])/dy - circdiff(v,[-1,0])/dx;
        
        CFL = max(max(abs(u)*dt/dx));
        if(isnan(CFL) || CFL == inf)
            cost = nan;
            disp('Forward simulation blew up!');
            return
        end
        Solver_F_postprocessing;
    end
    Solver_F_wrapup;
end