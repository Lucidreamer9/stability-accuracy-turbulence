function [du,dv,dr] = LF_simulation(du0, dv0, dr0, U, V, R, Setups)
    % Linearized two-dimensional Navier-Stokes solver,
    % du0, dv0, dr0 are initial conditions
    % U, V, R are the base flow.
    % There is currently no option for visualization in linearized forward simulation

    Solver_LF_initialization;

    for steps = 1:Nt - 1
        if(steps > 1)
            alpha = alpha_record; 
            beta = alpha_record - 1;
        end
    
        % EXPLICIT DIFFUSION
        du_diff = (circshift(du,[1,0]) + circshift(du,[-1,0]) - 2*du)/dx/dx + (circshift(du,[0,1]) + circshift(du,[0,-1]) - 2*du)/dy/dy;
        dv_diff = (circshift(dv,[1,0]) + circshift(dv,[-1,0]) - 2*dv)/dx/dx + (circshift(dv,[0,1]) + circshift(dv,[0,-1]) - 2*dv)/dy/dy;
        if(r_flag) 
            dr_diff = (circshift(dr,[1,0]) + circshift(dr,[-1,0]) - 2*dr)/dx/dx + (circshift(dr,[0,1]) + circshift(dr,[0,-1]) - 2*dr)/dy/dy;
        end
        
        % ADVECTION TERMS
        duv = shifthalf(U(:,:,steps),[0,-1]).*shifthalf(dv,[-1,0]) + shifthalf(du,[0,-1]).*shifthalf(V(:,:,steps),[-1,0]);
    
        dduu_dx = 2.0*circdiff(shifthalf(du,[-1,0]).*shifthalf(U(:,:,steps),[-1,0]),[1,0])/dx;
        dduv_dy = circdiff(duv,[0,1])/dy;
        dduv_dx = circdiff(duv,[1,0])/dx;
        ddvv_dy = 2.0*circdiff(shifthalf(dv,[0,-1]).*shifthalf(V(:,:,steps),[0,-1]),[0,1])/dy;
        
        adv_du = -dduu_dx - dduv_dy;
        adv_dv = -dduv_dx - ddvv_dy;
    
        % RHS
        rhs_du = adv_du + 0.5*nu*du_diff;
        rhs_dv = adv_dv + 0.5*nu*dv_diff;
        if(r_flag)
            rhs_dv = rhs_dv - shifthalf(dr,[0,-1])*g;
            rhs_dr = -circdiff(shifthalf(R(:,:,steps),[-1,0]).*du,[1,0])/dx - circdiff((shifthalf(R(:,:,steps),[0,-1])).*dv,[0,1])/dy;
            rhs_dr = rhs_dr - circdiff(shifthalf(dr,[-1,0]).*U(:,:,steps),[1,0])/dx - circdiff((shifthalf(dr,[0,-1])).*V(:,:,steps),[0,1])/dy - shifthalf(dv,[0,1])*dRHOdy;
            rhs_dr = rhs_dr + 0.5*kp*dr_diff;
        end
        
        % INVERT DIFFUSION WITH ADAMS-BASHFORTH FOR RHS

        dustar = real(ifft2(fft2(du + dt*((alpha * rhs_du - beta * rhs_du_old)))./(1+0.5*dt*nu*Modified_k2_ori)));
        dvstar = real(ifft2(fft2(dv + dt*((alpha * rhs_dv - beta * rhs_dv_old)))./(1+0.5*dt*nu*Modified_k2_ori)));
        
        rhs_du_old = rhs_du;
        rhs_dv_old = rhs_dv;

        if(r_flag)
            dr     = real(ifft2(fft2(dr + dt*((alpha * rhs_dr - beta * rhs_dr_old)))./(1+0.5*dt*kp*Modified_k2_ori)));
            rhs_dr_old = rhs_dr;
        end

        % PROJECTION
        div = circdiff(dustar, [1,0])/dx + circdiff(dvstar, [0,1])/dy;
            % solve Poisson equation using modified wave numbers for central difference
        dp = real(ifft2(fft2(div)./Modified_k2));
    
        ddpdx = circdiff(dp, [-1, 0])/dx;
        ddpdy = circdiff(dp, [0, -1])/dy;
        
            % projection step
        du = dustar - ddpdx;
        dv = dvstar - ddpdy;

        Solver_LF_postprocessing;
    end
end