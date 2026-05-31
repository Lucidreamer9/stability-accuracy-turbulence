function [udagger, vdagger, rdagger, U_adj, V_adj, R_adj] = AD_simulation(udagger0, vdagger0, rdagger0, U, V, R, Setups)
    % Adjoint two-dimensional Navier-Stokes solver,
    % udagger0, vdagger0, rdagger0 are initial conditions
    % U, V, R are the base flow.
    % There is currently no option for visualization in adjoint simulation

    Solver_AD_initialization;

    %% main loop
    for steps = 1:Nt - 1
        % Adams-Bashforth
        if(steps == 1)
            alpha = alpha_record; beta = 0;
        end
        if(steps == 2)
            alpha = alpha_record; beta = alpha - 1;
        end
        if(steps == Nt - 1)
            alpha = 1; beta = alpha_record - 1;
        end

        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)

            % insert the source term from the observations
            if(DA_flag == 2 && ismember(Nt + 1 - steps, DA_obs_tlist))
                if(DA_obs_record_full)
                    obs_used_iter = Nt - steps;
                else
                    obs_used_iter = obs_used_iter - 1;
                end
                if(control_vector_type == "psi(r)")
                    rdagger = rdagger + (R(:,:,Nt + 1 - steps) - R_obs(:,:,obs_used_iter));
                else
                    udagger = udagger + (U(:,:,Nt + 1 - steps) - U_obs(:,:,obs_used_iter)).*DA_mask_u;
                    vdagger = vdagger + (V(:,:,Nt + 1 - steps) - V_obs(:,:,obs_used_iter)).*DA_mask_v;
                end
            end
    
            % adjoint projection
            div = circdiff(udagger, [1,0])/dx + circdiff(vdagger, [0,1])/dy;
                % solve Poisson equation using modified wave numbers for
            pdagger = real(ifft2(fft2(div)./Modified_k2));
        
            dpdagger_dx = circdiff(pdagger, [-1, 0])/dx;
            dpdagger_dy = circdiff(pdagger, [0, -1])/dy;
            
            udagger = udagger - dpdagger_dx;
            vdagger = vdagger - dpdagger_dy;
            
            % adjoint diffusion
            udagger = real(ifft2(fft2(udagger)./(1+0.5*dt*nu*Modified_k2_ori)));
            vdagger = real(ifft2(fft2(vdagger)./(1+0.5*dt*nu*Modified_k2_ori)));
        end
        if(r_flag) rdagger = real(ifft2(fft2(rdagger)./(1+0.5*dt*kp*Modified_k2_ori))); end

        % Adams-Bashforth
        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
            udagger_star = alpha * udagger - beta * udagger_old;
            vdagger_star = alpha * vdagger - beta * vdagger_old;
        end
        if(r_flag)
            rdagger_star = alpha * rdagger - beta * rdagger_old; 
        end

        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
            % update old variable
            udagger_old = udagger;
            vdagger_old = vdagger;
        end
        if(r_flag) rdagger_old = rdagger; end

        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
            udagger = udagger_star;
            vdagger = vdagger_star;
        end
        if(r_flag) rdagger = rdagger_star; end

        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
            udagger_diff = (circshift(udagger,[1,0]) + circshift(udagger,[-1,0]) - 2*udagger)/dx/dx + (circshift(udagger,[0,1]) + circshift(udagger,[0,-1]) - 2*udagger)/dy/dy;
            vdagger_diff = (circshift(vdagger,[1,0]) + circshift(vdagger,[-1,0]) - 2*vdagger)/dx/dx + (circshift(vdagger,[0,1]) + circshift(vdagger,[0,-1]) - 2*vdagger)/dy/dy;
        end
        
        if(r_flag)
            rdagger_diff = (circshift(rdagger,[1,0]) + circshift(rdagger,[-1,0]) - 2*rdagger)/dx/dx + (circshift(rdagger,[0,1]) + circshift(rdagger,[0,-1]) - 2*rdagger)/dy/dy;
        end

        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
            uvdagger = circdiff(udagger,[0,-1])/dy + circdiff(vdagger,[-1,0])/dx;
            rhs_udagger = -2*shifthalf(shifthalf(U(:,:,Nt-steps),[-1,0]).*circdiff(udagger,[-1,0]),[1,0])/dx;
            rhs_udagger = rhs_udagger - shifthalf(shifthalf(V(:,:,Nt-steps),[-1,0]).*uvdagger,[0,1]);
            rhs_vdagger = - shifthalf(shifthalf(U(:,:,Nt-steps),[0,-1]).*uvdagger,[1,0]);
            rhs_vdagger = rhs_vdagger - 2*shifthalf(shifthalf(V(:,:,Nt-steps),[0,-1]).*circdiff(vdagger,[0,-1]),[0,1])/dy;
        end
        if(r_flag)
            if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
                rhs_udagger = rhs_udagger - shifthalf(R(:,:,Nt-steps),[-1,0]).*circdiff(rdagger,[-1,0])/dx;
                rhs_vdagger = rhs_vdagger - shifthalf(R(:,:,Nt-steps),[0,-1]).*circdiff(rdagger,[0,-1])/dy;
                rhs_vdagger = rhs_vdagger - dRHOdy*shifthalf(rdagger,[0,-1]);
            end
            rhs_rdagger = -shifthalf(U(:,:,Nt-steps).*circdiff(rdagger,[-1,0]),[1,0])/dx;
            rhs_rdagger = rhs_rdagger - shifthalf(V(:,:,Nt-steps).*circdiff(rdagger,[0,-1]),[0,1])/dy;
            rhs_rdagger = rhs_rdagger - g*shifthalf(vdagger,[0,1]);
            
            rhs_rdagger = rhs_rdagger + 0.5*kp*rdagger_diff;
            
            rdagger = rdagger_old + rhs_rdagger*dt;
            
            if exist('damp_r','var')
                if(damp_r == 1)
                    rdagger = rdagger - dt*rdagger_star.*damping;
                end
            end
        end

        if (~exist('ignore_adjoint_uv','var') || ignore_adjoint_uv == 0)
            rhs_udagger = rhs_udagger + 0.5*nu*udagger_diff;
            rhs_vdagger = rhs_vdagger + 0.5*nu*vdagger_diff;
    
            udagger = udagger_old + rhs_udagger*dt;
            vdagger = vdagger_old + rhs_vdagger*dt;
        else
            udagger = udagger0*0;
            vdagger = vdagger0*0;
        end
        
    Solver_AD_postprocessing;
    end

    % Add H1 regularizer contribution at initial time: -lambda * Δ(u0), -lambda * Δ(v0)
    % Uses the initial forward state U(:,:,1), V(:,:,1) and grid spacings dx, dy.
    if exist('lambda','var')
        lambda_reg = lambda;
    else
        lambda_reg = 0;
    end
    if lambda_reg ~= 0
        disp(lambda_reg);
        % lap_u0 = (circshift(U(:,:,1),[1,0]) + circshift(U(:,:,1),[-1,0]) - 2*U(:,:,1))/(dx*dx) + (circshift(U(:,:,1),[0,1]) + circshift(U(:,:,1),[0,-1]) - 2*U(:,:,1))/(dy*dy);
        % lap_v0 = (circshift(V(:,:,1),[1,0]) + circshift(V(:,:,1),[-1,0]) - 2*V(:,:,1))/(dx*dx) + (circshift(V(:,:,1),[0,1]) + circshift(V(:,:,1),[0,-1]) - 2*V(:,:,1))/(dy*dy);
        % lap_u0 = (circdiff(U(:,:,1),[1,0])/dx)^2 + (circdiff(U(:,:,1),[0,1])/dy)^2;
        % lap_v0 = (circdiff(V(:,:,1),[1,0])/dx)^2 + (circdiff(V(:,:,1),[0,1])/dy)^2;
        u0=U(:,:,1);
        v0=V(:,:,1);
        lap_u0 = real(ifft2(-Modified_k2_ori.*fft2(u0)));
        lap_v0 = real(ifft2(-Modified_k2_ori.*fft2(v0)));
        udagger = udagger - lambda_reg *dx*dy* lap_u0;
        vdagger = vdagger - lambda_reg *dx*dy* lap_v0;
    end

    % a final projection step to make sure the solution space is divergence-free

    div = circdiff(udagger, [1,0])/dx + circdiff(vdagger, [0,1])/dy;
        % solve Poisson equation using modified wave numbers for
    pdagger = real(ifft2(fft2(div)./Modified_k2));

    dpdagger_dx = circdiff(pdagger, [-1, 0])/dx;
    dpdagger_dy = circdiff(pdagger, [0, -1])/dy;
    
    udagger = udagger - dpdagger_dx;
    vdagger = vdagger - dpdagger_dy;
    
end