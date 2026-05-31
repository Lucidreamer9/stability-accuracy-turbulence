% 2D Navier-Stokes 4DVar data-assimilation case class.
% Decaying isotropic turbulence on a periodic square, staggered grid,
% modified wave numbers, Crank-Nicolson diffusion, Adams-Bashforth advection.
%
% Conventions: Methods MyMethod; Structures My_Structure;
% outside functions outsideFunctions; variables my_variables.

classdef DA_Case
    properties
        Setups
        Results
    end

    methods
        function obj = DA_Case(Setups)
            % Copy nested setup struct
            fields = fieldnames(Setups);
            for k = 1:numel(fields)
                fn = fieldnames(Setups.(fields{k}));
                for m = 1:numel(fn)
                    obj.Setups.(fields{k}).(fn{m}) = Setups.(fields{k}).(fn{m});
                end
            end

            % Grid + stencil + physical quantities
            Lx = obj.Setups.Parameters.Lx;
            Ly = obj.Setups.Parameters.Ly;
            Nx = obj.Setups.Parameters.Nx;
            Ny = obj.Setups.Parameters.Ny;
            xx = linspace(0, Lx, Nx+1);
            xv = 0.5*(xx(1:end-1) + xx(2:end));
            yy = linspace(0, Ly, Ny+1);
            yu = 0.5*(yy(1:end-1) + yy(2:end));
            xu = xx(1:end-1);
            yv = yy(1:end-1);
            xp = xv; yp = yu;
            xw = xu; yw = yv;
            dx = xx(2) - xx(1);
            dy = yy(2) - yy(1);
            obj.Setups.Parameters.dx = dx;
            obj.Setups.Parameters.dy = dy;
            [obj.Setups.Parameters.XU, obj.Setups.Parameters.YU] = meshgrid(xu,yu);
            [obj.Setups.Parameters.XV, obj.Setups.Parameters.YV] = meshgrid(xv,yv);
            [obj.Setups.Parameters.XP, obj.Setups.Parameters.YP] = meshgrid(xp,yp);
            obj.Setups.Parameters.xp = xp;
            obj.Setups.Parameters.yp = yp;
            obj.Setups.Parameters.xw = xw;
            obj.Setups.Parameters.yw = yw;
            obj.Setups.Parameters.xu = xu;
            obj.Setups.Parameters.yu = yu;
            obj.Setups.Parameters.xv = xv;
            obj.Setups.Parameters.yv = yv;

            obj.Setups.Parameters.nu = 1/obj.Setups.Parameters.Re;
            obj.Setups.Parameters.kp = 1/obj.Setups.Parameters.Pe;
            obj.Setups.Parameters.g  = -(1/obj.Setups.Parameters.Fr/2/pi)^2/obj.Setups.Parameters.dRHOdy;
            if(isnan(obj.Setups.Parameters.g))
                obj.Setups.Parameters.g = 0;
            end
            obj.Setups.Parameters.Nt = fix(obj.Setups.Parameters.T/obj.Setups.Parameters.dt) + 1;

            % Poisson solver wave numbers
            obj.Setups.Parameters.Modified_k2_ori = (2*sin(pi/Lx*dx*(0:(Nx-1))')/dx).^2 + (2*sin(pi/Ly*dy*(0:Ny-1))/dy).^2;
            obj.Setups.Parameters.Modified_k2_extend2 = (2*sin(pi/Lx*dx*(0:(Nx-1))')/dx).^2 + (2*sin(pi/Ly/2*dy*(0:(2*Ny-1)))/dy).^2;
            obj.Setups.Parameters.Modified_k2_extend2_nan = obj.Setups.Parameters.Modified_k2_extend2;
            obj.Setups.Parameters.Modified_k2_extend2_nan(1,1) = inf;
            obj.Setups.Parameters.Modified_k2_extend4 = (2*sin(pi/Lx*dx*(0:(Nx-1))')/dx).^2 + (2*sin(pi/Ly/4*dy*(0:(4*Ny-1)))/dy).^2;
            obj.Setups.Parameters.Modified_kx2 = (2*sin(pi/Lx*dx*(0:(Nx-1))')/dx).^2;
            obj.Setups.Parameters.Modified_k2 = obj.Setups.Parameters.Modified_k2_ori;
            obj.Setups.Parameters.Modified_k2(1,1) = inf;
        end

        function varargout = Solver(obj, solver_type, varargin)
            switch solver_type
                case "F"
                    [varargout{1:nargout}] = F_simulation(varargin{:});
                case "AD"
                    [varargout{1:nargout}] = AD_simulation(varargin{:});
                case "LF"
                    [varargout{1:nargout}] = LF_simulation(varargin{:});
                otherwise
                    error('Invalid solver type. Use ''F'', ''LF'' or ''AD''.');
            end
        end

        function [u0, v0, r0] = InputFromControlVector(obj, input)
            Nx = obj.Setups.Parameters.Nx;
            Ny = obj.Setups.Parameters.Ny;
            switch obj.Setups.Flags.control_vector_type
                case "uv"
                    u0 = reshape(input(1:Nx*Ny),[Nx,Ny]);
                    v0 = reshape(input(Nx*Ny+1:2*Nx*Ny),[Nx,Ny]);
                    r0 = zeros(size(u0));
                otherwise
                    error('Unsupported control_vector_type "%s". This release only supports "uv".', ...
                          obj.Setups.Flags.control_vector_type);
            end
        end

        function [u0, v0, r0] = InputFromPsi(obj, input)
            Nx = obj.Setups.Parameters.Nx;
            Ny = obj.Setups.Parameters.Ny;
            dx = obj.Setups.Parameters.dx;
            dy = obj.Setups.Parameters.dy;
            psi = reshape(input,[Nx,Ny]);
            u0 =  circdiff(psi,[0,1])/dy;
            v0 = -circdiff(psi,[1,0])/dx;
            r0 = zeros(size(u0));
        end

        function control = ControlVectorFromInput(obj, u, v, ~, ~)
            switch obj.Setups.Flags.control_vector_type
                case "uv"
                    control = [u(:); v(:)];
                otherwise
                    error('Unsupported control_vector_type "%s". This release only supports "uv".', ...
                          obj.Setups.Flags.control_vector_type);
            end
        end

        function [obj, ic] = InitialCondition(obj, varargin)
            if (nargin > 1)
                psi = reshape(varargin{1}, [obj.Setups.Parameters.Nx, obj.Setups.Parameters.Ny]);
            else
                psi = decay_ic(obj.Setups.Parameters.Nx, obj.Setups.Parameters.Ny, obj.Setups.Parameters.dx);
            end
            obj.Results.IC.psi =  psi;
            obj.Results.IC.u   =  circdiff(psi,[0,1])/obj.Setups.Parameters.dy;
            obj.Results.IC.v   = -circdiff(psi,[1,0])/obj.Setups.Parameters.dx;
            xx                 =  obj.Setups.Parameters.xp - obj.Setups.Parameters.Lx/2;
            yy                 =  obj.Setups.Parameters.yp - obj.Setups.Parameters.Ly/2;
            [X, Y]             =  meshgrid(xx,yy);
            R2                 =  X.^2 + Y.^2;
            radius             =  0.1;
            obj.Results.IC.r   =  1/sqrt(2*pi)/radius*exp(-R2/radius^2/2);
            ic                 =  psi(:);
        end

        function GBs = CheckStorage(obj)
            GBs = obj.Setups.Parameters.Nt*obj.Setups.Parameters.Nx*obj.Setups.Parameters.Ny*(3 - obj.Setups.Flags.r_flag)*8*2/1024/1024/1024;
            disp("This case should take " + num2str(GBs) + " GB of space.");
            if(GBs > 32)
                disp("Warning: Required space too large");
            end
        end

        function [U, V, R] = ForwardSimulation(obj, dryrun, varargin)
            if(nargin > 2)
                [u0, v0, r0] = InputFromControlVector(obj, varargin{1});
            else
                u0 = obj.Results.IC.u;
                v0 = obj.Results.IC.v;
                r0 = obj.Results.IC.r;
            end
            obj.CheckStorage();

            if(dryrun)
                obj.Setups.Data_Assimilation.DA_flag = 0;
            else
                obj.Setups.Data_Assimilation.DA_flag = 1;
            end
            [U, V, R, ~] = Solver(obj, "F", u0, v0, r0, obj.Setups);
        end

        function [u_adj, v_adj, r_adj, U_adj, V_adj, R_adj] = AdjointSimulation(obj, U_base, V_base, R_base, varargin)
            if(nargin > 4)
                [u0, v0, r0] = InputFromControlVector(obj, varargin{1});
            else
                u0 = obj.Results.IC.u;
                v0 = obj.Results.IC.v;
                r0 = obj.Results.IC.r;
            end
            obj.CheckStorage();
            [u_adj, v_adj, r_adj, U_adj, V_adj, R_adj] = Solver(obj, "AD", u0, v0, r0, U_base, V_base, R_base, obj.Setups);
        end

        function [cost, grad] = ForwardAdjoint(obj, input)
            obj.Setups.Data_Assimilation.DA_flag = 2;
            [u0, v0, r0]    = InputFromControlVector(obj, input);

            [U, V, R, cost] = Solver(obj, "F", u0, v0, r0, obj.Setups);
            if(isnan(cost))
                cost = 1e20;
                grad = ones(size(input))*1e20;
                return
            end
            % H1 regularizer on the initial velocity field
            lambda_reg = 0;
            if isfield(obj.Setups,'Data_Assimilation') && isfield(obj.Setups.Data_Assimilation,'lambda')
                lambda_reg = obj.Setups.Data_Assimilation.lambda;
            end
            if lambda_reg ~= 0
                dx = obj.Setups.Parameters.dx;
                dy = obj.Setups.Parameters.dy;
                du_dx = circdiff(u0,[1,0])/dx;
                du_dy = circdiff(u0,[0,1])/dy;
                dv_dx = circdiff(v0,[1,0])/dx;
                dv_dy = circdiff(v0,[0,1])/dy;
                cost = cost + 0.5*lambda_reg * dx * dy * ( sum(sum(du_dx.^2 + du_dy.^2)) + sum(sum(dv_dx.^2 + dv_dy.^2)) );
            end
            [u_AD, v_AD, r_AD] = Solver(obj, "AD", u0*0, v0*0, r0*0, U, V, R, obj.Setups);
            grad = ControlVectorFromInput(obj, u_AD, v_AD, r_AD, 1);
        end

        function [obj, updated_control_vector] = PerformOptimization(obj, IG)
            options = optimset('MaxIter',obj.Setups.Data_Assimilation.iteration_number,'Display','iter','GradObj','on','TolFun',1e-20);
            output_filename = obj.Setups.Flags.folder + obj.Setups.Flags.control_vector_type + ".txt";
            options.OutputFile = output_filename;
            updated_control_vector = fminlbfgs(@(input) obj.ForwardAdjoint(input), IG, options);
            obj.Results.initial_guess = IG;
            obj.Results.updated_control_vector = updated_control_vector;
        end

        function accuracy = ValidateLinearizedForward(obj, input, time_step)
            [u0, v0, r0]          = InputFromControlVector(obj, input);

            obj.Setups.Parameters.Nt = time_step;
            obj.Setups.Post_Processing.history_flag = 1;
            obj.Setups.Data_Assimilation.DA_flag = 0;

            [U, V, R]             = Solver(obj, "F", u0, v0, r0, obj.Setups);
            [du0, dv0, dr0]       = InputFromControlVector(obj, randn(size(input)));
            [du_LF, dv_LF, dr_LF] = Solver(obj, "LF", du0, dv0, dr0, U, V, R, obj.Setups);

            epsilon_pool = 10.^(-1:-1:-15);
            count = 0;
            accuracy = [];
            figure(100)
            for epsilon = epsilon_pool
                count = count + 1;
                [Up, Vp, Rp] = Solver(obj, "F", u0 + epsilon * du0, v0 + epsilon * dv0, r0 + epsilon * dr0, obj.Setups);
                du_F = (Up(:,:,end) - U(:,:,end))/epsilon;
                dv_F = (Vp(:,:,end) - V(:,:,end))/epsilon;
                dr_F = (Rp(:,:,end) - R(:,:,end))/epsilon;
                accuracy(end+1) = min([mean(max(du_F - du_LF)), mean(max(dv_F - dv_LF))]);
                plot(epsilon_pool(1:count), accuracy, 'k', 'LineWidth',2);
                pbaspect([1,1,1]);
                set(gca,'YScale','log','XScale','log')
                pause(0.1);
            end
        end

        function accuracy = ValidateLinearizedForwardAdjoint(obj, input, time_step)
            [u0, v0, r0] = InputFromControlVector(obj, input);

            obj.Setups.Parameters.Nt = time_step;
            obj.Setups.Post_Processing.history_flag = 1;
            obj.Setups.Data_Assimilation.DA_flag = 0;

            [U, V, R]                      = Solver(obj, "F", u0, v0, r0, obj.Setups);
            [du0, dv0, dr0]                = InputFromControlVector(obj, randn(size(input)));
            [du0, dv0]                     = divergence_free_projection(du0, dv0, obj.Setups);
            input_f0                       = ControlVectorFromInput(obj, du0, dv0, dr0, 0);
            [udagger0, vdagger0, rdagger0] = InputFromControlVector(obj, randn(size(input)));
            [udagger0, vdagger0]           = divergence_free_projection(udagger0, vdagger0, obj.Setups);
            input_a0                       = ControlVectorFromInput(obj, udagger0, vdagger0, rdagger0, 1);

            [du_LF, dv_LF, dr_LF]          = Solver(obj, "LF", du0, dv0, dr0, U, V, R, obj.Setups);
            input_fT                       = ControlVectorFromInput(obj, du_LF, dv_LF, dr_LF, 0);
            [u_AD, v_AD, r_AD]             = Solver(obj, "AD", udagger0, vdagger0, rdagger0, U, V, R, obj.Setups);
            input_aT                       = ControlVectorFromInput(obj, u_AD, v_AD, r_AD, 1);

            LHS = sum(input_f0.*input_aT);
            RHS = sum(input_fT.*input_a0);
            accuracy = abs(LHS-RHS)/LHS;
        end

        function accuracy = ValidateAdjointGradient(obj, input)
            [u0, v0, r0]    = InputFromControlVector(obj, input);

            obj.Setups.Data_Assimilation.DA_flag = 2;
            [U, V, R, cost] = Solver(obj, "F", u0, v0, r0, obj.Setups);

            control_perturbation = randn(size(input));
            control_perturbation = control_perturbation/sqrt(sum(control_perturbation.^2));
            [du0, dv0, dr0]      = InputFromControlVector(obj, control_perturbation);
            [du0,dv0]            = divergence_free_projection(du0, dv0, obj.Setups);
            [u_AD, v_AD, r_AD]   = Solver(obj, "AD", u0*0, v0*0, r0*0, U, V, R, obj.Setups);
            adjoint_gradient     = sum(ControlVectorFromInput(obj, u_AD, v_AD, r_AD, 1).*control_perturbation);

            accuracy = [];
            epsilon_pool = 10.^(-1:-0.5:-15);
            count = 0;
            figure(200)
            for epsilon = 10.^(-1:-1:-15)
                count = count + 1;
                [~, ~, ~, cost_perturb] = Solver(obj, "F", u0 + epsilon * du0, v0 + epsilon * dv0, r0 + epsilon * dr0, obj.Setups);
                gradient = (cost_perturb - cost) / epsilon;
                accuracy(end+1) = abs(gradient - adjoint_gradient) / abs(adjoint_gradient);
                plot(epsilon_pool(1:count), accuracy, 'k', 'LineWidth',2);
                pbaspect([1,1,1]);
                set(gca,'YScale','log','XScale','log')
                pause(0.1);
            end
        end
    end
end
