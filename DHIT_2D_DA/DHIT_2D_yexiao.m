% 2D decaying isotropic turbulence + 4DVar data assimilation demo
%
% Two-phase workflow controlled by Data_Assimilation.DA_flag:
%   DA_flag = 1 : run truth simulation, save InitialCondition.mat and obs.dat
%   DA_flag = 2 : load truth IC and observations, run 4DVar via L-BFGS

clear all;
addpath(genpath(pwd));

%% Flow Type
Flags.flow_type           = "decaying isotropic";
Flags.r_flag              = 0;  % turn off the density solver
Flags.time_AB             = 2;  % second-order Adams-Bashforth
Flags.control_vector_type = "uv";

Data_Assimilation.lambda  = 0;  % H1 regularizer weight (0 = none)
Flags.random_forcing      = 0;

Flags.folder = fullfile(pwd, "cases", "DA_DHIT_1000_"+Flags.control_vector_type) + filesep;
if ~exist(Flags.folder,'dir')
    mkdir(Flags.folder);
end

%% Grid Parameters
Parameters.Lx = 1;
Parameters.Ly = 1;
Parameters.Nx = 64;
Parameters.Ny = 64;
Parameters.dt = 0.001;
Parameters.T  = 0.2;

%% Physical Parameters
Parameters.Re     = 1000;   % Reynolds number
Parameters.Pe     = 200;    % Peclet number
Parameters.Fr     = inf;    % Froude number
Parameters.dRHOdy = 0;      % Density gradient

%% Post-processing setup
Post_Processing.image_steps    = 2;
Post_Processing.video_steps    = 0;
Post_Processing.paraview_steps = 0;
Post_Processing.record_steps   = 0;
Post_Processing.history_flag   = 1;
Post_Processing.vis_option     = 1;
Post_Processing.Nsubplots      = 4;
Post_Processing.u_image        = 1;
Post_Processing.v_image        = 1;
Post_Processing.r_image        = 1;
Post_Processing.o_image        = 1;
Post_Processing.plot_xmin      = 1;
Post_Processing.plot_xmax      = Parameters.Nx;
Post_Processing.plot_ymin      = 1;
Post_Processing.plot_ymax      = Parameters.Ny;
Post_Processing.plot_xgap      = 2;
Post_Processing.plot_ygap      = 2;
Post_Processing.display_steps  = 0;
Post_Processing.video_file     = "forward_movie";
Post_Processing.paraview_file  = "paraview_output_";

%% Data assimilation setup
validation_flag                      = 0;
Data_Assimilation.DA_flag            = 2;   % 1 = generate truth + obs, 2 = run 4DVar
obs_gap                              = 1;
Data_Assimilation.obs_gap            = obs_gap;
Data_Assimilation.DA_mask_u          = zeros([Parameters.Nx, Parameters.Ny]);
Data_Assimilation.DA_mask_u(1:obs_gap:end,1:obs_gap:end) = 1;
Data_Assimilation.DA_mask_v          = zeros([Parameters.Nx, Parameters.Ny]);
Data_Assimilation.DA_mask_v(1:obs_gap:end,1:obs_gap:end) = 1;
Data_Assimilation.DA_obs_tlist       = fix(Parameters.T/Parameters.dt) + 1;
Data_Assimilation.DA_obs_file        = "obs";
Data_Assimilation.DA_obs_record_full = 0;
Data_Assimilation.obs_p              = 0;
Data_Assimilation.iteration_number   = 100;

%% Instantiate the DA case
Setups.Parameters        = Parameters;
Setups.Flags             = Flags;
Setups.Post_Processing   = Post_Processing;
Setups.Data_Assimilation = Data_Assimilation;

Isotropic_DA = DA_Case(Setups);

%% Phase 1: generate truth IC and observations
if(Data_Assimilation.DA_flag == 1)
    [Isotropic_DA, ic_psi] = Isotropic_DA.InitialCondition();
    save(Flags.folder + "InitialCondition","ic_psi")
    [U, V, R] = Isotropic_DA.ForwardSimulation(0);
end

%% Phase 2: load IC + obs and run 4DVar
if(Data_Assimilation.DA_flag == 2)
    load(Flags.folder + "InitialCondition", "ic_psi");
    Isotropic_DA = Isotropic_DA.InitialCondition(ic_psi);
end

%% Optional: gradient / adjoint validation
if(validation_flag)
    [u0, v0, r0] = Isotropic_DA.InputFromPsi(ic_psi); %#ok<UNRCH>
    ic = Isotropic_DA.ControlVectorFromInput(u0, v0, r0, 0);
    accuracy_linearization = Isotropic_DA.ValidateLinearizedForward(ic, 10);
    accuracy_adjoint       = Isotropic_DA.ValidateLinearizedForwardAdjoint(ic, 10);
    accuracy_gradient      = Isotropic_DA.ValidateAdjointGradient(ic*0.8);
end

%% Run the L-BFGS optimization (Phase 2 only)
if(Data_Assimilation.DA_flag == 2)
    load(Flags.folder + Data_Assimilation.DA_obs_file)
    [u_guess, v_guess] = inter_sparse_obs(U_obs(:,:,1), V_obs(:,:,1), Isotropic_DA.Setups, obs_gap);

    fprintf('U_obs size: %s\n', mat2str(size(U_obs)));
    fprintf('DA_obs_tlist: %s\n', mat2str(Data_Assimilation.DA_obs_tlist));

    initial_guess = Isotropic_DA.ControlVectorFromInput(u_guess, v_guess, u_guess*0, 0);

    [Isotropic_DA, updated_control_vector] = Isotropic_DA.PerformOptimization(initial_guess);
    save(Flags.folder + "DA_results_" + string(Flags.control_vector_type)+".mat", ...
         'initial_guess','updated_control_vector','Setups');
end
