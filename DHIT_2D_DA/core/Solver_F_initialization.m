%% Solver_initialization
% Similar for every solver
% unroll the setup parameters
fields = fieldnames(Setups);
for k = 1:numel(fields)
    fn = fieldnames(Setups.(fields{k}));
    for m = 1:numel(fn)
        eval(string(fn{m}) + "=" + "Setups.(fields{k}).(fn{m});");
    end
end

% assign initial condition
u = u0; v = v0; r = r0;

% time stepping
if(time_AB == 1)
    alpha_record = 1;
else
    alpha_record = 1.5;
end

% initialize the old rhs
rhs_v_old = 0;
rhs_u_old = 0;
if(r_flag) rhs_r_old = 0; end

% diagnostic information
diagnostic_kinetic_energy = [];
diagnostic_enstropy = [];

% record the velocity fields
if(record_steps)
    U_rec = [];
    V_rec = [];
    DUDX_rec = [];
    DVDX_rec = [];
    DUDY_rec = [];
    DVDY_rec = [];
    if(r_flag) R_rec = []; end
    O_rec = [];
end    

if(DA_flag == 1)
    % record obs file
    U_obs = zeros([Nx,Ny,0]);
    V_obs = zeros([Nx,Ny,0]);
    R_obs = zeros([Nx,Ny,0]);
    if(obs_p)
        dpdx_obs = zeros([Nx,Ny,0]);
        dpdy_obs = zeros([Nx,Ny,0]);
        p_obs    = zeros([Nx,Ny,0]);
    end
end
    
if(DA_flag >= 2 || history_flag)
    U = u0; V = v0; R = r0;
end

% load observation data
cost = 0;
if(DA_flag == 2)
    load(folder + DA_obs_file);
    obs_used_iter = 0;
end

if(video_steps)
    vobj = VideoWriter(folder + video_file + ".avi");
    open(vobj);
end

% The first step would always be Euler
alpha = 1; beta = 0;

