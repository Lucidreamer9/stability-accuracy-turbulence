%% unroll the setup parameters
fields = fieldnames(Setups);
for k = 1:numel(fields)
    fn = fieldnames(Setups.(fields{k}));
    for m = 1:numel(fn)
        eval(string(fn{m}) + "=" + "Setups.(fields{k}).(fn{m});");
    end
end

% initial condition assigned
udagger = udagger0; vdagger = vdagger0; rdagger = rdagger0;

% Adams-Bashforth
if(time_AB == 1)
    alpha_record = 1;
else
    alpha_record = 1.5;
end

udagger_old = 0; vdagger_old = 0; rdagger_old = 0;

% The first step is always Forward Euler
alpha = 1; beta = 0;

if(DA_flag == 2)
    load(folder + DA_obs_file);
    obs_used_iter = size(U_obs, 3) + 1;
end

if(history_flag)
    R_adj = [];
    U_adj = [];
    V_adj = [];
    % Store initial condition
    U_adj(:,:,1) = udagger;
    V_adj(:,:,1) = vdagger;
    if(r_flag)
        R_adj(:,:,1) = rdagger;
    end
end