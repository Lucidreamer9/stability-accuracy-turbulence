%% unroll the setup parameters
fields = fieldnames(Setups);
for k = 1:numel(fields)
    fn = fieldnames(Setups.(fields{k}));
    for m = 1:numel(fn)
        eval(string(fn{m}) + "=" + "Setups.(fields{k}).(fn{m});");
    end
end

% Assign initial condition 
du = du0; dv = dv0; dr = dr0;

% time stepping
rhs_dv_old = 0;
rhs_du_old = 0;
rhs_dr_old = 0;

if(time_AB == 1)
    alpha_record = 1;
else
    alpha_record = 1.5;
end

% The first step is always Forward Euler
alpha = 1; beta = 0;