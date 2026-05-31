%% In-Solver POST-PROCESSING
domega = circdiff(du,[0,-1])/dy - circdiff(dv,[-1,0])/dx;

% diagnostics
if(display_steps > 0 && mod(steps, display_steps) == 0)
    CFL = max(max(abs(du)*dt/dx));
    disp("LF: "+ num2str(steps) + ", maximum CFL: " + num2str(CFL));
end