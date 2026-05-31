CFL = max(max(abs(udagger)*dt/dx));
if(display_steps > 0 && mod(steps, display_steps) == 0)
    disp("A: "+ num2str(steps) + ", maximum CFL: " + num2str(CFL)); 
end
% record history for adjoint variables (stored backward in time: t=T -> t=0)
if(history_flag)
    U_adj(:,:,end+1) = udagger;
    V_adj(:,:,end+1) = vdagger;
    if(r_flag)
        R_adj(:,:,end+1) = rdagger;
    end
end