%% In-Solver POST-PROCESSING

if(display_steps > 0 && mod(steps, display_steps) == 0) 
        disp("F: "+ num2str(steps) + ", maximum CFL: " + num2str(CFL));
    if(flow_type == "forced isotropic")
        diagnostic_enstropy(end + 1) = dx*dy*sum(sum(omega.^2)) / Lx / Ly / 2 /forcing_wavenumber^2;
        diagnostic_kinetic_energy(end + 1) = dx*dy*sum(sum(u.^2 + v.^2)) / Lx / Ly / 2;
        disp("enstropy per unit area: " + num2str(diagnostic_enstropy(end))); 
        disp("kinetic energy per unit area: " + num2str(diagnostic_kinetic_energy(end))); 
        figure(404)
        plot(dt*(1:steps), diagnostic_kinetic_energy, 'b');
        hold on
        plot(dt*(1:steps), diagnostic_enstropy, 'r');
        hold off
    end
end

% record history for data assimilation
if(DA_flag >= 2 || history_flag)
    U(:,:,end+1) = u;
    V(:,:,end+1) = v;
    if(r_flag)
        R(:,:,end+1) = r;
    end
end

% record observation data
if(DA_flag == 1)
    if(DA_obs_record_full || ismember(steps + 1, DA_obs_tlist))
        U_obs(:,:,end+1) = u;
        V_obs(:,:,end+1) = v;
        if(r_flag)
            R_obs(:,:,end+1) = r;
        end
        % Fetch gradp as a function of time
        if(obs_p)
            dpdx_obs(:,:,end+1) = dpdx;
            p_obs(:,:,end+1)    = p;
            dpdy_obs(:,:,end+1) = dpdy;
        end
    end
end

% update the cost function based on observations
if(DA_flag == 2 && ismember(steps + 1, DA_obs_tlist))
    if(DA_obs_record_full)
        obs_used_iter = steps + 1;
    else
        obs_used_iter = obs_used_iter + 1;
    end
    if(control_vector_type == "psi(r)")
        cost = cost + 0.5 * sum(sum((r - R_obs(:,:,obs_used_iter)).^2));
    else
        cost = cost + 0.5 * sum(sum((u - U_obs(:,:,obs_used_iter)).^2.*DA_mask_u));
        cost = cost + 0.5 * sum(sum((v - V_obs(:,:,obs_used_iter)).^2.*DA_mask_v));
    end
end

if(record_steps > 0 && mod(steps, record_steps) == 0)
    utmp = shifthalf(u,[0,-1]);
    vtmp = shifthalf(v,[-1,0]);
    if(r_flag) rtmp = shifthalf(r,[-1,-1]); end
    dudxtmp = shifthalf(circdiff(u,[1,0]),[-1,-1])/dx;
    dudytmp = -circdiff(u,[0,-1])/dy;
    dvdxtmp = -circdiff(v,[-1,0])/dx;
    dvdytmp = shifthalf(circdiff(v,[0,1]),[-1,-1])/dy;
    U_rec(:,:,end+1) = utmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
    V_rec(:,:,end+1) = vtmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
    DUDX_rec(:,:,end+1) = dudxtmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
    DVDX_rec(:,:,end+1) = dvdxtmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
    DUDY_rec(:,:,end+1) = dudytmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
    DVDY_rec(:,:,end+1) = dvdytmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
    if(r_flag) R_rec(:,:,end+1) = rtmp(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax); end
    O_rec(:,:,end+1) = omega(plot_xmin:plot_xgap:plot_xmax,plot_ymin:plot_ygap:plot_ymax);
end

if(video_steps > 0 && mod(steps, video_steps) == 0)
    figure(1)
    set(gcf,'Position',[0 0 1200 1200])
    video_making;    
    fh = getframe(gcf);
    writeVideo(vobj, fh);
end

if(exist('paraview_steps','var') && (paraview_steps > 0 && mod(steps, paraview_steps) == 0))
    outputToParaviewBinary(folder + paraview_file + num2str(steps) + ".vtk", xp, yp, ["omega","density"], omega, r)    
end