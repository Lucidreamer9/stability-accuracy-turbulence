 %% Solver_wrapup
% for dry run or collecting observations, take the last step as output
if(DA_flag < 2 && ~history_flag)
    U = u;
    V = v;
    R = r;
end

% close the video file
if(video_steps)
    close(vobj);
end

if(DA_flag == 1)
    save(folder + DA_obs_file, 'U_obs', 'V_obs', 'R_obs');
    Omega_obs = circdiff(U_obs,[0,-1])/dy - circdiff(V_obs,[-1,0])/dx;
    fp = fopen(string(folder) + string(DA_obs_file) + ".dat",'wb');
    fwrite(fp, Omega_obs, 'real*4');
    fclose(fp);
    if(obs_p)
        save(folder + DA_obs_p_file, 'p_obs', 'dpdx_obs', 'dpdy_obs');
    end
end

% record information about the simulation in the record_file
if(record_steps)
    x_stencil = xu(plot_xmin:plot_xgap:plot_xmax);
    y_stencil = yv(plot_ymin:plot_ygap:plot_ymax);
    save(folder + record_file + "rec_velo",'U_rec','V_rec');
    save(folder + record_file + "rec_omega",'O_rec');
    if(r_flag) save(folder + record_file + "rec_density",'R_rec'); end
    save(folder + record_file + "rec_grad",'DUDX_rec','DVDX_rec','DUDY_rec','DVDY_rec');
    save(folder + record_file + "rec_grid",'x_stencil','y_stencil');
end