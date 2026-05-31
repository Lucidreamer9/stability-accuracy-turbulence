function [u_guess, v_guess] = divergence_free_projection(u_guess, v_guess, Setups)
% A standalone Divergence-free projection operator
    dx = Setups.Parameters.dx;
    dy = Setups.Parameters.dy;
    Modified_k2 = Setups.Parameters.Modified_k2;
    div = circdiff(u_guess, [1,0])/dx + circdiff(v_guess, [0,1])/dy;
        % solve Poisson equation using modified wave numbers for
    pdagger = real(ifft2(fft2(div)./Modified_k2));

    dpdagger_dx = circdiff(pdagger, [-1, 0])/dx;
    dpdagger_dy = circdiff(pdagger, [0, -1])/dy;
    
    u_guess = u_guess - dpdagger_dx;
    v_guess = v_guess - dpdagger_dy;
end