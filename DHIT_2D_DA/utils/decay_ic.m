function psi = decay_ic(nx, ny, dx)
    epsilon = 1.0e-6;
    
    kx = zeros(nx, 1);
    ky = zeros(ny, 1);
    
    kx(1:floor(nx/2)) = 2*pi/(nx*dx)*((0:floor(nx/2)-1)');
    kx(floor(nx/2)+1:nx) = 2*pi/(nx*dx)*((-floor(nx/2):-1)');

    ky(1:ny) = kx(1:ny);
    
    kx(1) = epsilon;
    ky(1) = epsilon;

    [kx, ky] = meshgrid(kx, ky);
    
    ksi = 2.0*pi*rand(floor(nx/2+1), floor(ny/2+1));
    eta = 2.0*pi*rand(floor(nx/2+1), floor(ny/2+1));
    
    phase = zeros(nx, ny);
    
    for ix = 2:floor(nx/2)
        for iy = 2:floor(ny/2)
            phase(ix,iy)           = complex(cos( ksi(ix,iy) + eta(ix,iy)), sin( ksi(ix,iy) + eta(ix,iy)));
            phase(nx-ix+1,iy)      = complex(cos(-ksi(ix,iy) + eta(ix,iy)), sin(-ksi(ix,iy) + eta(ix,iy)));
            phase(ix,ny-iy+1)      = complex(cos( ksi(ix,iy) - eta(ix,iy)), sin( ksi(ix,iy) - eta(ix,iy)));
            phase(nx-ix+1,ny-iy+1) = complex(cos(-ksi(ix,iy) - eta(ix,iy)), sin(-ksi(ix,iy) - eta(ix,iy)));
        end
    end

    k0 = 10.0;
    c = 4.0/(3.0*sqrt(pi)*(k0^5));
    
    kk = sqrt(kx.^2 + ky.^2);
    es = c*(kk.^4).*exp(-(kk/k0).^2);
    wf = sqrt((kk.*es/pi)) .* phase * (nx*ny);
    
    ut = ifft2(wf./(kk.^2), 'symmetric');
    
    psi(1:nx, 1:ny) = ut;
end
