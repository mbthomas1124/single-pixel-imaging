function y=lens_prop_dis(beam,f,z,lamda,dx,Npxl)
% x space
x=dx*[-(Npxl)/2:1:(Npxl-2)/2];                       % x vector
y=dx*[-(Npxl)/2:1:(Npxl-2)/2];                       % y vector
X=repmat(x,Npxl,1);                                  %matrix in x
Y=repmat(y',1,Npxl);                                 %matrix in y

ks=2*pi/lamda;
% % k space kx/ky vector
kx=((2*pi)/(dx*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];
ky=((2*pi)/(dx*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];

prop_crys=fftshift(transpose(exp(-1i*kx.^2/(2*ks)*z))*exp(-1i*ky.^2/(2*ks)*z));%prop term in free space

lens_phase=-ks*((X.^2+Y.^2)/(2*f));
lens_phase=mod(lens_phase,2*pi);  

fs_lens=beam.*exp(1i.*lens_phase);

ffs2=fft2(fs_lens);
ffs_k=ffs2.*prop_crys;
y=ifft2(ffs_k);


