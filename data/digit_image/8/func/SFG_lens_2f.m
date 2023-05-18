function y=SFG_lens_2f(beam_s,f,L,lamda,dx,Npxl,Npxl_s)

% x=dx*[-(Npxl)/2:1:(Npxl-2)/2];                       % x vector
% y=dx*[-(Npxl)/2:1:(Npxl-2)/2];                       % y vector
% X=repmat(x,Npxl,1);                                  %matrix in x
% Y=repmat(y',1,Npxl);                                 %matrix in y
% n_s=sqrt(4.9048+0.11768/((lamda*10^6)^2-0.04750)-0.027169*(lamda*10^6)^2); %n(?)=4.9048+0.11768/(?^2-0.04750)-0.027169�?^2
% 
% ks=2*pi*n_s/lamda;%*(2.21094983010626)
% % % k space kx/ky vector
% kx=((2*pi)/(dx*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];
% ky=((2*pi)/(dx*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];

beam=zeros(Npxl,Npxl);
Npxl_ss=(Npxl-Npxl_s)/2;

for i=1:Npxl_s
    for j=1:Npxl_s
    beam(Npxl_ss+i,Npxl_ss+j)=beam_s(i,j);
    end
end

beam_f=prop_z(beam,f-L/2,lamda,dx,Npxl);
y=lens_prop_dis(beam_f,f,f,lamda,dx,Npxl);



