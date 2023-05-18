function [fsf_out,fs_out,fp_out]=SFG_300(fs,fp,lamd_s,lamd_p,Npxl,dx,L,kai)

% lamd_s=1558*10^-9; %unit is meter
% lamd_p=1544*10^-9;
lamd_sf=lamd_s*lamd_p/(lamd_s+lamd_p);

%diffraction index
n_s=sqrt(4.9048+0.11768/((lamd_s*10^6)^2-0.04750)-0.027169*(lamd_s*10^6)^2); %n(?)=4.9048+0.11768/(?^2-0.04750)-0.027169�?^2
n_p=sqrt(4.9048+0.11768/((lamd_p*10^6)^2-0.04750)-0.027169*(lamd_p*10^6)^2);
n_sf=sqrt(4.9048+0.11768/((lamd_sf*10^6)^2-0.04750)-0.027169*(lamd_sf*10^6)^2);

%wave number
k_s=2*pi*n_s/lamd_s;
k_p=2*pi*n_p/lamd_p;
k_sf=k_s+k_p;


%parameter
% kai=1.437650917638231e-11;
c=3*10^8;
omega_s=2*pi*c/lamd_s; %2 pi c/lambda
omega_p=2*pi*c/lamd_p;
eta_s=omega_s/(n_s*c);
eta_p=omega_p/(n_p*c);
eta_sf=(omega_s+omega_p)/(n_sf*c);



fsf=zeros(Npxl,Npxl);      % Nx and Ny is the length of x,y 

% define k space
 kx=((2*pi)/(dx*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];
 ky=((2*pi)/(dx*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];

 %kx=((2*pi*dx)/(lamd_s*f)).*[-(Npxl)/2:1:(Npxl-2)/2];
 %ky=kx;
 %((2*pi*dx)/(lamd_s*f)).*[-(Npxl)/2:1:(Npxl-2)/2];
 
 
% kx=((2*pi)/sqrt(lamd_s*f*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];
% ky=((2*pi)/sqrt(lamd_s*f*Npxl)).*[-(Npxl)/2:1:(Npxl-2)/2];
dz=10^-2*L; 
   
z=0;
while z<L
    %PropA=fftshift(exp(1i*mu*kx.^2*dz)'*exp(1i*mu*ky.^2*dz)); 
    %A=fftshift(exp(2i*pi^2/k_s.*(Sx.^2+Sy.^2)*dz));
    %B=fftshift(exp(2i*pi^2/k_p.*(Sx.^2+Sy.^2)*dz));
    %C=fftshift(exp(2i*pi^2/k_sf.*(Sx.^2+Sy.^2)*dz));
    A=fftshift(transpose(exp(-1i*kx.^2/(2*k_s)*dz))*exp(-1i*ky.^2/(2*k_s)*dz));%signal
    B=fftshift(transpose(exp(-1i*kx.^2/(2*k_p)*dz))*exp(-1i*ky.^2/(2*k_p)*dz));%pump
    C=fftshift(transpose(exp(-1i*kx.^2/(2*k_sf)*dz))*exp(-1i*ky.^2/(2*k_sf)*dz));%SF
    % half d
    %A_h=fftshift(exp(2i*pi^2/k_s.*(Sx.^2+Sy.^2)*dz/2));
    %B_h=fftshift(exp(2i*pi^2/k_p.*(Sx.^2+Sy.^2)*dz/2));
    %C_h=fftshift(exp(2i*pi^2/k_sf.*(Sx.^2+Sy.^2)*dz/2));
    
    A_h=fftshift(transpose(exp(-1i*kx.^2/(2*k_s)*dz*0.5))*exp(-1i*ky.^2/(2*k_s)*dz*0.5));%signal
    B_h=fftshift(transpose(exp(-1i*kx.^2/(2*k_p)*dz*0.5))*exp(-1i*ky.^2/(2*k_p)*dz*0.5));%pump
    C_h=fftshift(transpose(exp(-1i*kx.^2/(2*k_sf)*dz*0.5))*exp(-1i*ky.^2/(2*k_sf)*dz*0.5));%SF
    % linear part
    fs1=ifft2(A.*fft2(fs));
    fp1=ifft2(B.*fft2(fp));
    fsf1=ifft2(C.*fft2(fsf));
    
    % NL part
    fs1=fs1+1i*(kai*eta_s).*fsf1.*conj(fp1)*dz;
    fp1=fp1+1i*(kai*eta_p).*fsf1.*conj(fs1)*dz;
    fsf1=fsf1+1i*(kai*eta_sf).*(fp1).*fs1*dz;
    
   
    %adaptive= step first 0.5*dz
     % linear part
    fs2=ifft2(A_h.*fft2(fs));
    fp2=ifft2(B_h.*fft2(fp));
    fsf2=ifft2(C_h.*fft2(fsf));
    % NL part
    fs2=fs2+1i*(kai*eta_s).*fsf2.*conj(fp2)*dz*0.5;
    fp2=fp2+1i*(kai*eta_p).*fsf2.*conj(fs2)*dz*0.5;
    fsf2=fsf2+1i*(kai*eta_sf).*(fp2).*fs2*dz*0.5;
   
   
    %adaptive step second 0.5*dz
     % linear part
    fs3=ifft2(A_h.*fft2(fs2));
    fp3=ifft2(B_h.*fft2(fp2));
    fsf3=ifft2(C_h.*fft2(fsf2));
    
    % NL part
    fs3=fs3+1i*(kai*eta_s).*fsf3.*conj(fp3)*dz*0.5;
    fp3=fp3+1i*(kai*eta_p).*fsf3.*conj(fs3)*dz*0.5;
    fsf3=fsf3+1i*(kai*eta_sf).*(fp3).*fs3*dz*0.5;
    
   
    
 %compare the first step dz with twice dz/2
    while abs(sum(sum(abs(fs3).^2-abs(fs1).^2+abs(fp3).^2-abs(fp1).^2+abs(fsf3).^2-abs(fsf1).^2)))>0.00001*sum(sum(abs(fs3).^2+abs(fp3).^2+abs(fsf3).^2))
      
      fs1=fs2;
      fp1=fp2; 
      fsf1=fsf2;
      dz=dz/2;    
    % half d
    %A_h=fftshift(exp(2i*pi^2/k_s.*(Sx.^2+Sy.^2)*dz/2));
    %B_h=fftshift(exp(2i*pi^2/k_p.*(Sx.^2+Sy.^2)*dz/2));
    %C_h=fftshift(exp(2i*pi^2/k_sf.*(Sx.^2+Sy.^2)*dz/2));
   
    A_h=fftshift(transpose(exp(-1i*kx.^2/(2*k_s)*dz*0.5))*exp(-1i*ky.^2/(2*k_s)*dz*0.5));%signal
    B_h=fftshift(transpose(exp(-1i*kx.^2/(2*k_p)*dz*0.5))*exp(-1i*ky.^2/(2*k_p)*dz*0.5));%pump
    C_h=fftshift(transpose(exp(-1i*kx.^2/(2*k_sf)*dz*0.5))*exp(-1i*ky.^2/(2*k_sf)*dz*0.5));%SF  
      
 %adaptive step first 0.5*dz 
    %linear part
    fs2=ifft2(A_h.*fft2(fs));
    fp2=ifft2(B_h.*fft2(fp));
    fsf2=ifft2(C_h.*fft2(fsf));
    
    % NL part
    fs2=fs2+1i*(kai*eta_s).*fsf2.*conj(fp2)*dz*0.5;
    fp2=fp2+1i*(kai*eta_p).*fsf2.*conj(fs2)*dz*0.5;
    fsf2=fsf2+1i*(kai*eta_sf).*(fp2).*fs2*dz*0.5;
 
      
 %adaptive step second 0.5*dz
     % linear part
    fs3=ifft2(A_h.*fft2(fs2));
    fp3=ifft2(B_h.*fft2(fp2));
    fsf3=ifft2(C_h.*fft2(fsf2));
    
   % NL part
    fs3=fs3+1i*(kai*eta_s).*fsf3.*conj(fp3)*dz*0.5;
    fp3=fp3+1i*(kai*eta_p).*fsf3.*conj(fs3)*dz*0.5;
    fsf3=fsf3+1i*(kai*eta_sf).*fp3.*fs3*dz*0.5; 
  
   
    end
 z=z+dz;
 fs=fs3;
 fp=fp3;
 fsf=fsf3; 
 dz=dz*1.5; % let dz go back to the initial setting, to do a new adapt step
end
fs_out=fs;
fp_out=fp;
fsf_out=fsf;
