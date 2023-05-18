function y=Fourier_(beam,Npxl)

N2=(0:1:Npxl-1);                                          %vector 0 to N-1
Ny=repmat(transpose(N2),1,Npxl);                          %matrix of above vector repeated

FFT_s00=(1/Npxl).*(fft(beam.*exp((-1i*pi.*Ny))).*exp(1i*pi*Ny)).';    
y=(fft(FFT_s00.*exp((-1i*pi.*Ny))).*exp(1i*pi*Ny)).';  
