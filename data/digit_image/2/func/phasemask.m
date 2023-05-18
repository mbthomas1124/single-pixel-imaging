function y=phasemask(p,l,w0_s,rho,phi)
params=[p l w0_s];
mode_1= LG(params, rho, phi); 
phase_1=angle(mode_1);
y=mod(phase_1,2*pi);

