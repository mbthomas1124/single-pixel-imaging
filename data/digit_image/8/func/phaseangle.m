function y=phaseangle(l,p,w0_s,rho,phi)
params=[l p w0_s];
lgmode= LG(params, rho, phi); 
y=angle(lgmode);

