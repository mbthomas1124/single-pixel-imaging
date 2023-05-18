

function y = LG( params, rho, phi ) 
p =params(1); % p
ml =(params(2)); %l
w = params(3); %beam waist
if w==0      
    msgbox('params(0) can not be equal to 0');
end
t = rho./w;

%y = sqrt(2*factorial(p)/pi/factorial(m+p))/w.* (sqrt(2).*t).^m.* laguerreL(p,m,2*t.^2).* exp(-t.^2 + 1i*m*phi);
y = sqrt(2*factorial(p)/pi/factorial(abs(ml)+p))/w.* (sqrt(2).*t).^abs(ml).* L([p ml], 2*t.^2).* exp(-t.^2 -1i*ml*phi);


%y = sqrt(2*factorial(p)/pi/factorial(m+p))/w.* (sqrt(2).*t).^m.* L([p m], 2*t.^2).* exp(-t.^2 + 1i*m*phi).*exp(-1i*(2*p+m+1));
%y = sqrt(2*factorial(p)/pi/factorial(m+p))/w.* (sqrt(2).*t).^m.* L([p m], 2*t.^2).* exp(-t.^2);   
  
    


