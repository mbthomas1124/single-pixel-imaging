  function y = L(params, x) 
        fact = @(x)arrayfun(@factorial, x);         
n = params(1);  % p     
k = abs(params(2));  % m l   
m = 0:n;        
a = factorial(n+k)*ones(1,length(m));     
b = fact(n-m);     
c = fact(k+m);     
d = fact(m);     
e = (-1).^m;         
y = zeros(size(x)); 
for s = 1:n+1          
    y  =  y + a(s) ./ b(s) ./ c(s) ./ d(s) .* e(s) .* x.^m(s);     
end 