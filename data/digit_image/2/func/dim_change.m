function y= dim_change(signal,Npxl,Npxl_s)
Npxl_ss=floor((Npxl-Npxl_s)/2);

signal_s=zeros(Npxl_s,Npxl_s);

for i=1:Npxl_s
    for j=1:Npxl_s
    signal_s(i,j)=signal(Npxl_ss+i,Npxl_ss+j);
   
    end
end

y=signal_s;