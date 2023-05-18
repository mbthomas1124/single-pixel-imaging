function y=pinhole(radius,dx,Npxl)
x_r=dx*[-(Npxl)/2:1:(Npxl-2)/2];                       % x vector
y_r=dx*[-(Npxl)/2:1:(Npxl-2)/2];                       % y vector


Npxl_r=floor(radius/dx); %R=FWHM*1.7/2=FWHM*0.85
r=Npxl_r*dx;
z=zeros(Npxl,Npxl);

for i=1:Npxl
    for j=1:Npxl
        dis=sqrt(x_r(1,i)^2+y_r(1,j)^2);
        if dis <= r
            z(i,j)=1;
        end
    end
end
y=z;