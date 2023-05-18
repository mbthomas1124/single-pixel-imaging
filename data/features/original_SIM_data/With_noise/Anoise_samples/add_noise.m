clear
close all


%noise mask
S=10;
edge=900/S;
percent_mod=0.2;
noise_perc=round(percent_mod*edge*edge);% percentage for modulation
noise_none=edge*edge-noise_perc;% percentage for non- modulation

rand_noise=[100*ones(1,noise_perc),zeros(1,noise_none)];
rand_noise=reshape(rand_noise(randperm(edge*edge)),[edge,edge]);
noise_mask=repelem(rand_noise,S,S);

sum(noise_mask(:));

figure;imagesc(noise_mask)


%% digit images
filename=['image50.png'];
A0= imread(filename);
A1=imresize(A0,[900 900]);
% A1=rgb2gray(A1);
[rows_image, columns_image, numberOfColorChannels1] = size(A1);
A_pre=double(A1);


figure;imagesc(A_pre)

%% noise + digit

A_noise=A_pre+noise_mask;

digi_mask=padarray(A_noise./max(max(A_noise)).*2*pi,[150 150],0,'both');


figure;imagesc(digi_mask)