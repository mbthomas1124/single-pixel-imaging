% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created 20-May-2019 13:56:35
%
% This script assumes these variables are defined:
%
%   simpleclassInputs - input data.
%   simpleclassTargets - target data.

close all
% path='/Users/hezhang/Documents/MATLAB/work_progress/April_May/Digits_recog/10digit_train/';
path1='/Users/hezhang/Documents/MATLAB/work_progress/2019/April_May_2019/Digits_recog/';
% path2='experimetal_result/';
path2='10digit_train/';

digit_total=10;% how many digits will be trained?
train_sample=200;% traning samples for each digit

input_data=zeros(110,train_sample);
target_data=zeros(digit_total,train_sample*digit_total);

for i = 1:digit_total
name1=[path1,path2,'LG_digit',num2str(i-1),'.mat'];
% name2=[path,'LG_digit1_4','.mat'];
% name3=[path,'LG_digit2_4','.mat'];
load(name1)
input_data(:,(i-1)*train_sample+1:i*train_sample)=(conv_cp);
target_data(i,(i-1)*train_sample+1:i*train_sample)=1;
%ones(1,200)*(i-1);
end
x = input_data;
t = target_data;




% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize =20;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)
save('try1','net')
% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)
