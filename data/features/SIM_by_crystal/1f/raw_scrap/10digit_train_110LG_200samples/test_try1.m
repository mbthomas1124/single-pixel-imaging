


path='/Users/hezhang/Documents/MATLAB/work_progress/2019/April_May_2019/Digits_recog/10digit_train/test_10digit_May17/';
% path='/Users/hezhang/Documents/MATLAB/work_progress/April_May/Digits_recog/exp_digit_recog_training/';
digit_total=10;% how many digits will be trained?
test_sample=400;% traning samples for each digit

accur_count=zeros(1,test_sample);
ans_digi=zeros(digit_total,test_sample);
accur=zeros(digit_total,test_sample);
digit_vec=eye(digit_total,digit_total);
% digit=1;
for j= 1:test_sample
    name=[path,'LG_digit_unknown_',num2str(j),'.mat'];
%     name=[path,'digit2_',num2str(j),'.mat'];
    load(name) 
%     test_cp=transpose(conv_cp)./max(conv_cp);
    ans_digi(:,j)=round(net(conv_cp));
    accur(:,j)=digit_vec(:,digit+1)-ans_digi(:,j);
    if sum(accur(digit+1,j)) == 0 
        accur_count(1,j)=1;
%     else accur_count(1,i)=0;
    end
end
    
eff=sum(accur_count)/test_sample
    
    