function cross_validation_regression(d,R,f,data_set,seed)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C)2019 All rights reserved.
%
%   Author        : Yasin Elmaci
%   Email         : yasin__elmaci@hotmail.com
%   File Name     : cross_validation_regression.m
%   
%   cross_validation_regression(d,R,f,data_set,seed)
%   utilises a gaussian-gaussian complete training data set with
%   cross-validation via split test-train for polynomial regression models
%   of order 0 to d and select the optimum model order D via cross-validation.
%   d is the max polynomial order to test i.e. d=4, 0 to d
%   R is the number of repeitions within the cross-validation algorithm
%   f is the fraction of the data to be used as test-train, i.e. 0.5 is
%   50/50
%   data_set takes in X and Y training data from a file, 
%   i.e. 'data.extension'
%   seed is the seed for the random number generator, i.e. 12345
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(seed) 
data=load(data_set); 
x=data(:,1);
y=data(:,2); 
N=length(data); 

prediction_range=(min(x):max(x)); %predicting every value within dataset
Y_hat=0; 

M=zeros(d+1,1); 
X_tst=cell(d,1);
Y_tst=cell(d,1);
X_a=cell(d,1);
W_hat=cell(d,1);
Y_p=cell(d,1);

%cross-validation algorithm implementation
for D = 0:d
    [~,~,Xtst,Ytst,Xa,W,Yp,~,m] = polynomial_cross_validation(x,y,N,D,R,f,seed);
    M(D+1,1)=m;
    X_tst{D+1}=Xtst;
    Y_tst{D+1}=Ytst;
    X_a{D+1}=Xa;
    W_hat{D+1}=W;
    Y_p{D+1}=Yp;
end

D_fit=find(M==min(M))-1;

Xa = zeros(N,D_fit+1);

%matrix augmentation
for i = 0:D_fit 
    Xa(:,i+1) = x.^i;
end

W=(Xa'*Xa)\(Xa'*y); %parameter estimation

y_hat = Xa*W; %estimating regression for training data set

%predicting regression across range for fitted model
for j=0:D_fit 
    for k=j+1 
        Y_hat=(prediction_range.^j)*W(k)+Y_hat;
    end
end

%polynomial order tested against the mean NLL
d_order=num2str((0:D)');
t1=table(d_order,M,'VariableNames',{'D','NLL'});
disp(t1);

%W parameter values where W=[W_1,W_2,..W_n]
W_n=num2str((0:length(W)-1)');
t2=table(W_n,W);
disp(t2);

%root mean square values for fitted regression model
rmse=(sqrt(sum((y-y_hat).^2))/N);
t3=table(rmse,'VariableNames',{'RMSE'});
disp(t3);

end