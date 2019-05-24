function polynomial_regression(D,data_set,seed) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C)2019 All rights reserved.
%
%   Author        : Yasin Elmaci
%   Email         : yasin__elmaci@hotmail.com
%   File Name     : polynomial_regression.m
%   
%   regression(D,data_set,seed) utilises a gaussian-gaussian
%   complete training data set to train a polynomial regression of order D.
%   D is the polynomial order i.e. 4
%   data_set takes in X and Y training data from a file,
%   i.e. 'data.extension'
%   seed is the seed for the random number generator, i.e. 12345
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(seed) 
data=load(data_set); 
x=data(:,1);
y=data(:,2); 
N=length(data); 
Xa = zeros(N,D+1);
prediction_range=(min(x):max(x)); %predicting every value within dataset
Y_hat=0; 

%matrix augmentation
for i = 0:D 
    Xa(:,i+1) = x.^i;
end

W=(Xa'*Xa)\(Xa'*y); %parameter estimation

y_hat=Xa*W; %estimating regression for training data set

%predicting regression across range for fitted model
for j=0:D 
    for k=j+1 
        Y_hat=(prediction_range.^j)*W(k)+Y_hat;
    end
end

%W parameter values where W=[W_1,W_2,..W_n
W_n=num2str((0:length(W)-1)');
t1=table(W_n,W);
disp(t1);

%root mean square values for fitted regression model
rmse=(sqrt(sum((y-y_hat).^2))/N);
t2=table(rmse,'VariableNames',{'RMSE'});
disp(t2);


end
