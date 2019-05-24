function [Xtrn,Ytrn,Xtst,Ytst,Xa,W,Yp,NLL,m] = polynomial_cross_validation(x,y,N,D,R,f,seed)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C)2019 All rights reserved.
%
%   Author        : Yasin Elmaci
%   Email         : yasin__elmaci@hotmail.com
%   File Name     : polynomial_cross_validation.m
%   
%   polynomial_cross_validation(x,y,N,D,R,f,seed)
%   takes in gaussian-gaussian distributed X,Y data and implements
%   cross-validation for polynomial order D over R iterations and a split
%   test-train fraction f
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(seed)
NLL = zeros(R,1);
Xa = zeros(N,D+1);
for i = 0:D 
    Xa(:,i+1) = x.^i;
end

%splitting data into test-train subsets and calculating model NLL every R
for r = 1:R 
    Nf = round(f*N);
    trnData = randsample(N,Nf); 
    tstData = setdiff(1:N,trnData);
    Xtrn = Xa(trnData,:);
    Ytrn = y(trnData,:);
    W = (Xtrn'*Xtrn)\(Xtrn'*Ytrn); 
    Xtst = Xa(tstData,:); 
    Ytst = y(tstData,:); 
    Yp = Xtst*W;
    NLL(r) = ((Ytst-Yp)'*(Ytst-Yp))/(N-Nf); 
end
    m = mean(NLL);
end