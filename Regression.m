function Regression(D,seed,data_set) 
rng(seed) 
data=load(data_set); 
x=data(:,1);
y=data(:,2); 
N=length(data); 

Xa = zeros(N,D+1);
for i = 0:D 
    Xa(:,i+1) = x.^i;
end

W=(Xa'*Xa)\Xa'*y;

y_hat = Xa*W;

prediction_range=(min(x):max(x));
Y_hat=0; 
for j=0:D 
    for k=j+1 
        Y_hat=(prediction_range.^j)*W(k)+Y_hat;
    end
end

w_n=num2str((0:length(W)-1)');
t1=table(w_n,W);
disp(t1);

rmse=(sqrt(sum((y-y_hat).^2))/N);
t2=table(rmse,'VariableNames',{'RMSE'});
disp(t2);


end
