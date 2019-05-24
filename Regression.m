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

% train_range=min(y):max(y); 

% figure; 
% plot(x,y,'k.-')
% title('Parkinson''s disease symptom progression')
% xlabel('Time (Days since diagnosis)')
% ylabel('Symptom severity (UPDRS score)')

% figure;
% plot(x,y,'k.')
% hold on
% plot(x,y_hat,'r.-')
% title1=sprintf('Parkinson''s disease symptom progression model fit \n polynomial model order D=%d',D);
% title(title1);
% xlabel('Time (Days since diagnosis)')
% ylabel('Symptom severity (UPDRS score)')
% legend({'Complete training data','Model fit'},'Location','northwest')
% 
% figure;
% plot(x,y,'k.')
% hold on
% plot(predict_days,Y_hat,'r-')
% title2=sprintf('Parkinson''s disease symptom progression prediction \n polynomial model order D=%d',D);
% title(title2);
% xlabel('Time (Days since diagnosis)')
% ylabel('Symptom severity prediction (UPDRS score)')
% legend({'Complete training data','Predicted progession'},'Location','northwest')
% 
% figure
% plot(train_range,train_range,'r--')
% hold on
% plot(y,y_hat,'k.')
% title('Training UPDRS score against predicted UPDRS score')
% xlabel('y')
% lable = {'$$ \hat{y} $$'}; 
% ylabel(lable, 'Interpreter','latex')
% leg = legend({'$ y=\hat{y} $';'$(y,\hat{y})$'});
% set(leg,'Interpreter','latex','Location','northwest');