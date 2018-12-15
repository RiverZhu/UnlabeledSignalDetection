%%
MC = 5000;
options = optimset('Largescale','off','GradObj','on','Hessian','off',...
            'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
N = 20; % sensors 
theta = 1; 
Delta = 2;  
h = linspace(-Delta,Delta,N)';
h = h+0.5;
tau = 0.5*h;
q_0 = 0.05;
q_1 = 0.05;
wavr = 1;

% generate a random permutation matrix 'Perm'
A = eye(N); 
idx = randperm(N);
Perm = A(idx, :);  

K_sam = [20 40 60 80 100 120 140 160 180 200];
% K_sam = [20 40 60 80 100 150 200 300 400 500 1000 2000 4000 6000 8000 10000];

MSE_lab_sam = zeros(length(K_sam),MC);
MSE_unl_sam = zeros(length(K_sam),MC);
MSE_ex_sam  = zeros(length(K_sam),MC);

FI = zeros(1,length(K_sam));
for K_index = 1:length(K_sam)
    K = K_sam(K_index);
    FI(K_index) = 1/(K*(1-q_0-q_1)^2/wavr*h'*(h.*normpdf((h*theta-tau)/sqrt(wavr)).*normpdf((h*theta-tau)/sqrt(wavr))./(1-q_0-(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))./(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))));
end

for K_index = 1:length(K_sam) % time
    K = K_sam(K_index);
    for mc = 1:MC
       %% 
        % generate N*K observations
        y = h*ones(1,K)*theta+sqrt(wavr)*randn(N,K)-tau*ones(1,K);
        % quantization and inversion
        eta = sum((sign(y)+1)/2,2)/K;
        for i=1:N
           eta(i) = (sum(rand(fix(K*eta(i)),1)>=q_1) + sum(rand(fix(K-K*eta(i)),1)<=q_0))/K;
        end          
        %% labeled case
        theta_lab = fminunc(@(theta)Fun_Q_T(eta,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);       
       %% unlabeled case with a random start point theta = Delta   
        eta_perm = Perm'* eta;
        [h_asc,h_asc_index]=sort(h,'ascend'); 
        [h_des,h_des_index]=sort(h,'descend'); 
        [eta_perm_asc,eta_perm_asc_index]=sort(eta_perm,'ascend');  % sort eta_perm
        Perm_h_asc = A(h_asc_index, :);
        Perm_h_des = A(h_des_index, :);
        Perm_eta_perm_asc = A(eta_perm_asc_index, :);
        Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
        Pi_des = Perm_h_des'*Perm_eta_perm_asc; % recover permutation
        theta_asc = fminunc(@(theta)Fun_Q_T((Pi_asc*eta_perm),h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_des = fminunc(@(theta)Fun_Q_T((Pi_des*eta_perm),h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        if theta_des > Delta
            theta_des = Delta;
        end
        if theta_asc > Delta
            theta_asc = Delta;
        end
        if theta_des < -Delta
            theta_des = -Delta;
        end
        if theta_asc < -Delta
            theta_asc = -Delta;
        end
        l_asc = (Pi_asc*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)))+(1-(Pi_asc*eta_perm)')*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr))));
        l_des = (Pi_des*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)))+(1-(Pi_des*eta_perm)')*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr))));
        if l_asc > l_des
            theta_unl = theta_asc;
        else
            theta_unl = theta_des;
        end
        MSE_lab_sam(K_index,mc) = (theta_lab-theta)^2;
        MSE_unl_sam(K_index,mc) = (theta_unl-theta)^2;
        %% navie algorithm using two extremes of observed signal  
        eta_ex=[eta_perm_asc(1);eta_perm_asc(N);]; % sort eta in ascending order
        h_asc_ex=[h(h_asc_index(1));h(h_asc_index(N));];
        h_des_ex=[h(h_des_index(1));h(h_des_index(N));];
        tau_asc_ex=[tau(h_asc_index(1));tau(h_asc_index(N));];
        tau_des_ex=[tau(h_des_index(1));tau(h_des_index(N));];
        
        theta_asc = fminunc(@(theta)Fun_Q_T(eta_ex,h_asc_ex,theta,tau_asc_ex,2,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_des = fminunc(@(theta)Fun_Q_T(eta_ex,h_asc_ex,theta,tau_des_ex,2,K,q_0,q_1,sqrt(wavr)),0,options);
        if theta_des > Delta
            theta_des = Delta;
        end
        if theta_asc > Delta
            theta_asc = Delta;
        end
        if theta_des < -Delta
            theta_des = -Delta;
        end
        if theta_asc < -Delta
            theta_asc = -Delta;
        end
        l_asc = eta_ex'*log(q_0+(1-q_0-q_1)*normcdf((h_asc_ex*theta_asc-tau_asc_ex)/sqrt(wavr)))+(1-eta_ex')*log(1-(q_0+(1-q_0-q_1)*normcdf((h_asc_ex*theta_asc-tau_asc_ex)/sqrt(wavr))));
        l_des = eta_ex'*log(q_0+(1-q_0-q_1)*normcdf((h_des_ex*theta_asc-tau_des_ex)/sqrt(wavr)))+(1-eta_ex')*log(1-(q_0+(1-q_0-q_1)*normcdf((h_des_ex*theta_asc-tau_des_ex)/sqrt(wavr))));
        if l_asc > l_des
            theta_ex = theta_asc;
        else
            theta_ex = theta_des;
        end
        MSE_ex_sam(K_index,mc) = (theta_ex-theta)^2;
    end
end
MSE_lab = mean(MSE_lab_sam,2);
MSE_unl = mean(MSE_unl_sam,2);
MSE_ex = mean(MSE_ex_sam,2);
%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
figure(1);
% semilogy(K_sam,MSE_lab,'-ro',K_sam,MSE_unl,'--b*',K_sam,FI,'-.kx','LineWidth',lw,'MarkerSize',msz)
plot(K_sam,MSE_lab,'-ro',K_sam,MSE_unl,'--b*',K_sam,FI,'-.kx',K_sam,MSE_ex,':m<','LineWidth',lw,'MarkerSize',msz)
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('MSE','Fontsize',fsz)
set(gca,'XLim',[20 200]);
hold on;
axes('position',[0.555,0.575,0.32,0.32]);
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
semilogy(K_sam,MSE_lab,'-ro',K_sam,MSE_unl,'--b*',K_sam,FI,'-.kx',K_sam,MSE_ex,':m<','LineWidth',lw,'MarkerSize',msz)
set(gca,'XLim',[20 200]);
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('MSE','Fontsize',fsz)
legend('Labeled data','Unlabeled data','CRLB','Unlabeled data (Extrema)');
% 
% alw = 0.75;    % AxesLineWidth
% fsz = 10;      % Fontsize
% lw = 1.5;      % LineWidth
% msz = 8;       % MarkerSize
% set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% figure(1);
% semilogy(K_sam,MSE_lab,'-ro',K_sam,MSE_unl,'--b*',K_sam,FI,'-.kx','LineWidth',lw,'MarkerSize',msz);
% % semilogy(K_sam,MSE_lab,'-ro',K_sam,MSE_unl_random,'-ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,'-kx','LineWidth',lw,'MarkerSize',msz)
% xlabel('$K$','interpreter','latex','Fontsize',fsz)
% ylabel('MSE','Fontsize',fsz)
% axes('position',[0.405,0.425,0.47,0.47]);
% set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% plot(K_sam,MSE_lab,'-ro',K_sam,MSE_unl,'--b*',K_sam,FI,'-.kx','LineWidth',lw,'MarkerSize',msz);
% set(gca,'XLim',[20 200]);
% legend('Labeled data','Unlabeled data','CRLB');

% save ramp_h_0806;
% save ramp_h_20180322;


