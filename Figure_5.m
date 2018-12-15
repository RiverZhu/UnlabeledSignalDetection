clc;
clear;
close all;
%%
MC = 5000;
options = optimset('Largescale','off','GradObj','on','Hessian','off',...
            'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
N = 20; % sensors 
theta = 1; 
Delta = 2;  
% sine signal
% t=rand(N,1);
% t=sort(t,'ascend');
% h = sin(2*pi*t);
% tau = 4*rand(N,1)-2;
h   = [  0.8441   0.9457   0.9511   0.9583   0.9813   0.8434   0.7035   0.6851   0.4941   0.1695   0.1333  -0.0040  -0.1966  -0.4215  -0.7669  -0.9159  -0.9318  -0.9373  -0.4842  -0.3699]';
tau = [ -0.2662   0.5539   1.0422   0.9742   0.9356  -0.0313   1.1529   1.6190   0.5164   1.5550   0.9644   1.8434   1.4548   1.2462   0.6496  -1.7689  -1.8761  -0.2137   1.9555  -1.4672]';
   
q_0 = 0.05;
q_1 = 0.05;
wavr = 1;
epsilon = 1e-7;
% generate a random permutation matrix 'Perm'
A = eye(N); 
idx = randperm(N);
Perm = A(idx, :);  
% K_sam = 3000:1000:5000;
K_sam = [20 40 60 80 100 150 200 300 400 500 1000 2000 4000 6000 8000 10000 3*1e4 1e5 3*1e5];


MSE_lab_sam = zeros(length(K_sam),MC);
MSE_unl_random_sam = zeros(length(K_sam),MC);
MSE_unl_calculated_sam = zeros(length(K_sam),MC);

for K_index = 1:length(K_sam) 
    K = K_sam(K_index);
    for mc = 1:MC
        % generate N*K observations
        y = h*ones(1,K)*theta+sqrt(wavr)*randn(N,K)-tau*ones(1,K);
        % quantization and inversion
        eta = sum((sign(y)+1)/2,2)/K;
        for i=1:N
           eta(i) = (sum(rand(fix(K*eta(i)),1)>=q_1) + sum(rand(fix(K-K*eta(i)),1)<=q_0))/K;
        end  
        
        %%  labeled case
                theta_lab = fminunc(@(theta)Fun_Q_T(eta,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        if theta_lab > Delta
            theta_lab = Delta;
        end
        if theta_lab < -Delta
            theta_lab = -Delta;
        end
        
       %% unlabeled eta projection to I
        eta_perm = Perm' * eta;
        lbound = q_0+(1-q_0-q_1)*normcdf(min(-abs(h)*Delta-tau));% project eta to I(eta)
        ubound = q_0+(1-q_0-q_1)*normcdf(max(+abs(h)*Delta-tau));
        I = eta_perm;
        for i = 1:N
                if I(i) <= lbound
                    I(i) = lbound+0.001;
                end
                if I(i) >= ubound
                    I(i) = ubound-0.001;
                end
        end
       %% unlabeled case with a random start point theta = Delta  
        theta_new = Delta;
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [h_asc,h_asc_index]=sort(h*theta_old-tau); 
            [eta_perm_asc,eta_perm_asc_index]=sort(I);  % sort eta_perm
            Perm_h_asc = A(h_asc_index, :);
            Perm_eta_perm_asc = A(eta_perm_asc_index, :);
            Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
            theta_new = fminunc(@(theta)Fun_Q_T((Pi_asc*I),h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_Delta = theta_new;
        if theta_Delta > Delta
            theta_Delta = Delta;
        end
        if theta_Delta < -Delta
            theta_Delta = -Delta;
        end
        l_Delta = (Pi_asc*I)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_Delta-tau)/sqrt(wavr)))+(1-(Pi_asc*I)')*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_Delta-tau)/sqrt(wavr))));

       %% unlabeled case with a random start point theta = -Delta
        theta_new = -Delta;
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [h_asc,h_asc_index]=sort(h*theta_old-tau); 
            [eta_perm_asc,eta_perm_asc_index]=sort(I);  % sort eta_perm
            Perm_h_asc = A(h_asc_index, :);
            Perm_eta_perm_asc = A(eta_perm_asc_index, :);
            Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
            theta_new = fminunc(@(theta)Fun_Q_T((Pi_asc*I),h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_Delta_neg = theta_new;
        if theta_Delta_neg > Delta
            theta_Delta_neg = Delta;
        end
        if theta_Delta_neg < -Delta
            theta_Delta_neg = -Delta;
        end
        l_Delta_neg = (Pi_asc*I)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_Delta_neg-tau)/sqrt(wavr)))+(1-(Pi_asc*I)')*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_Delta_neg-tau)/sqrt(wavr))));
       %% Decide random initial point
       if l_Delta > l_Delta_neg
           theta_random = theta_Delta;
       else
           theta_random = theta_Delta_neg;
       end
       %% unlabeled case with a calculated start point
        m = sqrt(wavr)* norminv((I-q_0)/(1-q_0-q_1));
        if (m'*m-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2>=0
            theta1 = (tau'* h)/(h'*h) + sqrt((m'*m-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
            theta2 = (tau'* h)/(h'*h) - sqrt((m'*m-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
        else
            theta1 = Delta;
            theta2 = -Delta;
        end
        if theta1>Delta
            theta1 = Delta;
        end
        if theta1<-Delta
            theta1 = -Delta;
        end
        if theta2>Delta
            theta2 = Delta;
        end
        if theta2<-Delta
            theta2 = -Delta;
        end
        % iterate with theta 1       
        theta_new = theta1;
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [h_asc,h_asc_index]=sort(h*theta_old-tau); 
            [eta_perm_asc,eta_perm_asc_index]=sort(I);  % sort eta_perm
            Perm_h_asc = A(h_asc_index, :);
            Perm_eta_perm_asc = A(eta_perm_asc_index, :);
            Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
            theta_new = fminunc(@(theta)Fun_Q_T((Pi_asc*I),h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),theta_old,options);
        end
        theta_1_final = theta_old;
        Pi_1 = Pi_asc;
        % iterate with theta 2   
        theta_new = theta2;
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [h_asc,h_asc_index]=sort(h*theta_old-tau); 
            [eta_perm_asc,eta_perm_asc_index]=sort(I);  % sort eta_perm
            Perm_h_asc = A(h_asc_index, :);
            Perm_eta_perm_asc = A(eta_perm_asc_index, :);
            Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
            theta_new = fminunc(@(theta)Fun_Q_T((Pi_asc*I),h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),theta_old,options);
        end
        theta_2_final = theta_old;
        Pi_2 = Pi_asc;
        % Decide which theta is to be used         
        l_1=(Pi_1*I)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_1_final-tau)/sqrt(wavr)))+(1-(Pi_1*I)')*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_1_final-tau)/sqrt(wavr))));
        l_2=(Pi_2*I)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_2_final-tau)/sqrt(wavr)))+(1-(Pi_2*I)')*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_2_final-tau)/sqrt(wavr))));
        if l_1 > l_2
            theta_calculated = theta_1_final;
        else
            theta_calculated = theta_2_final;
        end
        if theta_calculated > Delta
            theta_calculated = Delta;
        end
        if theta_calculated < -Delta
            theta_calculated = -Delta;
        end
%         theta1
%         theta_calculated
%         theta_random
       %% record square error
        MSE_lab_sam(K_index,mc) = (theta_lab-theta)^2;
        MSE_unl_random_sam(K_index,mc) = (theta_random-theta)^2;
        MSE_unl_calculated_sam(K_index,mc) = (theta_calculated-theta)^2;
    end
end

MSE_lab = mean(MSE_lab_sam,2);
MSE_unl_random = mean(MSE_unl_random_sam,2);
MSE_unl_calculated = mean(MSE_unl_calculated_sam,2);

FI = zeros(1,length(K_sam));
for K_index = 1:length(K_sam)
    K = K_sam(K_index);
    FI(K_index) = 1/(K*(1-q_0-q_1)^2/wavr*h'*(h.*normpdf((h*theta-tau)/sqrt(wavr)).*normpdf((h*theta-tau)/sqrt(wavr))./(1-q_0-(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))./(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))));
end

%% figure
% alw = 0.75;    % AxesLineWidth
% fsz = 10;      % Fontsize
% lw = 1.5;      % LineWidth
% msz = 8;       % MarkerSize
% set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% figure(1);
% plot(K_sam,MSE_lab,'-.ro',K_sam,MSE_unl_random,'--ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,':kx','LineWidth',lw,'MarkerSize',msz);
% % semilogy(K_sam,MSE_lab,'-ro',K_sam,MSE_unl_random,'-ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,'-kx','LineWidth',lw,'MarkerSize',msz)
% xlabel('$K$','interpreter','latex','Fontsize',fsz)
% ylabel('MSE','Fontsize',fsz)
% set(gca,'XLim',[20 200]);
% axes('position',[0.405,0.425,0.47,0.47]);
% set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% semilogy(K_sam,MSE_lab,'-.ro',K_sam,MSE_unl_random,'--ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,':kx','LineWidth',lw,'MarkerSize',msz);
% legend('Labeled data ','Unlabeled data','Unlabeled data, good initial points','CRLB');


% load('random_h_random_tau_MSE_data_less10000.mat')
% MSE_lab_plus = MSE_lab;
% MSE_unl_random_plus = MSE_unl_random;
% MSE_unl_calculated_plus = MSE_unl_calculated;
% FI_plus =  FI;
% K_sam_plus = K_sam;
% load('random_h_random_tau_MSE_data_more10000.mat')
% MSE_lab = [ MSE_lab_plus' MSE_lab'];
% MSE_unl_random = [ MSE_unl_random_plus' MSE_unl_random' ];
% MSE_unl_calculated = [  MSE_unl_calculated_plus' MSE_unl_calculated' ];
% FI = [ FI_plus FI  ];
% K_sam = [ K_sam_plus K_sam ];


alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
figure(1);
loglog(K_sam,MSE_lab,'-.ro',K_sam,MSE_unl_random,'--ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,':kx','LineWidth',lw,'MarkerSize',msz);
% semilogy(K_sam,MSE_lab,'-.ro',K_sam,MSE_unl_random,'--ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,':kx','LineWidth',lw,'MarkerSize',msz);
set(gca,'XLim',[20 1e5]);
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('MSE','Fontsize',fsz)
hold on;
axes('position',[0.555,0.575,0.32,0.32]);
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
plot(K_sam,MSE_lab,'-.ro',K_sam,MSE_unl_random,'--ms',K_sam,MSE_unl_calculated,'-b*',K_sam,FI,':kx','LineWidth',lw,'MarkerSize',msz);
set(gca,'XLim',[20 80]);
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('MSE','Fontsize',fsz)
legend('Labeled data ','Unlabeled data','Unlabeled data, good initial points','CRLB');



% save sine_h_0806;

