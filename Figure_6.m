clc;
clear;
close all;
%% initialization
MC = 5000;
theta = 1; 
Delta = 2;  
epsilon = 1e-7;
q_0 = 0.05;
q_1 = 0.05;
a = q_0/(1-q_0-q_1);
b = q_1/(1-q_0-q_1);
P_fa = 0.05; % Pfa
K_sam = 10:10:140; %Note that the values of N and K in the matlab code represent the values of  K and N in the paper, respectively.
N = 20; 
wavr = 9;
A = eye(N); % generate a random permutation matrix 'Perm'
idx = randperm(N);
Perm = A(idx, :);  
%% Subparagraph a : ramp signal
h = linspace(-2,2,N)'+0.5; % 
tau = 0.5 * h;
%%%%%%%%
T= zeros(1,MC);
T_0_a = T;
T_1_a = T;
T_0_b = T;
T_1_b = T;
T_0_c = T;
T_1_c = T;
T_0_d = T;
T_1_d = T;
T_0_e = T;
T_1_e = T;
P_d_K_NP = zeros(1,length(K_sam)); % blank matrix 
P_d_K_label = zeros(1,length(K_sam));
P_d_K_reorder = zeros(1,length(K_sam));
P_d_K_unlabel_normal = zeros(1,length(K_sam));
P_d_K_unlabel_good = zeros(1,length(K_sam));
% MC simulation
for K_index = 1:length(K_sam) % time
    K = K_sam(K_index);
%     [P_d_K_NP(K_index), P_d_K_label(K_index), P_d_K_reorder(K_index), P_d_K_unlabel(K_index)] = LRT(N,K,h,wavr,tau,theta,MC,P_fa,q_0,q_1,T,Perm);
    options = optimset('Largescale','off','GradObj','on','Hessian','off',...
               'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
    for mc = 1:MC
       %%  generate observations 
        % generate N*K observations
        noise = sqrt(wavr)*randn(N,K);
        y_0 = noise - tau*ones(1,K);
        y_1 = h*ones(1,K)*theta + noise- tau*ones(1,K);
        % quantization 
        eta_0 = sum((sign(y_0)+1)/2,2)/K;
        eta_1 = sum((sign(y_1)+1)/2,2)/K;
        % inversion
        for i=1:N
           eta_0(i) = (sum(rand(fix(K*eta_0(i)),1)>=q_1) + sum(rand(fix(K-K*eta_0(i)),1)<=q_0))/K;
           eta_1(i) = (sum(rand(fix(K*eta_1(i)),1)>=q_1) + sum(rand(fix(K-K*eta_1(i)),1)<=q_0))/K;
        end 
        % unlabel
        eta_p_0 = Perm*eta_0;
        eta_p_1 = Perm*eta_1;
       %% MLE  theta / Pi known (case a)
        l_num_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        l_num_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        T_0_a(mc) = l_num_0-l_den_0;
        T_1_a(mc) = l_num_1-l_den_1;        
       %% MLE  theta unknown / Pi known (case b)
        theta_0 = fminunc(@(theta)Fun_Q_T(eta_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_1 = fminunc(@(theta)Fun_Q_T(eta_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        l_num_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_0-tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_0-tau)/sqrt(wavr))));
        l_den_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr))));
        l_num_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_1-tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_1-tau)/sqrt(wavr))));
        l_den_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr))));
        T_0_b(mc) = l_num_0-l_den_0;
        T_1_b(mc) = l_num_1-l_den_1;  
       %% MLE  theta known / Pi unknown (case c)
        [~,num_index]=sort(h*theta-tau); % sort h*theta-tau
        [~,den_index]=sort(-tau);
        [~,eta_p_0_index]=sort(eta_p_0); % sort H0 unlabeled observation statistics 
        [~,eta_p_1_index]=sort(eta_p_1); % sort H1 
        
        Pi_num = A(num_index, :); % sorting permutation
        Pi_den = A(den_index, :);
        Pi_eta_p_0 = A(eta_p_0_index, :);
        Pi_eta_p_1 = A(eta_p_1_index, :);       
        
        Pi_num_0_est = Pi_eta_p_0'*Pi_num; % H0 estimated permutation
        Pi_den_0_est = Pi_eta_p_0'*Pi_den;
        Pi_num_1_est = Pi_eta_p_1'*Pi_num; % H1 estimated permutation
        Pi_den_1_est = Pi_eta_p_1'*Pi_den;
        
        l_num_0 = (Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_0 = (Pi_den_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-Pi_den_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        l_num_1 = (Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_1 = (Pi_den_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-Pi_den_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        T_0_c(mc) = l_num_0-l_den_0;
        T_1_c(mc) = l_num_1-l_den_1;  
       %% MLE  theta / Pi unknown, normal initial points(case d)
        % alternating minimization
        % H1: normal initial point Delta
        theta_new = Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); % sort h*theta-tau
            Pi_num = A(num_index, :); % sorting permutation    
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_1 = theta_new;
        l_H1_1=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr))));
        % H1: normal initial point -Delta
        theta_new = -Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);   
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_2 = theta_new;
        l_H1_2=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr))));
        % H1: pick the larger MLE
        l_num_1 = max(l_H1_1,l_H1_2); 
        
        % H0: normal initial point Delta
        theta_new = Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :); 
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_1 = theta_new; 
        l_H0_1=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr))));
        % H0: normal initial point -Delta
        theta_new = -Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);  
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_2 = theta_new; 
        l_H0_2=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr))));
        % H0: pick the larger MLE 
        l_num_0 = max(l_H0_1,l_H0_2); 
        
        T_0_d(mc) = l_num_0-l_den_0;
        T_1_d(mc) = l_num_1-l_den_1;    
       %% MLE  theta / Pi unknown, good initial points(case e)
        % unlabeled eta projection to I
        lbound = q_0+(1-q_0-q_1)*normcdf(min(-abs(h)*Delta-tau));% project eta to I(eta)
        ubound = q_0+(1-q_0-q_1)*normcdf(max(+abs(h)*Delta-tau));
        I_0 = eta_p_0;
        I_1 = eta_p_1;
        for i = 1:N
                if I_0(i) <= lbound
                    I_0(i) = lbound+0.001;
                end
                if I_0(i) >= ubound
                    I_0(i) = ubound-0.001;
                end
                if I_1(i) <= lbound
                    I_1(i) = lbound+0.001;
                end
                if I_1(i) >= ubound
                    I_1(i) = ubound-0.001;
                end
        end
       % good initial points
        m_0 = sqrt(wavr)* norminv((I_0-q_0)/(1-q_0-q_1)); % H0 m
        if (m_0'*m_0-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2>=0
            theta_m0_1 = (tau'* h)/(h'*h) + sqrt((m_0'*m_0-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
            theta_m0_2 = (tau'* h)/(h'*h) - sqrt((m_0'*m_0-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
        else
            theta_m0_1 = Delta;
            theta_m0_2 = -Delta;
        end
        theta_m0_1 = Interval_Delta(theta_m0_1,Delta); % project to [-Delta,Delta]
        theta_m0_2 = Interval_Delta(theta_m0_2,Delta);
        
        m_1 = sqrt(wavr)* norminv((I_1-q_0)/(1-q_0-q_1)); % H1 m
        if (m_1'*m_1-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2>=0
            theta_m1_1 = (tau'* h)/(h'*h) + sqrt((m_1'*m_1-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
            theta_m1_2 = (tau'* h)/(h'*h) - sqrt((m_1'*m_1-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
        else
            theta_m1_1 = Delta;
            theta_m1_2 = -Delta;
        end
        theta_m1_1 = Interval_Delta(theta_m1_1,Delta);
        theta_m1_2 = Interval_Delta(theta_m1_2,Delta);

        % alternating minimization
        % H1: good initial point theta_1
        theta_new = theta_m1_1; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); % sort h*theta-tau
            Pi_num = A(num_index, :); % sorting permutation    
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_1 = theta_new; 
        l_H1_1=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr))));
        % H1: good initial point theta_2
        theta_new = theta_m1_2; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);   
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_2 = theta_new;
        l_H1_2=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr))));
        % H1: pick the larger MLE
        l_num_1 = max(l_H1_1,l_H1_2); 
        
        % H0: good initial point theta_1
        theta_new = theta_m0_1; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :); 
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_1 = theta_new; 
        l_H0_1=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr))));
        % H0: good initial point theta_1
        theta_new = theta_m0_2; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);  
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_2 = theta_new; 
        l_H0_2=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr))));
        % H0: pick the larger MLE 
        l_num_0 = max(l_H0_1,l_H0_2); 
        
        T_0_e(mc) = l_num_0-l_den_0;
        T_1_e(mc) = l_num_1-l_den_1;
    end
    P_d_K_NP(K_index) = MC_Pd(T_0_a,T_1_a,MC,P_fa);
    P_d_K_label(K_index) = MC_Pd(T_0_b,T_1_b,MC,P_fa); 
    P_d_K_reorder(K_index) = MC_Pd(T_0_c,T_1_c,MC,P_fa);
    P_d_K_unlabel_normal(K_index) = MC_Pd(T_0_d,T_1_d,MC,P_fa);
    P_d_K_unlabel_good(K_index) = MC_Pd(T_0_e,T_1_e,MC,P_fa);
end
% load('Pd_vs_N_equispaced_h_20170828.mat');
X1=K_sam;
Y1=P_d_K_NP;
Z1=P_d_K_label;
V1=P_d_K_reorder;
W1=P_d_K_unlabel_normal;
%% Subparagraph b: sinusoidal signal
h_x=rand(N,1);
h_x_ascend=sort(h_x,'ascend');
h=sin(2*pi*h_x_ascend);
tau=2*Delta*(rand(N,1)-0.5);
% h   = [  0.8441   0.9457   0.9511   0.9583   0.9813   0.8434   0.7035   0.6851   0.4941   0.1695   0.1333  -0.0040  -0.1966  -0.4215  -0.7669  -0.9159  -0.9318  -0.9373  -0.4842  -0.3699]';
% tau = [ -0.2662   0.5539   1.0422   0.9742   0.9356  -0.0313   1.1529   1.6190   0.5164   1.5550   0.9644   1.8434   1.4548   1.2462   0.6496  -1.7689  -1.8761  -0.2137   1.9555  -1.4672]';
%%%%%%%%%%%%
T= zeros(1,MC);
T_0_a = T;
T_1_a = T;
T_0_b = T;
T_1_b = T;
T_0_c = T;
T_1_c = T;
T_0_d = T;
T_1_d = T;
T_0_e = T;
T_1_e = T;
P_d_K_NP = zeros(1,length(K_sam)); % blank matrix 
P_d_K_label = zeros(1,length(K_sam));
P_d_K_reorder = zeros(1,length(K_sam));
P_d_K_unlabel_normal = zeros(1,length(K_sam));
P_d_K_unlabel_good = zeros(1,length(K_sam));
% MC simulation
for K_index = 1:length(K_sam) % time
    K = K_sam(K_index);
%     [P_d_K_NP(K_index), P_d_K_label(K_index), P_d_K_reorder(K_index), P_d_K_unlabel(K_index)] = LRT(N,K,h,wavr,tau,theta,MC,P_fa,q_0,q_1,T,Perm);
    options = optimset('Largescale','off','GradObj','on','Hessian','off',...
               'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
    for mc = 1:MC
       %%  generate observations 
        % generate N*K observations
        noise = sqrt(wavr)*randn(N,K);
        y_0 = noise - tau*ones(1,K);
        y_1 = h*ones(1,K)*theta + noise- tau*ones(1,K);
        % quantization 
        eta_0 = sum((sign(y_0)+1)/2,2)/K;
        eta_1 = sum((sign(y_1)+1)/2,2)/K;
        % inversion
        for i=1:N
           eta_0(i) = (sum(rand(fix(K*eta_0(i)),1)>=q_1) + sum(rand(fix(K-K*eta_0(i)),1)<=q_0))/K;
           eta_1(i) = (sum(rand(fix(K*eta_1(i)),1)>=q_1) + sum(rand(fix(K-K*eta_1(i)),1)<=q_0))/K;
        end 
        % unlabel
        eta_p_0 = Perm*eta_0;
        eta_p_1 = Perm*eta_1;
       %% MLE  theta / Pi known (case a)
        l_num_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        l_num_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        T_0_a(mc) = l_num_0-l_den_0;
        T_1_a(mc) = l_num_1-l_den_1;        
       %% MLE  theta unknown / Pi known (case b)
        theta_0 = fminunc(@(theta)Fun_Q_T(eta_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_1 = fminunc(@(theta)Fun_Q_T(eta_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        l_num_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_0-tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_0-tau)/sqrt(wavr))));
        l_den_0 = (eta_0)'*log(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr)))+(1-eta_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr))));
        l_num_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_1-tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_1-tau)/sqrt(wavr))));
        l_den_1 = (eta_1)'*log(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr)))+(1-eta_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr))));
        T_0_b(mc) = l_num_0-l_den_0;
        T_1_b(mc) = l_num_1-l_den_1;  
       %% MLE  theta known / Pi unknown (case c)
        [~,num_index]=sort(h*theta-tau); % sort h*theta-tau
        [~,den_index]=sort(-tau);
        [~,eta_p_0_index]=sort(eta_p_0); % sort H0 unlabeled observation statistics 
        [~,eta_p_1_index]=sort(eta_p_1); % sort H1 
        
        Pi_num = A(num_index, :); % sorting permutation
        Pi_den = A(den_index, :);
        Pi_eta_p_0 = A(eta_p_0_index, :);
        Pi_eta_p_1 = A(eta_p_1_index, :);       
        
        Pi_num_0_est = Pi_eta_p_0'*Pi_num; % H0 estimated permutation
        Pi_den_0_est = Pi_eta_p_0'*Pi_den;
        Pi_num_1_est = Pi_eta_p_1'*Pi_num; % H1 estimated permutation
        Pi_den_1_est = Pi_eta_p_1'*Pi_den;
        
        l_num_0 = (Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_0 = (Pi_den_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-Pi_den_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        l_num_1 = (Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_1 = (Pi_den_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-Pi_den_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        T_0_c(mc) = l_num_0-l_den_0;
        T_1_c(mc) = l_num_1-l_den_1;  
       %% MLE  theta / Pi unknown, normal initial points(case d)
        % alternating minimization
        % H1: normal initial point Delta
        theta_new = Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); % sort h*theta-tau
            Pi_num = A(num_index, :); % sorting permutation    
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_1 = theta_new;
        l_H1_1=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr))));
        % H1: normal initial point -Delta
        theta_new = -Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);   
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_2 = theta_new;
        l_H1_2=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr))));
        % H1: pick the larger MLE
        l_num_1 = max(l_H1_1,l_H1_2); 
        
        % H0: normal initial point Delta
        theta_new = Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :); 
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_1 = theta_new; 
        l_H0_1=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr))));
        % H0: normal initial point -Delta
        theta_new = -Delta; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);  
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_2 = theta_new; 
        l_H0_2=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr))));
        % H0: pick the larger MLE 
        l_num_0 = max(l_H0_1,l_H0_2); 
        
        T_0_d(mc) = l_num_0-l_den_0;
        T_1_d(mc) = l_num_1-l_den_1;    
       %% MLE  theta / Pi unknown, good initial points(case e)
        % unlabeled eta projection to I
        lbound = q_0+(1-q_0-q_1)*normcdf(min(-abs(h)*Delta-tau));% project eta to I(eta)
        ubound = q_0+(1-q_0-q_1)*normcdf(max(+abs(h)*Delta-tau));
        I_0 = eta_p_0;
        I_1 = eta_p_1;
        for i = 1:N
                if I_0(i) <= lbound
                    I_0(i) = lbound+0.001;
                end
                if I_0(i) >= ubound
                    I_0(i) = ubound-0.001;
                end
                if I_1(i) <= lbound
                    I_1(i) = lbound+0.001;
                end
                if I_1(i) >= ubound
                    I_1(i) = ubound-0.001;
                end
        end
       % good initial points
        m_0 = sqrt(wavr)* norminv((I_0-q_0)/(1-q_0-q_1)); % H0 m
        if (m_0'*m_0-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2>=0
            theta_m0_1 = (tau'* h)/(h'*h) + sqrt((m_0'*m_0-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
            theta_m0_2 = (tau'* h)/(h'*h) - sqrt((m_0'*m_0-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
        else
            theta_m0_1 = Delta;
            theta_m0_2 = -Delta;
        end
        theta_m0_1 = Interval_Delta(theta_m0_1,Delta); % project to [-Delta,Delta]
        theta_m0_2 = Interval_Delta(theta_m0_2,Delta);
        
        m_1 = sqrt(wavr)* norminv((I_1-q_0)/(1-q_0-q_1)); % H1 m
        if (m_1'*m_1-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2>=0
            theta_m1_1 = (tau'* h)/(h'*h) + sqrt((m_1'*m_1-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
            theta_m1_2 = (tau'* h)/(h'*h) - sqrt((m_1'*m_1-tau'*tau)/(h'*h)+((tau'* h)/(h'*h))^2);
        else
            theta_m1_1 = Delta;
            theta_m1_2 = -Delta;
        end
        theta_m1_1 = Interval_Delta(theta_m1_1,Delta);
        theta_m1_2 = Interval_Delta(theta_m1_2,Delta);

        % alternating minimization
        % H1: good initial point theta_1
        theta_new = theta_m1_1; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); % sort h*theta-tau
            Pi_num = A(num_index, :); % sorting permutation    
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_1 = theta_new; 
        l_H1_1=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_1-tau)/sqrt(wavr))));
        % H1: good initial point theta_2
        theta_new = theta_m1_2; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);   
            Pi_num_1_est = Pi_eta_p_1'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H1_2 = theta_new;
        l_H1_2=(Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H1_2-tau)/sqrt(wavr))));
        % H1: pick the larger MLE
        l_num_1 = max(l_H1_1,l_H1_2); 
        
        % H0: good initial point theta_1
        theta_new = theta_m0_1; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :); 
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_1 = theta_new; 
        l_H0_1=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_1-tau)/sqrt(wavr))));
        % H0: good initial point theta_1
        theta_new = theta_m0_2; 
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); 
            Pi_num = A(num_index, :);  
            Pi_num_0_est = Pi_eta_p_0'*Pi_num; 
            theta_new = fminunc(@(theta)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_H0_2 = theta_new; 
        l_H0_2=(Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_H0_2-tau)/sqrt(wavr))));
        % H0: pick the larger MLE 
        l_num_0 = max(l_H0_1,l_H0_2); 
        
        T_0_e(mc) = l_num_0-l_den_0;
        T_1_e(mc) = l_num_1-l_den_1;
    end
    P_d_K_NP(K_index) = MC_Pd(T_0_a,T_1_a,MC,P_fa);
    P_d_K_label(K_index) = MC_Pd(T_0_b,T_1_b,MC,P_fa); 
    P_d_K_reorder(K_index) = MC_Pd(T_0_c,T_1_c,MC,P_fa);
    P_d_K_unlabel_normal(K_index) = MC_Pd(T_0_d,T_1_d,MC,P_fa);
    P_d_K_unlabel_good(K_index) = MC_Pd(T_0_e,T_1_e,MC,P_fa);
end
% load('Pd_vs_K_sine_20170829.mat')
X2=K_sam;
Y2=P_d_K_NP;
Z2=P_d_K_label;
V2=P_d_K_reorder;
W2=P_d_K_unlabel_normal;
U2=P_d_K_unlabel_good;
%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 5;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
figure(1)
subplot(1,2,1);
plot(X1,Y1,':ro',X1,Z1,'-kx',X1,V1,'-.b*',X1,W1,'--ms','LineWidth',lw,'MarkerSize',msz)
set(gca,'XLim',[10 140]);
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('P_D','Fontsize',15)
h=legend('NP detector','GLRT detector (9)','Detector $$T_2(\tilde\eta)$$ (19)','Detector $$T_3(\tilde\eta)$$ (20)');
set(h,'Interpreter','latex')
text(120,0.9,'(a)','Fontsize',20);

subplot(1,2,2);
plot(K_sam,P_d_K_NP,':ro',K_sam,P_d_K_label,'-kx',K_sam,P_d_K_reorder,'-.b*',K_sam,P_d_K_unlabel_normal,'-b+',K_sam,P_d_K_unlabel_good,'--ms','LineWidth',lw,'MarkerSize',msz)
set(gca,'XLim',[10 120]);
xlabel('$N$','interpreter','latex','Fontsize',fsz);
ylabel('P_D','Fontsize',15);
h=legend('NP detector','GLRT detector (9)','Detector $$T_2(\tilde\eta)$$ (19)','Detector $$T_3(\tilde\eta)$$ (20)', 'Detector $$T_3(\tilde\eta)$$ (20), good initial points');
set(h,'Interpreter','latex')
text(105,0.9,'(b)','Fontsize',20);





