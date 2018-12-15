clc;
clear;
close all;
%% initialization
MC = 1000;
options = optimset('Largescale','off','GradObj','on','Hessian','off',...
            'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
N = 20;
theta = 1.5; 
Delta = 2;  
q_0 = 0;
q_1 = 0;
wavr =1;
A = eye(N);  % generate a random permutation matrix 'Perm'
idx = randperm(N);
Perm = A(idx, :);  
%% Subgraph a: ramp signal
h = flip(linspace(-0.8,1,N)');
c=0.5;
tau = c * h;
K_sam = [ 200 316 462 685 1000 1480 2160 3160 4620 6850 10000];
%%%%%%%%%
Prec_experiment = zeros(1,length(K_sam)); % reordering , known theta
Prec_experiment_am = zeros(1,length(K_sam)); % alternating maximization, unknwon theta and Pi
Prec_lower = zeros(1,length(K_sam));
Prec_lower_ori = zeros(1,length(K_sam));
Prec_lower_tilde = zeros(1,length(K_sam));
Prec_lower_ori_tilde = zeros(1,length(K_sam));
pu = zeros(N,1);
t = zeros(1,N-1);
tt= zeros(1,N-1);
%%%%%%%%%%%
for i = 1:N
    pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
end
for i = 1:(N-1)
   t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
   tt(i)= (pu(i)-pu(i+1));
end
tm = min(t);
ttm= min(tt);
for K_index = 1:length(K_sam) % time
    K = K_sam(K_index);
    for mc = 1:MC
        % generate N*K observations
        y = h*ones(1,K)*theta+sqrt(wavr)*randn(N,K)-tau*ones(1,K);
        % quantization and inversion
        eta = sum((sign(y)+1)/2,2)/K;
        for i=1:N
           eta(i) = (sum(rand(fix(K*eta(i)),1)>=q_1) + sum(rand(fix(K-K*eta(i)),1)<=q_0))/K;
        end  
        eta_perm = Perm' * eta;
        % reordering, theta known, Pi unknown
        [eta_perm_sort,eta_perm_index]=sort(eta_perm,'descend');  % sort eta_perm
        Pi_rec = A(eta_perm_index, :);
        if Pi_rec == Perm
            Prec_experiment(K_index)=Prec_experiment(K_index)+1;
        end        
       % alternating maximization, theta and Pi unknown
        [h_asc,h_asc_index]=sort(h,'ascend'); 
        [h_des,h_des_index]=sort(h,'descend'); 
        [eta_perm_asc,eta_perm_asc_index]=sort(eta_perm,'ascend');  % sort eta_perm
        Perm_h_asc = A(h_asc_index, :);
        Perm_h_des = A(h_des_index, :);
        Perm_eta_perm_asc = A(eta_perm_asc_index, :);
        Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
        Pi_des = Perm_h_des'*Perm_eta_perm_asc; % recover permutation
        
        theta_asc = fminunc(@(theta)Fun_Q_T(Pi_asc'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_des = fminunc(@(theta)Fun_Q_T(Pi_des'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
                
        l_asc = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)));
        l_des = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)));
        if l_asc > l_des
            Pi_rec_am = Pi_asc;
        else
            Pi_rec_am = Pi_des;
        end
        if Pi_rec_am == Perm
            Prec_experiment_am(K_index)=Prec_experiment_am(K_index)+1;
        end
    end
    % lower bounded Pr(N,K)
    Prec_lower_ori(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(tm)-0.5*log(K)-tm^2/2*K);
    if Prec_lower_ori(K_index)<0
        Prec_lower(K_index)=0;
    else
        Prec_lower(K_index)=Prec_lower_ori(K_index);
    end

   % lower bounded ~Pr(N,K)
      Prec_lower_ori_tilde(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(sqrt(2)*ttm)-0.5*log(K)-(ttm*sqrt(2))^2/2*K);
    if Prec_lower_ori_tilde(K_index)<0
        Prec_lower_tilde(K_index)=0;
    else
        Prec_lower_tilde(K_index)=Prec_lower_ori_tilde(K_index);
    end
end
Prec_experiment = Prec_experiment/MC;
Prec_experiment_am = Prec_experiment_am/MC;
% load ('Prec_vs_K_data_equispaced_20170706.mat');
X1=K_sam;
Y1=Prec_experiment;
Z1=Prec_experiment_am;
V1=Prec_lower;
W1=Prec_lower_tilde;
%% Subgraph b: random generated h
h = randn(N,1);
h = sort(h,'descend');
c=0.5;
tau = c * h;
K_sam = [ 1e3 2.15*1e3 4.62*1e3 1e4 2.15*1e4 4.62*1e4 1e5 2.15*1e5 4.62*1e5 1e6 1.5*1e6 2.25*1e6 3.375*1e6 5.0625*1e6 ];
%%%%%%%%%
Prec_experiment = zeros(1,length(K_sam)); % reordering , known theta
Prec_experiment_am = zeros(1,length(K_sam)); % alternating maximization, unknwon theta and Pi
Prec_lower = zeros(1,length(K_sam));
Prec_lower_ori = zeros(1,length(K_sam));
Prec_lower_tilde = zeros(1,length(K_sam));
Prec_lower_ori_tilde = zeros(1,length(K_sam));
pu = zeros(N,1);
t = zeros(1,N-1);
tt= zeros(1,N-1);
%%%%%%%%%%%
for i = 1:N
    pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
end
for i = 1:(N-1)
   t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
   tt(i)= (pu(i)-pu(i+1));
end
tm = min(t);
ttm= min(tt);
for K_index = 1:length(K_sam) % time
    K = K_sam(K_index);
    for mc = 1:MC
        % generate N*K observations
        y = h*ones(1,K)*theta+sqrt(wavr)*randn(N,K)-tau*ones(1,K);
        % quantization and inversion
        eta = sum((sign(y)+1)/2,2)/K;
        for i=1:N
           eta(i) = (sum(rand(fix(K*eta(i)),1)>=q_1) + sum(rand(fix(K-K*eta(i)),1)<=q_0))/K;
        end  
        eta_perm = Perm' * eta;
        % reordering, theta known, Pi unknown
        [eta_perm_sort,eta_perm_index]=sort(eta_perm,'descend');  % sort eta_perm
        Pi_rec = A(eta_perm_index, :);
        if Pi_rec == Perm
            Prec_experiment(K_index)=Prec_experiment(K_index)+1;
        end        
       % alternating maximization, theta and Pi unknown
        [h_asc,h_asc_index]=sort(h,'ascend'); 
        [h_des,h_des_index]=sort(h,'descend'); 
        [eta_perm_asc,eta_perm_asc_index]=sort(eta_perm,'ascend');  % sort eta_perm
        Perm_h_asc = A(h_asc_index, :);
        Perm_h_des = A(h_des_index, :);
        Perm_eta_perm_asc = A(eta_perm_asc_index, :);
        Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
        Pi_des = Perm_h_des'*Perm_eta_perm_asc; % recover permutation
        
        theta_asc = fminunc(@(theta)Fun_Q_T(Pi_asc'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_des = fminunc(@(theta)Fun_Q_T(Pi_des'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
                
        l_asc = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)));
        l_des = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)));
        if l_asc > l_des
            Pi_rec_am = Pi_asc;
        else
            Pi_rec_am = Pi_des;
        end
        if Pi_rec_am == Perm
            Prec_experiment_am(K_index)=Prec_experiment_am(K_index)+1;
        end
    end
    % lower bounded Pr(N,K)
    Prec_lower_ori(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(tm)-0.5*log(K)-tm^2/2*K);
    if Prec_lower_ori(K_index)<0
        Prec_lower(K_index)=0;
    else
        Prec_lower(K_index)=Prec_lower_ori(K_index);
    end
   % lower bounded ~Pr(N,K)
      Prec_lower_ori_tilde(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(sqrt(2)*ttm)-0.5*log(K)-(ttm*sqrt(2))^2/2*K);
    if Prec_lower_ori_tilde(K_index)<0
        Prec_lower_tilde(K_index)=0;
    else
        Prec_lower_tilde(K_index)=Prec_lower_ori_tilde(K_index);
    end
end
Prec_experiment = Prec_experiment/MC;
Prec_experiment_am = Prec_experiment_am/MC;
% load ('Prec_vs_K_data_gaussian_20170706.mat');
X2=K_sam;
Y2=Prec_experiment;
Z2=Prec_experiment_am;
V2=Prec_lower;
W2=Prec_lower_tilde;
%% Subgraph c: sinusoidal signal
h_x=rand(N,1);
h_x_ascend=sort(h_x,'ascend');
h=sin(2*pi*h_x_ascend);
tau=2*Delta*(rand(N,1)-0.5);
% h   = [  0.8441   0.9457   0.9511   0.9583   0.9813   0.8434   0.7035   0.6851   0.4941   0.1695   0.1333  -0.0040  -0.1966  -0.4215  -0.7669  -0.9159  -0.9318  -0.9373  -0.4842  -0.3699]';
% tau = [ -0.2662   0.5539   1.0422   0.9742   0.9356  -0.0313   1.1529   1.6190   0.5164   1.5550   0.9644   1.8434   1.4548   1.2462   0.6496  -1.7689  -1.8761  -0.2137   1.9555  -1.4672]';
K_sam = [ 1e3 2.15*1e3 4.62*1e3 1e4 2.15*1e4 4.62*1e4 1e5 2.15*1e5 4.62*1e5 1e6 1.5*1e6 2.25*1e6 3.375*1e6 5.0625*1e6 ];
%%%%%%%%%
Prec_experiment = zeros(1,length(K_sam)); % reordering , known theta
Prec_experiment_am = zeros(1,length(K_sam)); % alternating maximization, unknwon theta and Pi
Prec_lower = zeros(1,length(K_sam));
Prec_lower_ori = zeros(1,length(K_sam));
Prec_lower_tilde = zeros(1,length(K_sam));
Prec_lower_ori_tilde = zeros(1,length(K_sam));
pu = zeros(N,1);
t = zeros(1,N-1);
tt= zeros(1,N-1);
%%%%%%%%%%%
for i = 1:N
    pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
end
for i = 1:(N-1)
   t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
   tt(i)= (pu(i)-pu(i+1));
end
tm = min(t);
ttm= min(tt);
for K_index = 1:length(K_sam) % time
    K = K_sam(K_index);
    for mc = 1:MC
        % generate N*K observations
        y = h*ones(1,K)*theta+sqrt(wavr)*randn(N,K)-tau*ones(1,K);
        % quantization and inversion
        eta = sum((sign(y)+1)/2,2)/K;
        for i=1:N
           eta(i) = (sum(rand(fix(K*eta(i)),1)>=q_1) + sum(rand(fix(K-K*eta(i)),1)<=q_0))/K;
        end  
        eta_perm = Perm' * eta;
        % reordering, theta known, Pi unknown
        [eta_perm_sort,eta_perm_index]=sort(eta_perm,'descend');  % sort eta_perm
        Pi_rec = A(eta_perm_index, :);
        if Pi_rec == Perm
            Prec_experiment(K_index)=Prec_experiment(K_index)+1;
        end        
       % alternating maximization, theta and Pi unknown
        [h_asc,h_asc_index]=sort(h,'ascend'); 
        [h_des,h_des_index]=sort(h,'descend'); 
        [eta_perm_asc,eta_perm_asc_index]=sort(eta_perm,'ascend');  % sort eta_perm
        Perm_h_asc = A(h_asc_index, :);
        Perm_h_des = A(h_des_index, :);
        Perm_eta_perm_asc = A(eta_perm_asc_index, :);
        Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
        Pi_des = Perm_h_des'*Perm_eta_perm_asc; % recover permutation
        
        theta_asc = fminunc(@(theta)Fun_Q_T(Pi_asc'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_des = fminunc(@(theta)Fun_Q_T(Pi_des'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
                
        l_asc = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)));
        l_des = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)));
        if l_asc > l_des
            Pi_rec_am = Pi_asc;
        else
            Pi_rec_am = Pi_des;
        end
        if Pi_rec_am == Perm
            Prec_experiment_am(K_index)=Prec_experiment_am(K_index)+1;
        end
    end
    % lower bounded Pr(N,K)
    Prec_lower_ori(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(tm)-0.5*log(K)-tm^2/2*K);
    if Prec_lower_ori(K_index)<0
        Prec_lower(K_index)=0;
    else
        Prec_lower(K_index)=Prec_lower_ori(K_index);
    end

   % lower bounded ~Pr(N,K)
      Prec_lower_ori_tilde(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(sqrt(2)*ttm)-0.5*log(K)-(ttm*sqrt(2))^2/2*K);
    if Prec_lower_ori_tilde(K_index)<0
        Prec_lower_tilde(K_index)=0;
    else
        Prec_lower_tilde(K_index)=Prec_lower_ori_tilde(K_index);
    end
end
Prec_experiment = Prec_experiment/MC;
Prec_experiment_am = Prec_experiment_am/MC;
% load('Prec_vs_N_sine_20170830.mat');
X3=K_sam;
Y3=Prec_experiment;
Z3=Prec_experiment_am;
V3=Prec_lower;
W3=Prec_lower_tilde;
%% Subgraph d: random generated h
K = 10000;
N_sam = [10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100];
Prec_experiment = zeros(1,length(N_sam)); % reordering , known theta
Prec_experiment_am = zeros(1,length(N_sam)); % alternating maximization, unknwon theta and Pi
Prec_lower = zeros(1,length(N_sam));
Prec_lower_ori = zeros(1,length(N_sam));
Prec_lower_tilde = zeros(1,length(N_sam));
Prec_lower_ori_tilde = zeros(1,length(N_sam));
for N_index = 1:length(N_sam) % time
    N = N_sam(N_index);    
    A = eye(N); 
    idx = randperm(N);
    Perm = A(idx, :);     
    pu = zeros(N,1);
    t = zeros(1,N-1);
    tt = zeros(1,N-1);
    %%%%%%%
    h = flip(linspace(-0.8,1,N)');
    c = 0.5;
    tau = c * h;
    %%%%%%%
    for i = 1:N
        pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
    end
    for i = 1:(N-1)
        t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));
        tt(i) = (pu(i)-pu(i+1)); 
    end
    tm = min(t);
    ttm = min(tt);
    % experiment
    for mc = 1:MC
        % generate N*K observations
        y = h*ones(1,K)*theta+sqrt(wavr)*randn(N,K)-tau*ones(1,K);
        % quantization and inversion
        eta = sum((sign(y)+1)/2,2)/K;
        for i=1:N
           eta(i) = (sum(rand(fix(K*eta(i)),1)>=q_1) + sum(rand(fix(K-K*eta(i)),1)<=q_0))/K;
        end  
        eta_perm = Perm' * eta;
       % reordering, theta known, Pi unknown
       [eta_perm_sort,eta_perm_index]=sort(eta_perm,'descend');  % sort eta_perm
        Pi_rec = A(eta_perm_index, :);
        if Pi_rec == Perm
            Prec_experiment(N_index)=Prec_experiment(N_index)+1;
        end
       % alternating maximization, theta and Pi unknown
        [h_asc,h_asc_index]=sort(h,'ascend'); 
        [h_des,h_des_index]=sort(h,'descend'); 
        [eta_perm_asc,eta_perm_asc_index]=sort(eta_perm,'ascend');  % sort eta_perm
        Perm_h_asc = A(h_asc_index, :);
        Perm_h_des = A(h_des_index, :);
        Perm_eta_perm_asc = A(eta_perm_asc_index, :);
        Pi_asc = Perm_h_asc'*Perm_eta_perm_asc; % recover permutation
        Pi_des = Perm_h_des'*Perm_eta_perm_asc; % recover permutation
        
        theta_asc = fminunc(@(theta)Fun_Q_T(Pi_asc'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_des = fminunc(@(theta)Fun_Q_T(Pi_des'*eta_perm,h,theta,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
                
        l_asc = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_asc-tau)/sqrt(wavr)));
        l_des = (Pi_asc'*eta_perm)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)))+(1-(Pi_asc'*eta_perm)')*log(1-q_0-(1-q_0-q_1)*normcdf((h*theta_des-tau)/sqrt(wavr)));
        if l_asc > l_des
            Pi_rec_am = Pi_asc;
        else
            Pi_rec_am = Pi_des;
        end
        if Pi_rec_am == Perm
            Prec_experiment_am(N_index)=Prec_experiment_am(N_index)+1;
        end  
    end    
    % lower bounded Pr(N,K)
    Prec_lower_ori(N_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(tm)-0.5*log(K)-tm^2/2*K);
    if Prec_lower_ori(N_index)<0
        Prec_lower(N_index)=0;
    else
        Prec_lower(N_index)=Prec_lower_ori(N_index);
    end
   % lower bounded ~Pr(N,K)
      Prec_lower_ori_tilde(N_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(sqrt(2)*ttm)-0.5*log(K)-(ttm*sqrt(2))^2/2*K);
    if Prec_lower_ori_tilde(N_index)<0
        Prec_lower_tilde(N_index)=0;
    else
        Prec_lower_tilde(N_index)=Prec_lower_ori_tilde(N_index);
    end
end
Prec_experiment = Prec_experiment/MC;
Prec_experiment_am = Prec_experiment_am/MC;
% load('Prec_vs_N_data_equispaced_20170706.mat');
X4=N_sam;
Y4=Prec_experiment;
Z4=Prec_experiment_am;
V4=Prec_lower;
W4=Prec_lower_tilde;
%% loading
% load ('Prec_vs_K_data_equispaced_20170706.mat');
% X1=K_sam;
% Y1=Prec_experiment;
% Z1=Prec_experiment_am;
% V1=Prec_lower;
% W1=Prec_lower_tilde;
% load ('Prec_vs_K_data_gaussian_20170706.mat');
% X2=K_sam;
% Y2=Prec_experiment;
% Z2=Prec_experiment_am;
% V2=Prec_lower;
% W2=Prec_lower_tilde;
% load('Prec_vs_N_sine_20170830.mat');
% X3=K_sam;
% Y3=Prec_experiment;
% Z3=Prec_experiment_am;
% V3=Prec_lower;
% W3=Prec_lower_tilde;
% load('Prec_vs_N_data_equispaced_20170706.mat');
% X4=N_sam;
% Y4=Prec_experiment;
% Z4=Prec_experiment_am;
% V4=Prec_lower;
% W4=Prec_lower_tilde;
%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 5;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
figure(1);
% Prec_vs_N_ramp_h
subplot(2,2,1); 
semilogx(X1,Y1,'-b*',X1,Z1,'--ro',X1,V1,'-.k+',X1,W1,':ms','LineWidth',lw,'MarkerSize',msz)
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('Permutation Recovery Probability','Fontsize',fsz)
set(gca,'XLim',[200 10000]);
text(225,0.9,'(a)','Fontsize',20);
leg=legend('experiment, $$\theta$$ known, reordering','experiment, $$\theta$$ unknown, AM','theorical approximation, $${\rm Pr}(K,N)$$','theorical approximation, $$\widetilde{{\rm Pr}}(K,N)$$');
set(leg,'Interpreter','latex');
% Prec_vs_N_gaussian_h
subplot(2,2,2);
semilogx(X2,Y2,'-b*',X2,Z2,'--ro',X2,V2,'-.k+',X2,W2,':ms','LineWidth',lw,'MarkerSize',msz)
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('Permutation Recovery Probability','Fontsize',fsz)
set(gca,'XLim',[1e3 5.0625*1e6]);
text(1400,0.9,'(b)','Fontsize',20);
% Prec_vs_N_sine_h
subplot(2,2,3);
semilogx(X3,Y3,'-b*',X3,Z3,'--ro',X3,V3,'-.k+',X3,W3,':ms','LineWidth',lw,'MarkerSize',msz)
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('Permutation Recovery Probability','Fontsize',fsz)
set(gca,'XLim',[1e3 4.62*1e6]);
text(1400,0.9,'(c)','Fontsize',20);
% Prec_vs_K_ramp_h
subplot(2,2,4);
plot(X4,Y4,'-b*',X4,Z4,'--ro',X4,V4,'-.k+',X4,W4,':ms','LineWidth',lw,'MarkerSize',msz)
set(gca,'XLim',[10 100]);
xlabel('$K$','interpreter','latex','Fontsize',fsz)
ylabel('Permutation Recovery Probability','Fontsize',fsz)
text(14,0.9,'(d)','Fontsize',20);






