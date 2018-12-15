clc;
clear;
close all;
%%
MC = 1000;
options = optimset('Largescale','off','GradObj','on','Hessian','off',...
            'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
N = 20;
K_sam = [ 200 316 462 685 1000 1480 2160 3160 4620 6850 1e4 1.48*1e4 2.16*1e4 3.16*1e4]; % alpha = 1, ct=0.4355;
theta = 1.5; 
Delta = 2;  
wavr =1;
% generate a random permutation matrix 'Perm'
A = eye(N); 
idx = randperm(N);
Perm = A(idx, :);  
% different flipping probability
q_value = [0 0.05 0.1 0.15];
Prec_experiment = zeros(1,length(K_sam)); % reordering , known theta
% Prec_experiment_am = zeros(1,length(K_sam)); % alternating maximization, unknwon theta and Pi
Prec_lower = zeros(1,length(K_sam));
Prec_lower_ori = zeros(1,length(K_sam));
Prec_lower_tilde = zeros(1,length(K_sam));
Prec_lower_ori_tilde = zeros(1,length(K_sam));
Prec_differ_q = zeros(length(q_value),length(K_sam));
Prec_differ_lower = zeros(length(q_value),length(K_sam));
Prec_differ_lower_tilde = zeros(length(q_value),length(K_sam));
pu = zeros(N,1);
t = zeros(1,N-1);
tt= zeros(1,N-1);
%% ramp signal
h = flip(linspace(-0.8,1,N)');
% h = randn(N,1);
% h = sort(h,'descend');
c=0.5;
tau = c * h;
%% Different (q_0,q_1)
for q_index = 1:length(q_value)
    q_0= q_value(q_index);
    q_1= q_value(q_index);
    for i = 1:N
        pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
    end
    for i = 1:(N-1)
       t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
       tt(i)= (pu(i)-pu(i+1));
    end
    tm = min(t);
    ttm= min(tt);
    %% experiment
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
            eta_perm = Perm' * eta;
            %% reordering, theta known, Pi unknown
            [eta_perm_sort,eta_perm_index]=sort(eta_perm,'descend');  % h is descending, so we can directly sort eta_perm
            Pi_rec = A(eta_perm_index, :);
            if Pi_rec == Perm
                Prec_experiment(K_index)=Prec_experiment(K_index)+1;
            end
        end
        %% lower bounded Pr(N,K)
        Prec_lower_ori(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(tm)-0.5*log(K)-tm^2/2*K);
    %     Prec_lower_ori(K_index)=1-1/sqrt(2*pi) * exp(3*log(N)-0.5*log(K)-K/(2*N^4));
        if Prec_lower_ori(K_index)<0
            Prec_lower(K_index)=0;
        else
            Prec_lower(K_index)=Prec_lower_ori(K_index);
        end

       %% lower bounded ~Pr(N,K)
    %      Prec_lower_ori_tilde(K_index)=1-1/(2*ct*sqrt(pi)) * exp((1+alpha)*log(N)-0.5*log(K)-ct^2*K/N^(2*alpha));
          Prec_lower_ori_tilde(K_index)=1-1/sqrt(2*pi) * exp(log(N-1)-log(sqrt(2)*ttm)-0.5*log(K)-(ttm*sqrt(2))^2/2*K);

        if Prec_lower_ori_tilde(K_index)<0
            Prec_lower_tilde(K_index)=0;
        else
            Prec_lower_tilde(K_index)=Prec_lower_ori_tilde(K_index);
        end
    end
    Prec_experiment = Prec_experiment/MC;
     % Prec_experiment_am = Prec_experiment_am/MC;
     
     Prec_differ_q(q_index,:) = Prec_experiment;
     Prec_differ_lower(q_index,:) = Prec_lower;
     Prec_differ_lower_tilde(q_index,:) = Prec_lower_tilde;
end

%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 10;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
figure(1);
semilogx(K_sam,Prec_differ_q(1,:),'-k*', K_sam*((1-2*q_value(1))/(1-2*q_value(2)))^2,Prec_differ_q(1,:),'-ro', K_sam,Prec_differ_q(2,:),'-.k', ...
    K_sam,Prec_differ_q(3,:),'-mx', K_sam*((1-2*q_value(1))/(1-2*q_value(3)))^2,Prec_differ_q(1,:),'--k',...
    K_sam,Prec_differ_q(4,:),'-b+',K_sam*((1-2*q_value(1))/(1-2*q_value(4)))^2,Prec_differ_q(1,:),':k',...
    'LineWidth',lw,'MarkerSize',msz);
xlabel('$N$','interpreter','latex','Fontsize',fsz)
ylabel('Permutation Recovery Probability','Fontsize',fsz)
set(gca,'XLim',[200 1.5*1e4]);
set(gca,'YLim',[0 1]);

leg=legend('experiment, $$q_0=q_1=0$$','prediction, $$q_0=q_1=0.05$$', 'experiment, $$q_0=q_1=0.05$$', 'prediction, $$q_0=q_1=0.1$$', 'experiment, $$q_0=q_1=0.1$$', 'prediction, $$q_0=q_1=0.15$$', 'experiment, $$q_0=q_1=0.15$$');
set(leg,'Interpreter','latex');
% save Differ_N_q01_20170901;

