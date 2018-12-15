%% initialization
clc;
clear;
close all
options = optimset('Largescale','off','GradObj','on','Hessian','off',...
            'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
MC = 1000;
theta = 1.5; 
Delta = 2;  
q_0 = 0;
q_1 = 0;
wavr =1;
% K = 1000;
N_sam = [10 20 30 100 300 1000 3000 1e4 3e4 1e5]';

tm_eq = zeros(1,length(N_sam));
tm_rd1 = zeros(1,length(N_sam));
tm_rd2 = zeros(1,length(N_sam));
tm_sine = zeros(length(N_sam),MC);

for N_index = 1:length(N_sam) % time
    N = N_sam(N_index);
    pu = zeros(N,1);
    t = zeros(1,N-1);
    %% asymmetric h
%     h = flip(linspace(-2,2,N)'+0.5);
    h = randn(N,1);
    h = sort(h,'descend');
    c=0.5;
    tau = c* h;
    for i = 1:N
        pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
    end
    for i = 1:(N-1)
        t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
    end
    tm_rd1(N_index) = min(t);
    
    h = randn(N,1);
    h = sort(h,'descend');
    tau = 0.5 * h;
    for i = 1:N
        pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
    end
    for i = 1:(N-1)
        t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
    end
    tm_rd2(N_index) = min(t);
       
    % ramp signal
    h = flip(linspace(-0.8,1,N)');
    tau = 0.5 * h;
    for i = 1:N
        pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
    end
    for i = 1:(N-1)
        t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
    end
    tm_eq(N_index) = min(t);
    
        % sine signal
        for mc = 1:MC
            h_x=rand(N,1);
            h = sin(2*pi*h_x);
            tau =0.5*h;
            for i = 1:N
                pu(i) = q_0+(1-q_0-q_1)*normcdf((h(i)*theta-tau(i))/sqrt(wavr));  % p_i    
            end
            pu=sort(pu,'descend');
            for i = 1:(N-1)
                t(i) = (pu(i)-pu(i+1))/sqrt(pu(i)*(1-pu(i))+pu(i+1)*(1-pu(i+1)));  % p_i-p_{i+1}/sqrt()
            end
            tm_sine(N_index,mc) = min(t);      
        end        
end
tm_sine_log = log(tm_sine);
% least squres
% assume log t = a*log N+b
t_log = mean(tm_sine_log,2);
N_log = log(N_sam);
a = ( length(N_sam)*sum(N_log.*t_log)-sum(N_log)*sum(t_log) ) / (length(N_sam)*sum(N_log.*N_log)-(sum(N_log))^2 );
b = (sum(N_log.*N_log)*sum(t_log)-sum(N_log)*sum(N_log.*t_log) ) / ( length(N_sam)*sum(N_log.*N_log)-(sum(N_log))^2 );
%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
figure(1);
loglog(N_sam,tm_eq,'-b*',N_sam,sqrt(2)*0.4355./N_sam.^1,'-.r',N_sam,sqrt(2)*0.6717./N_sam.^1,'-.m',N_sam,tm_rd1,'-bo',N_sam,tm_rd2,'-k+',sqrt(2)*N_sam,1./N_sam.^2,':r',N_sam,tm_sine(:,1),':bx',N_sam,tm_sine(:,2),':ks',N_sam,exp(b)*N_sam.^a,'--r','LineWidth',lw,'MarkerSize',msz);
set(gca,'XLim',[10 1e5]);
% set(gca,'YLim',[1 1e-14]);
xlabel('$K$','interpreter','latex','Fontsize',fsz);
ylabel('t','Fontsize',15);
% legend('the value of t at different N');
leg=legend('t, ramp signal realization','$$\sqrt{2}\tilde{t}=\sqrt{2}c_e/K$$','$$\sqrt{2}\tilde{t}=\sqrt{2}c_{ea}/K$$, accurate','t, random realization one','t, random realization two','$$\sqrt{2}\tilde{t}=\sqrt{2}c_g/K^2$$','t, sinusoidal signal realization one','t, sinusoidal signal realization two','t, fitted sinusoidal signal');
set(leg,'Interpreter','latex');
% save t_vs_N_20170830;






