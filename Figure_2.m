%% initialization
clc;
clear;
close all
%%%%%%
Delta = 2;  
q_0 = 0;
q_1 = 0;
N_sam = [10 17 31 56 100 178 317 563  1000 1780 3170 5630 10000 17800 31700 56300 1e5]';  % N is K in the paper.
%% h is Gau
t_Gau_a05 = zeros(length(N_sam),trials);
t_Gau_a1 = zeros(length(N_sam),trials);
t_Gau_a2 = zeros(length(N_sam),trials);
t_Gau_a4 = zeros(length(N_sam),trials);

for N_index = 1:length(N_sam) % time
    N = N_sam(N_index);
    p = zeros(N,1);
    t = zeros(1,N-1);
    for trial_index = 1:trials
        h = randn(N,1);
        h = sort(h,'descend');
        
        a=0.5; % a =1
        p = q_0+(1-q_0-q_1)*normcdf(a*h);  % p_i   
        v = (p(1:N-1)-p(2:N));
        t_Gau_a05(N_index,trial_index) = min(v);
        
        a=1; % a =0.25
        p = q_0+(1-q_0-q_1)*normcdf(a*h);  % p_i   
        v = (p(1:N-1)-p(2:N));
        t_Gau_a1(N_index,trial_index) = min(v);

        a=2; % a =1
        p=normcdf(a*h);
        neg_index=find(h<(-4/a));
        pos_index=find(h>(4/a));
        h_neg=h(neg_index);
        h_pos=h(pos_index);
        p_log=zeros(N,1);
        p_log(neg_index) = -log(2)-0.5*(a*h_neg).^2+log(erfcx(-(a*h_neg)/sqrt(2)));
        p_log(pos_index) = 1-(-log(2)-0.5*(-a*h_pos).^2+log(erfcx(-(-a*h_pos)/sqrt(2))));
        p(neg_index)=exp(p_log(neg_index));
        p(pos_index)=exp(p_log(pos_index));
        v = p(1:N-1)-p(2:N);
        t_Gau_a2(N_index,trial_index) = min(v);
        
        
        a=4; % a =1
        p=normcdf(a*h);
        neg_index=find(h<(-4/a));
        pos_index=find(h>(4/a));
        h_neg=h(neg_index);
        h_pos=h(pos_index);
        p_log=zeros(N,1);
        p_log(neg_index) = -log(2)-0.5*(a*h_neg).^2+log(erfcx(-(a*h_neg)/sqrt(2)));
        p_log(pos_index) = 1-(-log(2)-0.5*(-a*h_pos).^2+log(erfcx(-(-a*h_pos)/sqrt(2))));
        p(neg_index)=exp(p_log(neg_index));
        p(pos_index)=exp(p_log(pos_index));
        v = p(1:N-1)-p(2:N);
        t_Gau_a4(N_index,trial_index) = min(v);
    end
end

t_Gau_a05_mean=exp(mean(log(t_Gau_a05),2)); 
t_Gau_a1_mean=exp(mean(log(t_Gau_a1),2)); 
t_Gau_a2_mean=exp(mean(log(t_Gau_a2),2)); 
t_Gau_a4_mean=exp(mean(log(t_Gau_a4),2)); 
% 
%% figure
alw = 0.75;    % AxesLineWidth
fsz = 10;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
                   
figure(1);
subplot(2,2,1);
loglog(N_sam,t_Gau_a05(:,1),'--b',N_sam,t_Gau_a05(:,2),'-.k',N_sam,t_Gau_a05_mean,'-r','LineWidth',lw,'MarkerSize',msz);
% set(gca,'XLim',[10 1e5]);
% set(gca,'YLim',[min(min(log10(t_Gau_a05))) max(max(log10(t_Gau_a05)))]);
xlabel('$K$','interpreter','latex');
ylabel('$$\tilde t$$','interpreter','latex','Fontsize',15);
text(20,1e-11,'a=0.5','Fontsize',20);
text(3*1e4,1e-3,'(a)','Fontsize',20);
leg=legend('$$\tilde t$$, realization one','$$\tilde t$$, realization two','$$\tilde t$$, mean of 1000 realizations');
set(leg,'Interpreter','latex');
subplot(2,2,2);
loglog(N_sam,t_Gau_a1(:,1),'--b',N_sam,t_Gau_a1(:,2),'-.k',N_sam,t_Gau_a1_mean,'-r','LineWidth',lw,'MarkerSize',msz);
% set(gca,'XLim',[10 1e5]);
% set(gca,'YLim',[min(min(log10(t_Gau_a05))) max(max(log10(t_Gau_a05)))]);
xlabel('$K$','interpreter','latex');
ylabel('$$\tilde t$$','interpreter','latex','Fontsize',15);
text(20,1e-11,'a=1','Fontsize',20);
text(3*1e4,1e-3,'(b)','Fontsize',20);
subplot(2,2,3);
loglog(N_sam,t_Gau_a2(:,1),'--b',N_sam,t_Gau_a2(:,2),'-.k',N_sam,t_Gau_a2_mean,'-r','LineWidth',lw,'MarkerSize',msz);
% set(gca,'XLim',[10 1e5]);
% set(gca,'YLim',[min(min(log10(t_Gau_a05))) max(max(log10(t_Gau_a05)))]);
xlabel('$K$','interpreter','latex');
ylabel('$$\tilde t$$','interpreter','latex','Fontsize',15);
text(20,3*1e-18,'a=2','Fontsize',20);
text(3*1e4,3*1e-3,'(c)','Fontsize',20);
subplot(2,2,4);
loglog(N_sam,t_Gau_a4(:,1),'--b',N_sam,t_Gau_a4(:,2),'-.k',N_sam,t_Gau_a4_mean,'-r','LineWidth',lw,'MarkerSize',msz);
% set(gca,'XLim',[10 1e5]);
% set(gca,'YLim',[min(min(log10(t_Gau_a05))) max(max(log10(t_Gau_a05)))]);
xlabel('$K$','interpreter','latex');
ylabel('$$\tilde t$$','interpreter','latex','Fontsize',15);
text(20,1e-70,'a=4','Fontsize',20);
text(3*1e4,1e-10,'(d)','Fontsize',20);
% % save t_vs_N-2_20170924;

