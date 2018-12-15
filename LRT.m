function [P_d_a, P_d_b, P_d_c, P_d_d] = LRT(N,K,h,wavr,tau,theta,MC,P_fa,q_0,q_1,T,Perm)
options = optimset('Largescale','off','GradObj','on','Hessian','off',...
            'MaxFunEvals',2000,'MaxIter',1000,'Display','off','DerivativeCheck','off'); 
    T_0_a = T;
    T_1_a = T;
    T_0_b = T;
    T_1_b = T;
    T_0_c = T;
    T_1_c = T;
    T_0_d = T;
    T_1_d = T;
    A = eye(N);
    epsilon = 1e-1;
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
        theta_0 = fminunc(@(theta_0)Fun_Q_T(eta_0,h,theta_0,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        theta_1 = fminunc(@(theta_1)Fun_Q_T(eta_1,h,theta_1,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
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
        
        Pi_num_0_est = Pi_eta_p_0*Pi_num'; % H0 estimated permutation
        Pi_den_0_est = Pi_eta_p_0*Pi_den';
        Pi_num_1_est = Pi_eta_p_1*Pi_num'; % H1 estimated permutation
        Pi_den_1_est = Pi_eta_p_1*Pi_den';
        
        l_num_0 = (Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_0 = (Pi_den_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-Pi_den_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        l_num_1 = (Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta-tau)/sqrt(wavr))));
        l_den_1 = (Pi_den_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr)))+(1-Pi_den_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((       -tau)/sqrt(wavr))));
        T_0_c(mc) = l_num_0-l_den_0;
        T_1_c(mc) = l_num_1-l_den_1;  
       %% MLE  theta / Pi unknown (case d)
        theta_new = 1.1; % estimate theta and Pi under H1
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); % sort h*theta-tau
            Pi_num = A(num_index, :); % sorting permutation    
            Pi_num_1_est = Pi_eta_p_1*Pi_num'; 
            theta_new = fminunc(@(theta_d)Fun_Q_T(Pi_num_1_est'*eta_p_1,h,theta_d,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_1 = theta_new;
        theta_new = 0.1;  % estimate theta and Pi under H0
        theta_old = theta_new+1;
        while (abs(theta_old-theta_new)>epsilon)
            theta_old = theta_new;
            [~,num_index]=sort(h*theta_old-tau); % sort h*theta-tau
            Pi_num = A(num_index, :); % sorting permutation    
            Pi_num_0_est = Pi_eta_p_0*Pi_num'; 
            theta_new = fminunc(@(theta_d)Fun_Q_T(Pi_num_0_est'*eta_p_0,h,theta_d,tau,N,K,q_0,q_1,sqrt(wavr)),0,options);
        end
        theta_0 = theta_new;
        l_num_0 = (Pi_num_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_0-tau)/sqrt(wavr)))+(1-Pi_num_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_0-tau)/sqrt(wavr))));
        l_den_0 = (Pi_den_0_est'*eta_p_0)'*log(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr)))+(1-Pi_den_0_est'*eta_p_0)'*log(1-(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr))));
        l_num_1 = (Pi_num_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((h*theta_1-tau)/sqrt(wavr)))+(1-Pi_num_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((h*theta_1-tau)/sqrt(wavr))));
        l_den_1 = (Pi_den_1_est'*eta_p_1)'*log(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr)))+(1-Pi_den_1_est'*eta_p_1)'*log(1-(q_0+(1-q_0-q_1)*normcdf((         -tau)/sqrt(wavr))));
        T_0_d(mc) = l_num_0-l_den_0;
        T_1_d(mc) = l_num_1-l_den_1; 
    end
    T_0_a_sort = sort(T_0_a,'descend') ;
    T_0_b_sort = sort(T_0_b,'descend') ;
    T_0_c_sort = sort(T_0_c,'descend') ;
    T_0_d_sort = sort(T_0_d,'descend') ;
    gamma_a = T_0_a_sort(fix(MC*P_fa));
    gamma_b = T_0_b_sort(fix(MC*P_fa));
    gamma_c = T_0_c_sort(fix(MC*P_fa));
    gamma_d = T_0_d_sort(fix(MC*P_fa));
    P_d_a = length(find(T_1_a>gamma_a))/MC; 
    P_d_b = length(find(T_1_a>gamma_b))/MC; 
    P_d_c = length(find(T_1_a>gamma_c))/MC; 
    P_d_d = length(find(T_1_a>gamma_d))/MC; 
end

