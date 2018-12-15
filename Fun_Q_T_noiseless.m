function [neg_log_like,gradient_theta] = Fun_Q_T_noiseless(Eta,h,theta,tau,N,K,q0,q1,sigma_w) 

    probit_F = normcdf((h*theta-tau)*(1/sigma_w));
    neg_log_like = -K*((Eta'* log(q0*ones(N,1)+(1-q0-q1)*probit_F) + (ones(N,1)-Eta)'* log((1-q0)*ones(N,1)-(1-q0-q1)*probit_F)));
    
    probit_f = h.*normpdf((h*theta-tau)/sigma_w)/sigma_w;
    gradient_theta = -K*(1-q0-q1)*( Eta./(q0*ones(N,1)+(1-q0-q1)*probit_F)-(ones(N,1)-Eta)./((1-q0)*ones(N,1)-(1-q0-q1)*probit_F) )'*probit_f; % derivative based on -loglike
    
end