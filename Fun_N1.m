function [ obj_value ] = Fun_N1( alpha,h,tau,theta,r)
obj_value = (sum(1./(1+exp(-2*alpha*(h*theta-tau))))-r)^2; % object function
end

