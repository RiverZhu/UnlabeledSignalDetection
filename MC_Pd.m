function [ P_d ] = MC_Pd( T_0,T_1,MC,P_fa )
    T_0_sort = sort(T_0,'descend') ;
    gamma = T_0_sort(fix(MC*P_fa));
    P_d = length(find(T_1>gamma))/MC; 
end

