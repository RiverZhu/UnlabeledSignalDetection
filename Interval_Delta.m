function [ theta_projection ] = Interval_Delta( theta, Delta )
        theta_projection = theta;
        if theta_projection > Delta
            theta_projection = Delta;
        end
        if theta_projection < -Delta
            theta_projection = -Delta;
        end
end

