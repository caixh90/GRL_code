function f = obj_stage2(m,dt,por,qu,vcl,Peff, weights)

        m2 = [por qu vcl Peff m];
        dp = model_sand_CMC_vector(m2);
        diff = dp - dt;
        f = 0.5 * sum(weights.* (abs(diff).^2));  

end