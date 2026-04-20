function f = obj_stage1(m, dt, Peff, sc, weights)

        m2 = [m Peff sc];
        dp = model_sand_CMC_vector(m2);
        diff = dp - dt;
        f = 0.5 * sum(weights.* (abs(diff).^2));  

end