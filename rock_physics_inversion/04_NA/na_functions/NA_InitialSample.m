function na_models = NA_InitialSample(nd,nsample,range)

na_models = NaN(1,nsample*nd);
count = 0;
for i = 1:nsample
    for j = 1:nd
        count = count+1;
        
        a = rand(1,1);
        b = 1-a;
        
        na_models(count) = b*range(1,j) + a*range(2,j);
    end
end