function [x,restartNA] = NA_Restart(na_models,nd,mreset)

x = NaN(1,nd);
for i = 1:nd
    x(i) = na_models(i,mreset);
end

restartNA = true;

