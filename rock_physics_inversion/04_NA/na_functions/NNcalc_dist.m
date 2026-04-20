function [nodex,dlist] = NNcalc_dist(dim,dlist,bp,nd,nb,x)




dmin = 0.0;
for j=1:dim-1
    d = (x(j)-bp(j,1));
    d = d*d;
    dmin = dmin + d;
end
for j=dim+1:nd
    d = (x(j)-bp(j,1));
    d = d*d;
    dmin = dmin + d;
end
dlist(1) = dmin;
d = (x(dim)-bp(dim,1));
d = d*d;
dmin = dmin + d;
nodex = 1;

for i=2:nb
    dsum = 0.0;
    for j=1:dim-1
        d = (x(j)-bp(j,i));
        d = d*d;
        dsum = dsum + d;
    end 
    for j=dim+1:nd
        d = (x(j)-bp(j,i));
        d = d*d;
        dsum = dsum + d;
    end
    dlist(i) = dsum;
    d = (x(dim)-bp(dim,i));
    d = d*d;
    dsum = dsum + d;
    if dmin > dsum
        dmin = dsum;
        nodex = i;
    end
    %dnodex = dmin;
end






