function [node,dlist] = NNupdate_dlist(dim,dimlast,bp,nb,x,dlist)
                

d1 = (x(dimlast)-bp(dimlast,1));
d1 = d1*d1;
dmin = dlist(1)+d1;
node = 1;
d2 = (x(dim)-bp(dim,1));
d2 = d2*d2;
dlist(1) = dmin-d2;
for i=2:nb
    d1 = (x(dimlast)-bp(dimlast,i));
    ds = d1;
    d1 = dlist(i)+d1*d1;
    if dmin > d1
        dmin = d1;
        node = i;
    end
    d2 = (x(dim)-bp(dim,i));
    d2 = d2*d2;
    dlist(i) = d1-d2;
end