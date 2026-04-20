function [x1,x2] = NNaxis_intersect(dim,dlist,bp,nb,nodex,xmin,xmax)


% search through nodes
x1 = xmin;
x2 = xmax;
dp0   = dlist(nodex);
x0    = bp(dim,nodex);

% find intersection of current Voronoi cell with 1-D axis
for j=1:nodex-1
	xc    = bp(dim,j);
    dpc   = dlist(j);
	% calculate intersection of interface (between nodes nodex and j) and 1-D axis.
    dx = x0 - xc;
    if dx ~= 0.0
    	xi = 0.5*(x0+xc+(dp0-dpc)/dx);
        if xi > xmin && xi < xmax
            if xi > x1 && x0 > xc
            	x1 = xi;
            elseif xi < x2 && x0 < xc
                x2 = xi;
            end
        end
    end
end

for j=nodex+1:nb
	xc    = bp(dim,j);
	dpc   = dlist(j);
	% calculate intersection of interface (between nodes nodex and j) and 1-D axis.
    dx = x0 - xc;
    if dx ~= 0
        xi = 0.5*(x0+xc+(dp0-dpc)/dx);
        if xi > xmin && xi < xmax
            if xi > x1 && x0 > xc
                x1 = xi;
            elseif xi < x2 && x0 < xc
                x2 = xi;
            end
        end
    end
end