function [na_models1D,xcur,ic] = NA_Sample(na_models1D,ntot,nsample,nd,nsleep,ncells,...
        mfitord,range,xcur,restartNA,nclean,dlist,ic)
    
    
% Convert 1D na_models into 2D
na_models = NaN(nd,ntot);
count = 0;
for i = 1:ntot
    for j = 1:nd
        count = count+1;
        na_models(j,i) = na_models1D(count);
    end
end


% Choose axis randomly
idnext = randi(nd);

ic = ic + 1;

if mod(ic,nclean) == 0
    resetlist = true;
end
id = 0;
cell = 1;
mopt = mfitord(cell);

ind_cellnext = mopt;
ind_celllast = 0;
nrem = mod(nsample,ncells);
if nrem == 0
	nsampercell = floor(nsample/ncells);
else
    nsampercell = floor(1+nsample/ncells);
end


icount = 0;

for is = 1:nsample
    % Choose Voronoi cell for sampling
    ind_cell = ind_cellnext;
    icount = icount + 1;
    if ind_cell ~= ind_celllast
        %call NA_restart(na_models,nd,ind_cell,xcur,restartNA)
        %NA_restart(na_models,nd,mreset,x,restartNA)
        
        [xcur,restartNA] = NA_Restart(na_models,nd,ind_cell);
    end
    
    if restartNA
        resetlist = true;
        restartNA = false;
    end
    
    % Loop over walk steps
    for il = 1:nsleep
        for iw = 1:nd
            
            % Update dlist and nodex for new axis
            if ~resetlist
                % Incremental update
                % call NNupdate_dlist(idnext,id,dlist,na_models,nd,ntot,xcur,nodex,dminx)
                %NNupdate_dlist(dim,dimlast,dlist,bp,nd,nb,x,node,dmin)
                [nodex,dlist] = NNupdate_dlist(idnext,id,na_models,...
                    ntot,xcur,dlist);
                
            else
                % Full update
                % call NNcalc_dlist(idnext,dlist,na_models,nd,ntot,xcur,nodex,dminx)
                % NNcalc_dlist(dim,dlist,bp,nd,nb,x,nodex,dminx)
                [nodex,dlist] = NNcalc_dist(idnext,dlist,na_models,nd,ntot,xcur);
            end
            
            id = idnext;
            
            % Calculate intersection of current Voronoi cell with current 1-D axis
    		%call NNaxis_intersect(xcur,id,dlist,na_models,nd,ntot,nodex,range(1,id),range(2,id), &
    		%x1,x2)
            % NNaxis_intersect(x,dim,dlist,bp,nd,nb,nodex,xmin,xmax,x1,x2)
            [x1,x2] = NNaxis_intersect(id,dlist,na_models,ntot,...
                nodex,range(1,id),range(2,id));
            
            
            
            
            
            
            
            % Generate new node in Voronoi cell of input point
    		kd = id + (cell-1)*nd;
    		%call NA_deviate (x1,x2,kd,xcur(id))
			xcur(id) = NA_deviate(x1,x2,kd);
            
            % increment axis 
    		idnext = idnext + 1;
            if idnext > nd
                idnext=1;
            end
        end
    end
	
	% put new sample in list
    j = ntot+is;
    for i=1:nd
    	na_models(i,j) = xcur(i);
    end
	ind_celllast = ind_cell;
    
    % Only do this step if not at the end
    if is < nsample
        if icount == nsampercell
            icount = 0; 
            cell = cell + 1;
            ind_cellnext = mfitord(cell);
            if cell == nrem+1
                nsampercell = nsampercell - 1;
            end
        end
    end
end

% Bug fix - convert na_models back into 1-dimensional array
count=0;
for i=1:size(na_models,2)
	for j=1:nd
		count=count+1;
		na_models1D(count)=na_models(j,i);
	end
end
            
            
            




