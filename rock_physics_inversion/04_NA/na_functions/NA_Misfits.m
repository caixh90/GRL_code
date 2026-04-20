function [mopt,mfitmin,mfitord,work] = ...
    NA_Misfits(misfit,nsample,it,ntot,mfitmin,mopt,ncells)


mfitminc = misfit(ntot+1);
mfitmean = mfitminc;
iopt = ntot+1;

for i=ntot+2:ntot+nsample
	mfitmean = mfitmean + misfit(i);
	if misfit(i) < mfitminc
    	mfitminc = misfit(i);
        iopt = i;
	end 
end
%mfitmean = mfitmean/nsample;

if it == 1 || mfitminc < mfitmin  
	mopt = iopt;
    mfitmin = mfitminc;
end

% find models with lowest ncells misfit values
if ncells == 1
	mfitord(1) = mopt;
else 
	ntotal = ntot+nsample;
    work = NaN(1,ntotal);
	for i=1:ntotal
        work(i) = misfit(i);
	end
   
    % Strip to ncell values
    [~,index] = sort(work);    
    mfitord = index(1:ncells);
    
	% jumble initial indices to randomize order of models when misfits are equal
    %[ind,work] = jumble(ind,work,ntotal);
    
	%flow = na_select(ncells,ntotal,work,ind,iselect);

	%for j=1:ncells
	%	iwork(j) = ind(j);
	%end 
	
	% order misfit of lowest ncells
    %call indexx(ncells,work,ind)
	%[~,ind] = sort(work); 
    
	%for j=1:ncells
	%	mfitord(j) = iwork(ind(j));
	%end
    
end