function [ranget,scales,x,ic,restartNA] = NA_Initialize(rng,scales,nd)

ranget = NaN(2,nd);

restartNA = true;

ic = 1;

% Normalize parameter ranges by a-priori model co-variances
if scales(1) == 0
	% First option: No transform (All a priori model co-variances are equal to unity)
	for i=1:nd
		ranget(1,i) = rng(1,i);
        ranget(2,i) = rng(2,i);
        scales(i+1) = 1.0;
	end

elseif scales(1) == -1 % default
	% Second option: Use parameter range as a priori model co-variances 
    for i=1:nd
    	ranget(1,i) = 0.0;
        ranget(2,i) = 1.0;
        scales(i+1) = rng(2,i)-rng(1,i);
    end
else
	% Third option: Use scales array as a priori model co-variances 
	for i=1:nd
        if scales(i+1) == 0
        	error('Error in NA_Initialize');
        end
        ranget(1,i)  = 0.0;
        ranget(2,i)  = (rng(2,i)-rng(1,i))/scales(i+1);
	end
end

% calculate axis increments and initialize current point (used by NA_sample) to mid-point of parameter space
x = NaN(1,nd);
for i=1:nd
	x(i) = (ranget(2,i)+ranget(1,i))/2.0;
end
