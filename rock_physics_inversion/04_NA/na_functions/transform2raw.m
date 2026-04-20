function model_raw = transform2raw(model_sca,nd,range,scales)

model_raw = model_sca;

if scales(1) == 0
	for i=1:nd
		model_raw(i) = model_sca(i);
	end
elseif scales(1) == -1
	for i=1:nd
		b = model_sca(i);
		a = 1-b;
		model_raw(i) = a*range(1,i) + b*range(2,i);
	end
else
	for i=1:nd
		model_raw(i) = range(1,i) + scales(i+1)*model_sca(i);
	end
end

