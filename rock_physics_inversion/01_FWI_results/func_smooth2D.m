function model_smooth = func_smooth2D(model,nz,nx, Param_num, width)

model = reshape(model,nz,nx,Param_num);
xvals = -2*width:1:2*width;
Gauss = exp(-((xvals./width).^2));
Gauss = Gauss./sum(Gauss);
mid = ceil(length(Gauss)/2);
scale = convzs(ones(1,nx),Gauss);
C = toeplitz([Gauss(mid:end),zeros(1,nx-mid)],[Gauss(mid:end),zeros(1,nx-mid)]);
C = C./scale';

model_smooth = zeros(size(model));
for m = 1:Param_num
    for n = 1:nz
        model_smooth(n,:,m) = C*squeeze(model(n,:,m))';
    end
end