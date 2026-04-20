%% This code is used to examine the FWI results from Cai

clc;clear;close all

load snowflake_2026.mat

dz=5;dx=5;

zz=0:5:350;

xclip = 300;
xx=-xclip:dx:xclip;
index_x = round((xx + 500) / dx) + 1;

vp_b = Basevp(:,index_x);
vs_b = Basevs(:,index_x);
rho_b = Baserho(:,index_x); 
vp_m = Monvp(:,index_x);
vs_m = Monvs(:,index_x);
rho_m = Monrho(:,index_x);

%% lateral smoothing 
model=[vp_b(:); vs_b(:); rho_b(:); vp_m(:); vs_m(:); rho_m(:)];
[nz, nx] = size(vp_b);
Param_num = 6;
width = 5;
model_smooth = func_smooth2D(model,nz,nx,Param_num,width);
vp_b=model_smooth(:,:,1);
vs_b=model_smooth(:,:,2);
rho_b=model_smooth(:,:,3);
vp_m=model_smooth(:,:,4);
vs_m=model_smooth(:,:,5);
rho_m=model_smooth(:,:,6);

%%

vp_all=vp_m; 
vs_all=vs_m;
rho_all=rho_m;
diff_vp=vp_m-vp_b; diff_vs=vs_m-vs_b; diff_rho=rho_m-rho_b; 


fig=figure('Name','fig');
set(fig,'position',[100,100,1300,700])
subplot(331)
set(gca,'position',[0.05,0.72,0.24,0.23])
imagesc(xx,zz,vp_b);
clim([min(vp_all(:)) max(vp_all(:))]);
ylabel('Depth (m)')
title('Base $V_{\rm P}$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.3 0.72 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(332)
set(gca,'position',[0.37,0.72,0.24,0.23])
imagesc(xx,zz,vp_m);
clim([min(vp_all(:)) max(vp_all(:))]);
yticks([])
title('Monitor $V_{\rm P}$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.62 0.72 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(333)
set(gca,'position',[0.69,0.72,0.24,0.23])
imagesc(xx,zz,diff_vp);
clim([-max(abs(diff_vp(:))) max(abs(diff_vp(:)))]);
yticks([])
title('$\Delta V_{\rm P}$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.94 0.72 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(334)
set(gca,'position',[0.05,0.4,0.24,0.23])
imagesc(xx,zz,vs_b);
clim([min(vs_all(:)) max(vs_all(:))]);
xticks([])
ylabel('Depth (m)')
title('Base $V_{\rm S}$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.3 0.4 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(335)
set(gca,'position',[0.37,0.4,0.24,0.23])
imagesc(xx,zz,vs_m);
clim([min(vs_all(:)) max(vs_all(:))]);
xticks([])
yticks([])
title('Monitor $V_{\rm S}$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.62 0.4 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(336)
set(gca,'position',[0.69,0.4,0.24,0.23])
imagesc(xx,zz,diff_vs);
% clim([-max(abs(diff_vs(:))) max(abs(diff_vs(:)))]);
clim([-100 100]);
xticks([])
yticks([])
title('$\Delta V_{\rm S}$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.94 0.4 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(337)
set(gca,'position',[0.05,0.08,0.24,0.23])
imagesc(xx,zz,rho_b);
clim([min(rho_all(:)) max(rho_all(:))]);
xlabel('Position (m)'); 
ylabel('Depth (m)')
title('Base $\rho$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.3 0.08 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(338)
set(gca,'position',[0.37,0.08,0.24,0.23])
imagesc(xx,zz,rho_m);
clim([min(rho_all(:)) max(rho_all(:))]);
xlabel('Position (m)'); 
yticks([])
title('Monitor $\rho$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.62 0.08 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')

subplot(339)
set(gca,'position',[0.69,0.08,0.24,0.23])
imagesc(xx,zz,diff_rho);
% clim([-max(abs(diff_rho(:))) max(abs(diff_rho(:)))]);
clim([-100 100]);
xlabel('Position (m)'); 
yticks([])
title('$\Delta \rho$','Interpreter','LaTex')
h=colorbar;
set(h,'position',[0.94 0.08 0.01 0.23]);
set(gca,'FontSize',14,'fontname','arial','fontweight','normal')
colormap(jet)

save vpvsrho_base_moni vp_b vs_b rho_b vp_m vs_m rho_m xx zz



