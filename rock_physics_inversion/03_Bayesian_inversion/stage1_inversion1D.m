%
% 1D Bayesian inversion: recover porosity and mineral logs (quartz,
% clay fractions) from FWI-derived Vp, Vs, and density at the well location.
%
% Note: (1) Input data (Vp, Vs, rho) are noisy.
%       (2) Forward model (rock-physics relation) is imperfect
% so the results better reflect some uncertainties. 


addpath(genpath('../../rock_physics_inversion_2026'));

%% 
load Logs_qi_0_350_smooth.mat
load vpvsrho_base_moni.mat vp_b vs_b rho_b xx zz

depth2 = zz';

%% 
vpt = interp1(depth,vpt,depth2,'linear');
vst = interp1(depth,vst,depth2,'linear');
rhot = interp1(depth,rhot,depth2,'linear');
port = interp1(depth,port,depth2,'linear');
qut = interp1(depth,qut,depth2,'linear');
vclt = interp1(depth,vclt,depth2,'linear');
Pefft = interp1(depth,Pefft,depth2,'linear');

depth=depth2;

%% 
[vp,vs,rho]=model_sand_CMC(port,qut,vclt,Pefft,0);

%%
fig=figure('Name','fig');
set(fig,'position',[100,100,1500,700])
subplot(171)
plot(vpt,depth,'LineWidth', 2);set(gca,'YDir','reverse');
hold on
plot(vp,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Vp')
subplot(172)
plot(vst,depth,'LineWidth', 2);set(gca,'YDir','reverse');
hold on
plot(vs,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Vs')
subplot(173)
plot(rhot,depth,'LineWidth', 2);set(gca,'YDir','reverse');
hold on
plot(rho,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Rho')
subplot(174)
plot(port,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Porosity')
subplot(175)
plot(qut,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Vquartz')
subplot(176)
plot(vclt,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Vclay')
subplot(177)
plot(Pefft,depth,'LineWidth', 2);set(gca,'YDir','reverse');xlabel('Peff')
set(gca,'YDir','reverse');colormap('parula')

%% training dataset
mtrain = [port qut vclt];
dtrain = [vp vs rho];

%%
shift = 8;
vp_inv=vp_b(:,ceil(end/2)+shift); vs_inv=vs_b(:,ceil(end/2)+shift); rho_inv=rho_b(:,ceil(end/2)+shift);

%%
dcond = [vp_inv vs_inv rho_inv];
ns = size(dcond,1);
% 
fig=figure('Name','fig');
set(fig,'position',[100,100,1000,500])
subplot(131)
plot(vpt,depth, 'LineWidth', 2);set(gca,'YDir','reverse');
hold on
plot(vp_inv,depth, 'LineWidth', 2);
ylabel('depth (m)')
xlabel('Vp')
set(gca,'FontSize',13,'fontname','arial','fontweight','normal')
subplot(132)
plot(vst,depth, 'LineWidth', 2);set(gca,'YDir','reverse');
hold on
plot(vs_inv,depth, 'LineWidth', 2);
xlabel('Vs')
h=legend({'Actual-logs','FWI'},'Location','northoutside','orientation','horizontal','FontSize',18);
set(h,'position',[0.5 0.6 0.02 0.75]); legend('boxoff')
set(gca,'FontSize',13,'fontname','arial','fontweight','normal')
subplot(133)
plot(rhot,depth, 'LineWidth', 2);set(gca,'YDir','reverse');
hold on
plot(rho_inv,depth, 'LineWidth', 2);
xlabel('Density')
set(gca,'FontSize',13,'fontname','arial','fontweight','normal')

%%
% domain to evaluate the posterior PDF
phidomain = (0.1:0.002:0.3);   
qudomain = (0:0.01:1); 
vcldomain = (0:0.01:1);
[P,V,S] = ndgrid(phidomain, qudomain, vcldomain);
mdomain = [P(:) V(:) S(:)];

%% Error covariance matrix
ratio = 0.2;
sigmaerr = diag([ratio*mean(vpt),ratio*mean(vst),ratio*mean(rhot)]);
% sigmaerr = diag([var(vp-vp_inv),var(vs-vs_inv),var(rho-rho_inv)]);

%% Inversion
% Gaussian nonlinear case
[mupost, sigmapost, Ppost]  = RockPhysicsGaussInversion(mtrain, dtrain, mdomain, dcond, sigmaerr);

% marginal posterior distributions
Ppostphi = zeros(ns,length(phidomain));
Ppostqu = zeros(ns,length(qudomain));
Ppostvcl = zeros(ns,length(vcldomain));

Phimapind = zeros(ns,1);
Qumapind = zeros(ns,1);
Vclmapind = zeros(ns,1);

Phimap = zeros(ns,1);
Qumap = zeros(ns,1);
Vclmap = zeros(ns,1);

tic
parfor i=1:ns
    Ppostjoint=reshape(Ppost(i,:),length(phidomain),length(qudomain),length(vcldomain));
    Ppostphi(i,:)=sum(squeeze(sum(squeeze(Ppostjoint),3)),2);
    Ppostqu(i,:)=sum(squeeze(sum(squeeze(Ppostjoint),3)),1);
    Ppostvcl(i,:)=sum(squeeze(sum(squeeze(Ppostjoint),2)),1);

    Ppostphi(i,:)=Ppostphi(i,:)/sum(Ppostphi(i,:));
    Ppostqu(i,:)=Ppostqu(i,:)/sum(Ppostqu(i,:));
    Ppostvcl(i,:)=Ppostvcl(i,:)/sum(Ppostvcl(i,:));
    [~,Phimapind(i)]=max(Ppostphi(i,:));
    [~,Qumapind(i)]=max(Ppostqu(i,:));
    [~,Vclmapind(i)]=max(Ppostvcl(i,:));
    Phimap(i)=phidomain(Phimapind(i));
    Qumap(i)=qudomain(Qumapind(i));
    Vclmap(i)=vcldomain(Vclmapind(i));
end
toc

%% 
fig=figure('Name','fig');
set(fig,'position',[100,100,1200,650])
subplot(131)
set(gca,'position',[0.08,0.1,0.25,0.85])
pcolor(phidomain, depth, Ppostphi); 
hold on; 
shading interp;
colorbar; 
plot(port, depth, 'k', 'LineWidth', 2);  
ylabel('Depth (m)'); xlabel('Porosity'); 
xlim([0.1 0.25])
% plot(Phimap, depth, 'r', 'LineWidth', 2);
% plot(mupost(:,1), depth, 'b', 'LineWidth', 2);
set(gca,'YDir','reverse');
set(gca,'FontSize',13,'fontname','arial','fontweight','normal')
set(gca,'layer','top')

subplot(132)
set(gca,'position',[0.4,0.1,0.25,0.85])
pcolor(qudomain, depth, Ppostqu); 
hold on; shading interp; colorbar; 
plot(qut, depth, 'k', 'LineWidth', 2); 
xlabel('Vquartz'); 
xlim([0 0.9])
% plot(Qumap, depth, 'r', 'LineWidth', 2);
% plot(mupost(:,2), depth, 'b', 'LineWidth', 2);
set(gca,'YDir','reverse');
set(gca,'FontSize',13,'fontname','arial','fontweight','normal')
set(gca,'layer','top')

subplot(133)
set(gca,'position',[0.72,0.1,0.25,0.85])
pcolor(vcldomain, depth, Ppostvcl); 
hold on; 
shading interp; 
colorbar; 
plot(vclt, depth, 'k', 'LineWidth', 2); 
% plot(Vclmap, depth, 'r', 'LineWidth', 2);
% plot(mupost(:,3), depth, 'b', 'LineWidth', 2);
xlabel('Vclay');
xlim([0.1 1])
hbc=colorbar; title(hbc, 'Probability');
set(gca,'YDir','reverse');
set(gca,'FontSize',13,'fontname','arial','fontweight','normal')
set(gca,'layer','top')

save fig4f-h phidomain Ppostphi qudomain Ppostqu vcldomain Ppostvcl port qut vclt;
colormap(slanCM(2))

