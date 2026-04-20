% Run 1D test first to tune the inversion parameters. The 2D inversion is
% multiple 1D inversions that differ only in the input vpvsrho (per trace).

%
clc;clear;close all;
addpath(genpath('../../rock_physics_inversion_simple'));

load Logs_qi_0_350_smooth.mat
load vpvsrho_base_moni.mat vp_b vs_b rho_b xx zz

depth2 = zz';

%%

[nz,nx]=size(vp_b);

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
[vp,vs,rho]=model_sand_CMC(port,qut,vclt,Pefft, 0);

%% training dataset
mtrain = [port qut vclt];
dtrain = [vp vs rho];

%%

% domain to evaluate the posterior PDF
phidomain = (0.005:0.005:0.3);   
qudomain = (0:0.01:1); 
vcldomain = (0:0.01:1);
[P,V,S] = ndgrid(phidomain, qudomain, vcldomain);
mdomain = [P(:) V(:) S(:)];

%% Error covariance matrix
ratio = 0.2;
sigmaerr = diag([ratio*mean(vpt),ratio*mean(vst),ratio*mean(rhot)]);
% sigmaerr = diag([var(vp-vp_inv),var(vs-vs_inv),var(rho-rho_inv)]);

%% Inversion
por2D = zeros(nz, nx);
qu2D = zeros(nz, nx);
vcl2D = zeros(nz, nx);

tic
parfor j = 1:nx

    dcond = [vp_b(:,j) vs_b(:,j) rho_b(:,j)];
    
    % Gaussian nonlinear case
    [mupost, sigmapost, Ppost] = RockPhysicsGaussInversion(mtrain, dtrain, mdomain, dcond, sigmaerr);
    
    % marginal posterior distributions
    Ppostphi = zeros(nz, length(phidomain));
    Ppostqu = zeros(nz, length(qudomain));
    Ppostvcl = zeros(nz, length(vcldomain));
    
    por_col = zeros(nz, 1);
    qu_col = zeros(nz, 1);
    vcl_col = zeros(nz, 1);
    
    for i = 1:nz
        Ppostjoint = reshape(Ppost(i,:), length(phidomain), length(qudomain), length(vcldomain));
        Ppostphi(i,:) = sum(squeeze(sum(squeeze(Ppostjoint), 3)), 2);
        Ppostqu(i,:) = sum(squeeze(sum(squeeze(Ppostjoint), 3)), 1);
        Ppostvcl(i,:) = sum(squeeze(sum(squeeze(Ppostjoint), 2)), 1);
        
        Ppostphi(i,:) = Ppostphi(i,:) / sum(Ppostphi(i,:));
        Ppostqu(i,:) = Ppostqu(i,:) / sum(Ppostqu(i,:));
        Ppostvcl(i,:) = Ppostvcl(i,:) / sum(Ppostvcl(i,:));
        
        [~, Phimapind] = max(Ppostphi(i,:));
        [~, Qumapind] = max(Ppostqu(i,:));
        [~, Vclmapind] = max(Ppostvcl(i,:));
        
        por_col(i) = phidomain(Phimapind);
        qu_col(i) = qudomain(Qumapind);
        vcl_col(i) = vcldomain(Vclmapind);
    end
    
    por2D(:, j) = por_col;
    qu2D(:, j) = qu_col;
    vcl2D(:, j) = vcl_col;

    disp(j)
end
toc

%%
fig=figure('Name','fig');
set(fig,'position',[100,100,1200,500])
subplot(221)
imagesc(xx,zz,por2D);title('Porosity');colorbar;
set(gca,'FontSize',12,'fontname','arial','fontweight','normal')

subplot(222)
imagesc(xx,zz,qu2D);title('Quartz');colorbar;
set(gca,'FontSize',12,'fontname','arial','fontweight','normal')

subplot(223)
imagesc(xx,zz,vcl2D);title('Clay');colorbar;
set(gca,'FontSize',12,'fontname','arial','fontweight','normal')
colormap(jet)

co2D=1-qu2D-vcl2D;
co2D(co2D<0)=0;
subplot(224)
imagesc(xx,zz,co2D);title('Coal');colorbar;
set(gca,'FontSize',12,'fontname','arial','fontweight','normal')
colormap(jet)

%%
save inverted_porquvclco por2D qu2D vcl2D co2D 


