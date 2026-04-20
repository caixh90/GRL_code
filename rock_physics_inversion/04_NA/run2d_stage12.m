% Run 2D deterministic inversion using the neighborhood algorithm (trace by
% trace). Stage 1: find por, qu, vcl that fit baseline data. Stage 2: map
% the vpvsrho difference to sc.

% stage 1: [vp_b, vs_b, rho_b]=model_sand_CMC(por,qu,vcl,sc=0)
% stage 2: [vp_m, vs_m, rho_m]=model_sand_CMC(por,qu,vcl,sc)

clc;clear;close all;
addpath(genpath('../../rock_physics_inversion_2026'));
load Logs_qi_0_350_smooth.mat
load vpvsrho_base_moni vp_b vs_b rho_b vp_m vs_m rho_m xx zz

depth2 = zz';

%%
vpt = interp1(depth,vpt,depth2,'linear');
vst = interp1(depth,vst,depth2,'linear');
rhot = interp1(depth,rhot,depth2,'linear');
port = interp1(depth,port,depth2,'linear');
qut = interp1(depth,qut,depth2,'linear');
vclt = interp1(depth,vclt,depth2,'linear');
Pefft = interp1(depth,Pefft,depth2,'linear');

sct = zeros(size(port));
depth = depth2;

%%
[nz,nx] = size(vp_b);

%% Parameters for the Neighborhood Algorithm

nd1 = 3;
lb1 = [0 0 0];
ub1 = [0.4 1 1];

nd2 = 1;
lb2 = 0;
ub2 = 1;

itmax = 20;
nsamplei = 1000;
nsample = 100;
ncells = 10;
mfbreak = 1e-20;

weight = [1 0 1]; % weights of vpvsrho in data fitting

%% Stage 1: invert for por, qu, vcl (baseline)
por = zeros(nz, nx);
qu = zeros(nz, nx);
vcl = zeros(nz, nx);

tic
for j = 1:nx
    data_base = [vp_b(:,j) vs_b(:,j) rho_b(:,j)];
    parfor i = 1:nz
        rng(42 + i + j*nz); % rand seed
        ObjFunc = @(m)obj_stage1(m, data_base(i,:), Pefft(i), sct(i), weight);
        [model_opt,~,~,~] = NA_point(ObjFunc, nd1, lb1, ub1, itmax, nsamplei, nsample, ncells, mfbreak);
        por(i,j) = model_opt(1);
        qu(i,j) = model_opt(2);
        vcl(i,j) = model_opt(3);
    end
    disp(['Stage 1 trace ', num2str(j), '/', num2str(nx)]);
end
toc

%% Stage 2: invert for sc (monitoring)
sc = zeros(nz, nx);

tic
for j = 1:nx
    data_moni = [vp_m(:,j) vs_m(:,j) rho_m(:,j)];
    parfor i = 1:nz
        rng(42 + i + j*nz); % rand seed
        ObjFunc = @(m)obj_stage2(m, data_moni(i,:), por(i,j), qu(i,j), vcl(i,j), Pefft(i), weight);
        [model_opt,~,~,~] = NA_point(ObjFunc, nd2, lb2, ub2, itmax, nsamplei, nsample, ncells, mfbreak);
        sc(i,j) = model_opt(1);
    end
    disp(['Stage 2 trace ', num2str(j), '/', num2str(nx)]);
end
toc

%% Plot 2D results
% figure('Name', 'Stage 1');
% set(gcf, 'Position', [100 100 600 900])
% subplot(311); imagesc(xx, zz, por); colorbar; title('Porosity'); ylabel('Depth (m)'); xlabel('Offset (m)'); 
% subplot(312); imagesc(xx, zz, qu);  colorbar; title('Vquartz'); ylabel('Depth (m)'); xlabel('Offset (m)'); 
% subplot(313); imagesc(xx, zz, vcl); colorbar; title('Vclay');  ylabel('Depth (m)'); xlabel('Offset (m)'); 
% colormap(jet); set(gca,'FontSize',12);
% 
% figure('Name', 'Stage 2');
% imagesc(xx, zz, sc); colorbar; title('CO2 saturation'); ylabel('Depth (m)'); xlabel('Offset (m)');
% set(gca,'FontSize',12);
% colormap(jet); 
% clim([0 0.23]);

save fig4 sc 
