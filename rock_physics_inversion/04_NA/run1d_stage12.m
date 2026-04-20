% Run a deterministic inversion using the neighborhood algorithm. The goal
% is to find por, qu, vcl that fit the baseline data, so the vpvsrho
% difference can be mapped to sc in Stage 2.

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
depth=depth2;

%%
trace = 61;
vp_b_1d=vp_b(:,trace); vs_b_1d=vs_b(:,trace); rho_b_1d=rho_b(:,trace);
vp_m_1d=vp_m(:,trace); vs_m_1d=vs_m(:,trace); rho_m_1d=rho_m(:,trace);
data_base=[vp_b_1d vs_b_1d rho_b_1d]; % NN*3
data_moni=[vp_m_1d vs_m_1d rho_m_1d]; 

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
tic

por = zeros(length(depth),1);
qu = zeros(length(depth),1);
vcl = zeros(length(depth),1);
parfor i=1:length(depth)  
    rng(42 + i); % rand seed
    ObjFunc = @(m)obj_stage1(m, data_base(i,:),Pefft(i), sct(i), weight);
    [model_opt,mfitmin,na_modelsND,workNA2] = ...
    NA_point(ObjFunc,nd1,lb1,ub1,itmax,nsamplei,nsample,ncells,mfbreak);
    por(i)=model_opt(1);
    qu(i)=model_opt(2);
    vcl(i)=model_opt(3);
    disp(i);
end
toc

vp = zeros(size(por));
vs = zeros(size(por));
rho = zeros(size(por));
parfor i =1:length(por)
    m = [por(i) qu(i) vcl(i) Pefft(i) sct(i)];
    d = model_sand_CMC_vector(m);
    vp(i) = d(1);
    vs(i) = d(2);
    rho(i) = d(3);
end

plot_well_inversion;

%% Stage 2: invert for sc (monitoring)
sc = zeros(length(depth),1);
parfor i=1:length(depth)  
    rng(42 + i); % rand seed
    ObjFunc = @(m)obj_stage2(m, data_moni(i,:), por(i),qu(i),vcl(i),Pefft(i), weight);
    [model_opt,mfitmin,na_modelsND,workNA2] = ...
    NA_point(ObjFunc,nd2,lb2,ub2,itmax,nsamplei,nsample,ncells,mfbreak);
    sc(i)=model_opt(1);
    disp(i);
end

figure;
plot(sc,depth,'k','linewidth',2);set(gca,'YDir','reverse');
xlabel('CO2 saturation'); ylabel("Depth (m)")
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on




