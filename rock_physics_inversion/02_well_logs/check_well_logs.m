% This code is used to:
% 1. generate smooth well logs for testing
% 2. verify that the rock-physics model performs well

clc;clear; close all;

dat = readmatrix('Logs_qi.xlsx');

x=find(dat(:,1)<0 | dat(:,1)>520);  %% rock physics modeling
dat(x,:)=[];

depth = dat(:,1);
vp0 = dat(:,2); 
vs0=dat(:,3);   
rho0=dat(:,4);  
por0 = dat(:,5);
qu0 = dat(:,6);
vcl0 = dat(:,7);
co0 = dat(:,9);
Peff0= dat(:,10);

vsum0=qu0+vcl0+co0;
[qu0,vcl0,co0]=deal(qu0./vsum0,vcl0./vsum0,co0./vsum0);


span=30;
method='loess';
vp=smooth(vp0,span,method);
vs=smooth(vs0,span,method);
rho=smooth(rho0,span,method);
por=smooth(por0,span,method);
vcl=smooth(vcl0,span,method);
qu=smooth(qu0,span,method);
co=smooth(co0,span,method);
Peff=smooth(Peff0,span,method);

co(co<0)=0;

vsum=qu+vcl+co;
[qu,vcl,co]=deal(qu./vsum,vcl./vsum,co./vsum);


fig=figure('Name','fig');
set(fig,'position',[100,100,1500,600])
subplot(171); 
plot(vp0,depth);
hold on
plot(vp,depth);
grid on;set(gca,'yDir','reverse');xlabel('Vp');ylabel('Depth (m)');set(gca,'fontsize',12)
subplot(172); 
plot(vs0,depth);
hold on
plot(vs,depth);
grid on;set(gca,'yDir','reverse');xlabel('Vs');set(gca,'fontsize',12)
subplot(173); 
plot(rho0,depth);
hold on
plot(rho,depth);
grid on;set(gca,'yDir','reverse');xlabel('Den');set(gca,'fontsize',12)
subplot(174); 
plot(por0,depth);
hold on
plot(por,depth);
grid on;set(gca,'yDir','reverse');xlabel('Por');set(gca,'fontsize',12)
subplot(175); 
plot(qu0,depth);
hold on
plot(qu,depth);
grid on;set(gca,'yDir','reverse');xlabel('Quartz');set(gca,'fontsize',12)
subplot(176); 
plot(vcl0,depth);
hold on
plot(vcl,depth);
grid on;set(gca,'yDir','reverse');xlabel('Clay');set(gca,'fontsize',12)
subplot(177); 
plot(co0,depth);
hold on
plot(co,depth);
grid on;set(gca,'yDir','reverse');xlabel('Coal');set(gca,'fontsize',12)

%% check the simplified model
[vp2,vs2,rho2]=model_sand_CMC(por,qu,vcl,Peff, 0);

fig=figure('Name','fig');
set(fig,'position',[100,100,1000,600])
subplot(131)
plot(vp,depth);ylabel('Depth (m)');
hold on
plot(vp2,depth);
grid on;set(gca,'yDir','reverse');xlabel('Vp');set(gca,'fontsize',12)
subplot(132)
plot(vs,depth);
hold on
plot(vs2,depth);
grid on;set(gca,'yDir','reverse');xlabel('Vs');set(gca,'fontsize',12)
h=legend('true','predicted','location','northoutside','orientation','horizontal','FontSize',18);
legend('boxoff')
set(h,'position',[0.5 0.6 0.02 0.75]);
subplot(133)
plot(rho,depth);
hold on
plot(rho2,depth);
grid on;set(gca,'yDir','reverse');xlabel('Density');set(gca,'fontsize',12)


% vpt=[vp2(1:223);vp(224:end)];
% vst=[vs2(1:223);vs(224:end)];
% rhot=[rho2(1:223);rho(224:end)];

vpt=vp;
vst=vs;
rhot=rho;
port=por;
qut=qu;
vclt=vcl;
cot=co;
Pefft=Peff;

fig=figure('Name','fig');
set(fig,'position',[100,100,1500,600])
subplot(161); 
plot(vpt,depth,'linewidth',2);
hold on
plot(vp2,depth,'linewidth',2)
legend('true','predicted','location','best')
grid on;set(gca,'yDir','reverse');xlabel('Vp (m/s)');ylabel('Depth (m)');set(gca,'fontsize',12)
subplot(162); 
plot(vst,depth,'linewidth',2);
hold on
plot(vs2,depth,'linewidth',2)
grid on;set(gca,'yDir','reverse');xlabel('Vs (m/s)');set(gca,'fontsize',12)
subplot(163); 
plot(rhot,depth,'linewidth',2);
hold on
plot(rho2,depth,'linewidth',2)
grid on;set(gca,'yDir','reverse');xlabel('Density (kg/m3)');set(gca,'fontsize',12)
subplot(164); 
plot(port,depth,'linewidth',2);
grid on;set(gca,'yDir','reverse');xlabel('Porosity');set(gca,'fontsize',12)
subplot(165); 
plot(qut,depth,'linewidth',2);
hold on
plot(vclt,depth,'linewidth',2);
plot(cot,depth,'linewidth',2);
legend('quartz','clay','coal','location','best')
grid on;set(gca,'yDir','reverse');xlabel('Mineral Fraction');set(gca,'fontsize',12)
subplot(166); 
plot(Pefft,depth,'linewidth',2);
grid on;set(gca,'yDir','reverse');xlabel('Effective Pressure');set(gca,'fontsize',12)

%%

range = 1:351;
depth=depth(range);
vpt=vpt(range);
vst=vst(range);
rhot=rhot(range);
port=port(range);
qut=qut(range);
vclt=vclt(range);
cot=cot(range);
Pefft=Pefft(range);
save Logs_qi_0_350_smooth depth vpt vst rhot port qut vclt cot Pefft





