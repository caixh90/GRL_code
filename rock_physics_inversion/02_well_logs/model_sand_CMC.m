function [vp,vs,rho]=model_sand_CMC(por,qu,vcl,Peff,sc)  

mineral_properties

%% 50-200 Peff=1.5; 200-350 Peff=5;
co=1-qu-vcl;

Km=1/2*(qu*Kq+vcl*Kc+co*Kco+(1./(qu/Kq+vcl/Kc+co/Kco)));
Gm=1/2*(qu*Gq+vcl*Gc+co*Gco+(1./(qu/Gq+vcl/Gc+co/Gco)));
rhom=qu*rhoq+vcl*rhoc+co*rhoco;

%% fluid mixture: Brie's equation
Kf=(Kw-Kg).*(1-sc).^3+Kg; 
rhof=rhow*(1-sc)+rhog*sc;

rho=(1-por).*rhom+por.*rhof;

%% refers to rock physics handbook, Page.260-261  and Grana 2016
Peff=Peff/1e3;
C=30-34*por+por.^2; % coordination number: average number of contacts per grain
poisson=0.5*(3*Km-2*Gm)./(3*Km+Gm);
phi0=0.4;
%% Hertz-Mindlin theory
Khm=(Peff.*(C.*(1-phi0).*Gm).^2./(18*(pi*(1-poisson)).^2)).^(1/3);
f=0.4; % fraction of grain contacts
Ghm=1/5*(2+3*f-poisson*(1+3*f))./(2-poisson).*(Peff.*3.*(C.*(1-phi0).*Gm).^2./(2*(pi*(1-poisson)).^2)).^(1/3);

%% soft sand
xi=1/6*Ghm.*(9*Khm+8*Ghm)./(Khm+2*Ghm);
Kd=(por/phi0./(Khm+4/3*Ghm)+(1-por/phi0)./(Km+4/3*Ghm)).^(-1)-4/3*Ghm;
Gd=(por/phi0./(Ghm+xi)+(1-por/phi0)./(Gm+xi)).^(-1)-xi;

%% Gassmann
K=Kd+(1-Kd./Km).^2./(por./Kf+(1-por)./Km-Kd./Km.^2);
G=Gd;
%%
vp=sqrt((K+4/3*G)./rho);
vs=sqrt(G./rho);
[vp,vs,rho]=deal(vp*1e3,vs*1e3,rho*1e3);