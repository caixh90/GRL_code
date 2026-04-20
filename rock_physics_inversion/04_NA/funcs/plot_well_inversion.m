
fig=figure('Name','fig');
set(fig,'position',[100,100,1500,700])

subplot(1,6,1)
plot(vp_b_1d,depth,'color',[0 0.4 1],'linewidth',2);set(gca,'YDir','reverse');
hold on
plot(vp,depth,'color','r','linewidth',2,'LineStyle','--');
% xlim([3.8 4.8])
xlabel('$V_{\rm P}(\rm km/s)$','Interpreter','LaTex')
ylabel('Depth(km)','Interpreter','LaTex')
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on
legend('Input data','Predicted data','location','best')

subplot(1,6,2)

plot(vs_b_1d,depth,'color',[0 0.4 1],'linewidth',2);set(gca,'YDir','reverse');
hold on
plot(vs,depth,'color','r','linewidth',2,'LineStyle','--');
% xlim([2.2 3])
xlabel('$V_{\rm S}(\rm km/s)$','Interpreter','LaTex')
set(gca,'yticklabel',[])
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on

subplot(1,6,3)

plot(rho_b_1d,depth,'color',[0 0.4 1],'linewidth',2);set(gca,'YDir','reverse');
hold on
plot(rho,depth,'color','r','linewidth',2,'LineStyle','--');
% xlim([2 2.4])
xlabel('$\rho({\rm g}/{\rm c}{\rm m}^3)$','Interpreter','LaTex')
set(gca,'yticklabel',[])
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on

subplot(1,6,4)
plot(port,depth,'k','linewidth',2);set(gca,'YDir','reverse');
hold on
plot(por,depth,'r','linewidth',2);
% xlim([0.05 0.3])
xlabel('$\phi$','Interpreter','LaTex')
set(gca,'yticklabel',[])
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on

subplot(1,6,5)
plot(qut,depth,'k','linewidth',2);set(gca,'YDir','reverse');
hold on
plot(qu,depth,'r','linewidth',2);
% xlim([0.3 0.8])
% xticks([0.1 0.3 0.5])
xlabel('Quartz')
set(gca,'yticklabel',[])
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on

subplot(1,6,6)
plot(vclt,depth,'k','linewidth',2);set(gca,'YDir','reverse');
hold on
plot(vcl,depth,'r','linewidth',2);
% xlim([0.3 0.8])
% xticks([0.1 0.3 0.5])
xlabel('Clay')
set(gca,'yticklabel',[])
set(gca,'FontSize',18,'fontname','arial','fontweight','normal')
grid on
legend('True model','Inverted model','location','best')

