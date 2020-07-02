%FunnyCurrentBlock:sodium and potassium, IK and ICaL
load('1_0_0.mat');%U(t, V) at(1,0,0)
load('11_0_0.mat');        %at(11,0,0)
U1 = PseudoEcg1100(:,2);
U2 = PseudoEcg100(:,2);
U1(isnan(U1)) = mean(U1(~isnan(U1)));
U2(isnan(U2)) = mean(U2(~isnan(U2)));

% consider x= 5:0.1:6, and 1D on x 
x = 5:0.1:6;
d1 = sqrt((x-1).^2);
%d2 = sqrt((x-11).^2);
%U1./U2=
%log(C_in/C_                                                                                                                   out) = 1, constant for the zero drug dose basic cycle line 
C_in = 200;
C_out = 200;
T = 22+273.15;
nH = 1;
C = 200;

kda = 1;
ka = 2;

%sc50 = IC(kda, ka, nH, ic, L);
L = 1; %drug concentration

stim.period = 3000;
stim.amplitude = -40; 
stim.duration = 2000;
condition =  'regular';
mrTime = length(U1);
Istim = Stim(stim, mrTime, condition)

Pred = zeros(length(U1), 100);
Res = Pred;
LogRatio = Pred;
CV = U1;
Reward = U1;
Convergrncy = U1;

g_Na = 0.03946;
sc_Na = 0.81;
E_Na = Kineticenergy(8.23, 140, T, 1);
I_Na = Electricity(sc_Na, g_Na, U2, E_Na);
I_Na = I_Na-max(I_Na);
I_Na(isnan(I_Na)) = mean(I_Na(~isnan(I_Na)));


if mod(stim.step, 10) == 0 
    figure,
end
subplot(4,1,1)
plot(I_Na)
title(['I Na in membrane_hyperpolarisation_activated_funny_current_sodium_component_conductance scaled with',num2str(sc_Na)])

g_K = 0.0232;
sc_K = 0.81;
E_K = Kineticenergy(11, 1, T, 1)
I_K = Electricity(sc_K, g_K, U2, E_K)
I_K(isnan(I_K)) = mean(I_K(~isnan(I_K)));

subplot(4,1,2)
plot(I_K)
title(['I K in membrane_hyperpolarisation_activated_funny_current_sodium_component_conductance scaled with',num2str(sc_K)])

g_Kr = 0.0342;
sc_Kr = 0.65;
E_Kr = Kineticenergy(110, 5.4, T, 1);
I_Kr = Electricity(sc_Kr, g_Kr, U1, E_Kr);
I_Kr = I_Kr-min(I_Kr);
I_Kr(isnan(I_Kr)) = mean(I_Kr(~isnan(I_Kr)));

subplot(4,1,3)    
plot(I_Kr)
title(['I Kr in membrane_delayed_rectifier_potassium_current_conductance scaled with',num2str(sc_Kr)])

g_Ca =  7.7677e-2;
sc_Ca = 0.88;
E_Ca = Kineticenergy(4.36e-5, 1.8, T, 2)
I_Ca = Electricity(sc_Ca, g_Ca, U1, E_Ca)
I_Ca(isnan(I_Ca)) = mean(I_Ca(~isnan(I_Ca)));
I_Ca_BL = I_Ca(1:(length(I_Ca)-1));
for i = 1:(length(I_Ca)-1)
    I_Ca_BL(i)=mean(I_Ca(1:i));
end
I_Ca_L = -max(I_Ca_BL)+[I_Ca_BL;mean(I_Ca_BL)]

subplot(4,1,4)
plot(I_Ca_L)
title(['I CaL in membrane_L_type_calcium_current_conductance scaled with',num2str(sc_Ca)])

sc =  1;
    Cm = 179*10^3*10^(-12)*10^3*10^6; %179pF, F:omega
    D = 1.162*100*10^6/10^6%mum^2/ms
    DBL_MAX = max(abs(U1));
    dV = U1(2:length(U1))-U1(1:(length(U1)-1));

    Istim = Cm*[dV;mean(dV)]-(I_Na+I_K+I_Kr+I_Ca_L);
    Istim(length(Istim)-1) = Istim(length(Istim));
    
    %dx = sqrt(D)(n r) mum as no longer than 1/10 of lambda(sqrt(rm/ri))
    %using boundary at x= 1, and x = 11, dV/dx = 0, and D = 1.162 only *5
    %at x = 6, consider abs of parabolic dVdx = abs(a/(x-6)), |a|<1
    % stimulation at x = 11
    % consider boundary condition first at x = 1, at x= 2
    r = 148 %mum;
    dx = sqrt(D);
    d1 = sqrt((2-11)^2);
    drdr = 1/sqrt((3-11)^2)-1/sqrt((2-11)^2);
%    U2 = D*(U-U2)/dx^2-Cm*(U2(2:length(U2))-U2(1:(length(U2)-1)))=r^2*lambda^2/4*sum((U2-U)*drdr)
    U = (U2(1:(length(U2)-1))+Cm*(U2(2:length(U2))-U2(1:(length(U2)-1))))/D*dx^2+U2(1:(length(U2)-1));
    lambdar2 = 4*U/r^2/sum((U2(1:(length(U2)-1))-U)*drdr);
    dVdxr = abs(U-U2(1:(length(U2)-1)))
    ar = abs(dVdx*(3-6));
    U = [U(1);U];
    % consider boundary condition then at x = 11, at x= 2
    r = 148 %mum;
    d2 = sqrt((9-11)^2);
    drdl = 1/sqrt((10-11)^2)-1/sqrt((9-11)^2);
%    U1 = D*(U1-U)/dx^2-Cm*(U1(2:length(U1))-U1(1:(length(U1)-1)))=r^2*lambda^2/4*sum(-(U1-U)*drdr)
    Ue = U1(1:(length(U1)-1))-(U1(1:(length(U1)-1))+Cm*(U1(2:length(U1))-U1(1:(length(U1)-1))))/D*dx^2;
    lambdal2 = 4*Ue/r^2/sum((-U1(1:(length(U1)-1))+Ue)*drdl);
    dVdxl = abs(U1(1:(length(U1)-1))-Ue);
    al = abs(dVdxl*(9-6));
    Ue = [Ue(1);Ue];

    assert(x>=1 & x<= 11);
    % compute mid-point x = 6
    i = 4;
    Predu = zeros(size(U));
    while i < 6
        dVdxr = abs(ar/(i-6));
        drdr = 1/sqrt((i-11+1)^2)-1/sqrt((i-11)^2);
        V = r^2*lambdar2/4.*(-dVdxr).*drdr;
        Predu = Predu + [V(1);V];
        i = i+1;
    end
    i = 8;
    Prede = zeros(size(Ue));
    while i > 6
        dVdxl = abs(al/(i-6));
        drdl = 1/sqrt((i-11)^2)-1/sqrt((i+1-11)^2);
        V = r^2*lambdal2/4.*(-dVdxl).*drdl;
        if mean(V)<0
            V = V/mean(V);
        else
            V = V-min(V);
        end
        Prede = Prede + [V(1);V];
        i = i-1;
    end
    Pred =Prede(1:(length(Prede)-1))+Predu(1:(length(Predu)-1));
    Um = [Pred(1);Pred];
    % Consider prediction of U1 from Um forwardly

    x = 8;
    U8 = predicECG(sc, x, U2, U, Um, Ue, U1,Cm,r, ar, al,lambdar2, lambdal2, Istim,I_Na,I_K,I_Kr,I_Ca_L);   
    dVdt8 = U8(2:length(U8))-U8(1:(length(U8)-1));
    U10 = ([dVdt8(1);dVdt8]+(Istim+I_Na+I_K+I_Kr+I_Ca_L))/D*(2*dx)^2+2*U8-Um;
    dVdt10 = U10(2:length(U10))-U10(1:(length(U10)-1));
    U12 = ([dVdt10(1);dVdt10]+(Istim+I_Na+I_K+I_Kr+I_Ca_L))/D*(2*dx)^2+2*U10-U8;
    dVdt11 = U1(2:length(U1))-U1(1:(length(U1)-1));

steps = 1;
beststep = 1;
for sc = 0.1:0.1:10
    U11 = 0.5/Cm*(U10+U12-sc*([dVdt11(1);dVdt11]+(Istim+I_Na+I_K+I_Kr+I_Ca_L))/D*dx^2); 
    U11 = U1-mean(U11)+U11;
    pred = U11(1:(length(dVdt11)-1))+abs(dVdt11(length(dVdt11)-1));
    pred = [pred(1);pred;pred(length(pred))];
        
    Pred(:,steps)= pred;
    Res(:,steps) = pred - U1;
    LogRatio(:,steps) = log(pred./(U1+0.0000001));
    Convergency(2:size(LogRatio,1),steps) = (LogRatio(2:size(LogRatio,1),steps)-LogRatio(1:(size(LogRatio,1)-1),steps))./(Pred(2:size(LogRatio,1),steps)-Pred(1:(size(LogRatio,1)-1),steps));
    Convergency(1, steps) = Convergency(2, steps);
    means = Res(:,steps);
    stds = means;
    if steps>1
        if mean(Res(:,steps)) <= mean(Res(:,beststep))
           beststep = steps;
        end
    end
    steps = steps +1;
end

Reward = LogRatio(:,beststep);
Convergency(1,:) = Convergency(3,:);
Convergency(2,:) = Convergency(3,:);
Convergency(size(Convergency,1),:) = Convergency(size(Convergency,1)-1,:);

for i = 1:size(LogRatio(:,beststep),1)
    means(i) = mean(Res(1:i,beststep));
    stds(i) = std(Res(1:i,beststep));
    CV(i) = means(i)/stds(i);
end

figure(),
subplot(6 ,1, 2)
plot(Reward)
title('Reward(logratio) with ');
subplot(6, 1, 3)
plot(Convergency(:,beststep))
title('Convergency of Reward with');
subplot(6, 1, 4)
plot(CV)
title('CV');
subplot(6, 1, 1)
plot(Res(:, beststep))
title('Residule')
subplot(6, 1, 5)
errorbar(means,stds);
title('errorbar of CV:');
subplot(6, 1, 6)
plot(U1);
hold on
plot(U11);
legend([{'ECG'},{'reconstructed ECG'}])
title('comparison of ECG and reconstructed');

figure,
ic90 = IC(kda, ka, nH, 0.9, L);
subplot(4,1,1)
plot(U2*ic90)
hold on
plot(U2)
legend([{'APD90'},{'control'}])
title('APD90 at 1 0 0')

ic50 = IC(kda, ka, nH, 0.5, L);
subplot(4,1,2)
plot(U2*ic50)
title('APD50 at 1 0 0')

subplot(4,1,3)
plot(U1*ic90)
hold on
plot(U1)
legend([{'APD90'},{'control'}])
title('APD90 at 11 0 0')

subplot(4,1,4)
plot(U1*ic50)
title('APD50 at 11 0 0')


