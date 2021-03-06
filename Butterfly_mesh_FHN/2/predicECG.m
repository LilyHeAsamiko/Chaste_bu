function pseudo_ECG = predicECG(sc, x, U2, U, Um, Ue, U1, Cm,r, ar, al,lambdar2, lambdal2, Istim,I_Na,I_K,I_Kr,I_Ca_L)
    if x <= 5
        level = 1
    elseif x >= 7
        level = 2
    elseif x == 6
        Pred = Um;
    end    

    switch level
        case 1
            Pred = zeros(size(U));
            if x == 2
                temp = real(gammainc(sc*(Istim+I_Na+I_K+I_Kr+I_Ca_L)/Cm,1, 'upper'));
                temp(temp>1) = 1;
                temp = gammaincinv(temp, 1,'upper')/sc;
                Predu= [U2(1);U2(1:(length(U2)-1))+ temp(1:(length(U2)-1))];
                i = 5;
                Prede = zeros(size(U));
                while i >= x
                    dVdxl = abs(al/(i-6));
                    drdl = 1/sqrt((i-11)^2)-1/sqrt((i+1-11)^2);
                    V = r^2*lambdal2/4.*(-dVdxl).*drdl;
                    V = V-min(V);
                    Prede = Prede + [V(1);V];
                    i = i-1;
                end
                Pred =Prede(1:(length(Prede)-1))+Predu(1:(length(Predu)-1));
                Pred = [Pred(1);Pred];    
            elseif x== 1
                Pred = U2;
            elseif x ==3
                Pred = U;
            else
                i = 4;
                Predu = zeros(size(U));
                while i <= x
                    dVdxr = abs(ar/(i-6));
                    drdr = 1/sqrt((i-11+1)^2)-1/sqrt((i-11)^2);
                    V = r^2*lambdar2/4.*(-dVdxr).*drdr;
                    Predu = Predu + [V(1);V];
                    i = i+1;
                end
                i = 5;
                Prede = zeros(size(U));
                while i >= x
                    dVdxl = abs(al/(i-6));
                    drdl = 1/sqrt((i-11)^2)-1/sqrt((i+1-11)^2);
                    V = r^2*lambdal2/4.*(-dVdxl).*drdl;
                    V = V-min(V);
                    Prede = Prede + [V(1);V];
                    i = i-1;
                end
                Pred =Prede(1:(length(Prede)-1))+Predu(1:(length(Predu)-1));
                Pred = [Pred(1);Pred];    
            end
        case 2
            if x == 10
                temp = real(gammainc(sc*(Istim+I_Na+I_K+I_Kr+I_Ca_L)/Cm,1, 'upper'));
                temp(temp>1) = 1;
                temp = gammaincinv(temp, 1,'upper')/sc;
                Predu= [U1(1);U1(1:(length(U1)-1))+ temp(1:(length(U1)-1))];
                i = 7;
                Prede = zeros(size(Ue));                
                while i <= x
                    dVdxr = abs(ar/(i-6));
                    drdr = 1/sqrt((i-11+1)^2)-1/sqrt((i-11)^2);
                    V = r^2*lambdar2/4.*(-dVdxr).*drdr;
                    if mean(V)<0
                        V = V/mean(V);
                    else 
                        V = V-min(V);
                    end
                    Prede = Prede + [V(1);V];
                    i = i+1;
                end
                Pred =Prede(1:(length(Prede)-1))+Predu(1:(length(Predu)-1));
                Pred = [Pred(1);Pred];
            elseif x== 11
                Pred = U1;
            elseif x ==9
                Pred = Ue;
            else
                i = 8;
                Predu = zeros(size(Ue));
                while i >= x
                    dVdxl = abs(al/(i-6));
                    drdl = 1/sqrt((i-11)^2)-1/sqrt((i+1-11)^2);
                    V = r^2*lambdal2/4.*(-dVdxl).*drdl;
                    if mean(V)<0
                        V = V/mean(V);
                    else 
                        V = V-min(V);
                    end
                    Predu = Predu + [V(1);V];
                    i = i-1;
                end
                i = 7;
                Prede = zeros(size(Ue));                
                while i <= x
                    dVdxr = abs(ar/(i-6));
                    drdr = 1/sqrt((i-11+1)^2)-1/sqrt((i-11)^2);
                    V = r^2*lambdar2/4.*(-dVdxr).*drdr;
 %                   V = V-min(V);
                    Prede = Prede + [V(1);V];
                    i = i+1;
                end
                Pred =Prede(1:(length(Prede)-1))+Predu(1:(length(Predu)-1));
                Pred = [Pred(1);Pred];
            end
    end
    pseudo_ECG = Pred;
end