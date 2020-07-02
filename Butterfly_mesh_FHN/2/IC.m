function sc = IC(kda, ka, nH, ic, L)
    %simplest ic50=kd=ec50
    Kd = kda/ka;%L+R->LR with k1, LR desociate into L , R with k2
    Kd = max(ic, Kd);
    Y = L^nH/(L^nH+Kd);
    sc = 1-log(Y/(1-Y));
end
