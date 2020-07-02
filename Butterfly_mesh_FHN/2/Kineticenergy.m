function Ek = Kineticenergy(C_in, C_out,T,z)
    R = 1.989; %Cal/K/mo
    F = 23.061; %kcal/V*gram
    Ek = R*T*log(C_out/C_in)/(z*F);
end

