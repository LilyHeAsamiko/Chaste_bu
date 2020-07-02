# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:53:06 2020

@author: LilyHeAsamiko
"""
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as sci
import researchpy as rp
import pandas as pd
import statsmodels.formula.api as sfa
import statsmodels.api as sa
from sklearn.linear_model import LinearRegression
import quandl
import random
import scipy
import mpl_toolkits
from matplotlib import cm

def E(C_in, C_out, T, z):
    R = 8.3144626 #J/K/mol
    F = 96485.3329 #sA/mol
    E = R*T*np.log(C_out/C_in)
    return E
    

def I_Na(V, h_f, h_s,  G_Na, E_Na, m):
    minf = 1/(1+np.exp(-(V+48.6264)/9.871))
    taum = 1/(6.755*np.exp((V+11.64)/34.77)+8.552*np.exp(-(V+77.42)/5.955))
    hinf = 1/(1+np.exp((V+78.5)/6.22))
    tauh_f = 1/(3.686*10e-6*np.exp(-(V+3.8875)/7.8579)+16*np.exp((V-0.4963)/9.1843))
    tauh_s = 1/(0.009764*np.exp(-(V+17.95)/28.05)+0.3343*np.exp((V+5.730)/56.66))    
    Ah_f = 0.99
    Ah_s = 1 - Ah_f
    hCaMK_f = Ah_f*h_f
    hCaMK_s =  Ah_s*h_s
    jinf = hinf
    tauj = 4.8590 + 1/(0.8628*np.exp(-(V+166.7)/7.6)+1.1096*np.exp(-(V+6.2719)/9.0358))
    hCaMK_inf = 1/(1 + np.exp(-(V+84.7)/6.22))
    tauh_CaMK_s = 3*tauh_s
    Ah_CaMK_f = Ah_f
    Ah_CaMK_s = 1 - Ah_CaMK_f 
    h_CaMK_inf = h_f
    h_CaMK = Ah_CaMK_f*hCaMK_f + Ah_CaMK_s*hCaMK_s
    j_CaMK = jinf
    tauj_CaMK = 1.46*tauj
    Km_CaMK = 0.15
    phiINa_CaMK = 1/(1+Km_CaMK/(1+Km_CaMK))
    INa_f = G_Na*(V-E_Na)*m**3*((1-phiINa_CaMK)*h_CaMK*j_CaMK + phiINa_CaMK*h_CaMK*j_CaMK)
    return INa_f

def I_to(V, a, i_s, i_f, g, gto, EK):#transient Outward Current
    ainf = 1/np.exp((20-V)/13)
    taua = 1.0515/(1/(1.2*(1+np.exp(-(V-18.41)/29.38)))+3.5/(1+np.exp(-(V+100)/29.38)))
    i1_inf = 1/(1+np.exp((V+27)/13))
    i2_inf = i1_inf
    taui_S = 43 + 1/(0.001416*np.exp(-(V+96.52)/59.05)+1.7*10e-8*np.exp((V+11.41)/8.079))
    taui_F = 6.16 + 1/(0.39*np.exp(-(V+100)/100)+0.08*np.exp((V-8)/8.59))
    I_to = gto*a*i_s*i_f*g*(V-EK)
    return I_to

def I_sus(V, gsus, EK):
    a_sus = 1/(1+np.exp(-(V-12)/16))
    I_sus = gsus*a_sus*(V-EK)
    return I_sus

def I_Kr(V, CKpout, xr_s, xr_f, gkr, EK):
    xrinf = 1/(1+np.exp(-(V+8.337)/6.789))
    tauxr_f = 12.98 + 1/(0.3652*np.exp((V-14.06)/3.869)+4.123*10e-5*np.exp(-(V-30.18)/20.38))
    tauxr_s = 1.865 + 1/(0.06629*np.exp((V-19.7)/7.355)+1.128*10e-5*np.exp(-(V-12.54)/25.94))
    Axr_f = 1/(1+np.exp((V+54.81)/38.21))
    Axr_s = 1 - Axr_f 
    x_r = Axr_f*xr_f + Axr_s*xr_s 
    R_Kr = 1/(1+np.exp((V+55)/24))/(1+np.exp((V-10)/9.6))
    I_Kr = gkr*(CKpout/5.4)**0.5*x_r*R_Kr*(V-EK) 
    return I_Kr

def I_Ks(V, gks, Ca2p_sl, EK):
    xs1_inf = 1/(1+np.exp(-(V+11.6)/8.932))
    taux_s1 = 817.3 + 1/(0.001292*np.exp(-(V+210)/230)+2.326*10e-4*np.exp((V+48.28)/17.8))
    xs2_inf = xs1_inf   
    taux_s2 = 1/(0.01*np.exp((V-50)/20)+0.0193*np.exp(-(V+66.54)/31))
    I_Ks = gks*(1+0.6/(1+3.8*10e-5/Ca2p_sl)**1.4)*xs1_inf*xs2_inf*(V-EK) 
    return I_Ks

def I_f(V, gf_K, gf_Na, EK, ENa):
    y_inf = 1/(1+np.exp((V+87)/9.5))
    tauy = 2000/(np.exp(-(V+132)/10)+np.exp((V+57)/60))
    if_K = gf_K*y_inf*(V-EK)   
    if_Na = gf_Na*y_inf*(V-ENa) 
    I_f = if_K+if_Na
    return I_f

def I_K1(V, gk1, CKpout, Ek):
    xk1_inf = 1/(1+np.exp(-(V+2.5538*CKpout+144.59)/(1.5692*CKpout+3.8115)))
    taux_k1 = 122.2/(np.exp(-(V+127.2)/20.36)+np.exp((V+236.8)/69.33))
    R_K1 = 1/(1+np.exp((V+116-5.5*CKpout)/11))   
    taux_s2 = 1/(0.01*np.exp((V-50)/20)+0.0193*np.exp(-(V+66.54)/31))
    I_K1 = gk1*(CKpout/5.4)**0.5*xk1_inf*R_K1*(V-Ek) 
    return I_K1

def Calibration(V, fs):
    stair = np.repeat(0.1, 16760)
    stair[0:250] = np.repeat(-80, 250)
    stair[250:300] = np.linspace(-80, -120, 50)
    stair[300:700] = np.linspace(-120, -80, 400)
    stair[700:900] = np.repeat(-80, 200)
    stair[900:1900] = np.repeat(40, 1000)
    stair[1900:2400] = np.repeat(-120, 500)
    stair[2400:3400] = np.repeat(-80, 1000)
    
    stair[3400:3900] = np.repeat(-40, 500) 
    stair[3900:3920] = np.linspace(-40, -60, 20)
    stair[3920:4420] = np.repeat(-60, 500) 
    stair[4420:4440] = np.linspace(-60, -20, 20)
    stair[4440:4940] = np.repeat(-20, 500) 
    stair[4940:4960] = np.linspace(-20, -40, 20)
    stair[4960:5460] = np.repeat(-40, 500) 
    stair[5460:5480] = np.linspace(-40, 0, 20)
    stair[5480:5980] = np.repeat(0, 500) 
    stair[5980:6000] = np.linspace(0, -20, 20)
    stair[6000:6500] = np.repeat(-20, 500) 
    stair[6500:6520] = np.linspace(-20, 20, 20)
    stair[6520:7020] = np.repeat(20, 500) 
    stair[7020:7040] = np.linspace(20, 0, 20)
    stair[7040:7540] = np.repeat(0, 500) 
    stair[7540:7560] = np.linspace(0, 40, 20)
    stair[7560:8060] = np.repeat(40, 500) 
    stair[8060:8080] = np.linspace(40, 20, 20)
    stair[8080:8580] = np.repeat(20, 500) 
    stair[8580:14760] = list(stair[2400: 8580])[::-1]
    
    stair[14760:15260] = np.repeat(-80, 500)
    stair[15260:15760] = np.linspace(-80, 40, 500)
    stair[15760:15770] = np.linspace(40, -70, 10)
    stair[15770:15870] = np.linspace(-70, -110, 100)
    stair[15870:16260] = np.linspace(-110, -120, 390)
    stair[16260:16760] = np.repeat(-80, 500) 
    xnew = np.linspace(0,len(stair), round(len(stair)*fs/1000))
    xnew = np.linspace(0,len(stair), len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(stair)-1, len(stair)), stair)
#    temp = f(xnew)
    calib = f(xnew)

    return calib

def Val1(V):
    V1 = np.ones((1700, 7))+0.1
    t = 0
    while t < 7:
        V1[0:100, t] = np.repeat(-80, 100)
        V1[100: 1100, t] = np.repeat(-50+t*15, 1000)    
        V1[1100: 1600, t] = np.repeat(-40, 500)
        V1[1600: 1700, t] = np.repeat(-80, 100)
        t += 1 
    xnew = np.linspace(0,len(V1), len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(V1)-1, len(V1)), V1)
    V1 = f(xnew) 
    return V1

def Val2(V):
    V2 = np.ones((1200, 10))+0.1
    t = 0
    while t < 10:
        V2[0: 100, t] = np.repeat(-80, 100)
        V2[100: 600, t] = np.repeat(20, 500)        
        V2[600: 1100, t] = np.repeat(-140+t*20, 500)  
        V2[1100: 1200, t] = np.repeat(-80, 100)
        t += 1 
    xnew = np.linspace(0,len(V)-1, len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(V)-1, len(V)), V)
    V2 = f(xnew) 
    return V2

def Val3(V):
    V3 = np.repeat(0.1, 1350)
    V3[0: 100] = np.repeat(-80, 100)
    V3[100: 150] = np.repeat(-40, 50)        
    V3[150: 650] = np.repeat(20, 500)    
    V3[650: 1150] = np.repeat(-40, 500)
    V3[1150: 1350] = np.repeat(-80, 200)
    xnew = np.linspace(0,len(V)-1, len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(V), len(V)), V)
    V3 = f(xnew) 
    return V3

def Val4(V):
    DAD = np.repeat(0.1, 749)
    DAD[0: 50] = np.repeat(-80, 50)
    DAD[50: 53] = np.repeat(34, 3)        
    DAD[53: 61] = np.linspace(34, 30, 8)    
    DAD[61: 76] = np.linspace(30, 26, 15)
    DAD[76: 219] = np.linspace(26, -5, 143)    
    DAD[219: 257] = np.linspace(-5, -21, 38)    
    DAD[257: 325] = np.linspace(-21, -70, 68)
    DAD[325: 327] = np.repeat(-20, 2)    
    DAD[327: 347] = np.linspace(-20, -30, 20)    
    DAD[347: 357] = np.linspace(-30, -40, 10)
    DAD[357: 372] = np.linspace(-40, -65, 15)
    DAD[372: 384] = np.linspace(-65, -80, 12)
    DAD[384: 749] = np.repeat(-80, 365)
    xnew = np.linspace(0,len(DAD)-1, len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(DAD), len(DAD)), DAD)
    DAD_con = f(xnew) 
    return DAD_con

def Val5(V):
    EAD = np.repeat(0.1, 720)
    EAD[0: 50] = np.repeat(-80, 50)
    EAD[50: 53] = np.repeat(40, 3) 
    EAD[53: 56] = np.repeat(20, 3)       
    EAD[56: 76] = np.linspace(20, 30, 20)    
    EAD[76: 86] = np.repeat(30, 10)    
    EAD[86: 254] = np.linspace(30, -10, 168)    
    EAD[254: 305] = np.linspace(-10, -15.5, 51)
    EAD[305: 366] = np.linspace(-15.5, -20, 61)
    EAD[366: 426] = np.repeat(-20, 60)    
    EAD[426: 436] = np.linspace(-20, -10, 10)
    EAD[436: 446] = np.repeat(-10, 10)    
    EAD[446: 496] = np.linspace(-10, -20, 50)
    EAD[496: 516] = np.linspace(-20, -30, 20)
    EAD[516: 556] = np.linspace(-30, -75, 40)
    EAD[556: 606] = np.linspace(-75, -80, 50)
    EAD[606: 720] = np.repeat(-80, 114)
    xnew = np.linspace(0,len(EAD)-1, len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(EAD), len(EAD)), EAD)
    EAD_con = f(xnew) 
    return EAD_con

def Val6(V):
    AP = np.repeat(0.1, 395)
    AP[0: 50] = np.repeat(-80, 50)
    AP[50: 53] = np.repeat(34, 3) 
    AP[53: 61] = np.linspace(34, 30, 8)       
    AP[61: 76] = np.linspace(30, 26, 15)    
    AP[76: 260] = np.linspace(26, -8, 184)    
    AP[260: 299] = np.linspace(-8, -21, 39)    
    AP[299: 370] = np.linspace(-21, -68, 71)
    AP[370: 395] = np.linspace(-68, -80, 25)
    xnew = np.linspace(0,len(AP)-1, len(V))
    f = scipy.interpolate.interp1d(np.linspace(0,len(AP), len(AP)), AP)
    AP_con = f(xnew) 
    return AP_con

def approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkr, V, EK, state):
    # one dimension V should be a verticle vector
    # 9 parameters units: s-1, V-1, s-1, V-1, ..., pS
    k1 = p1*np.exp(p2*V) #activation rate CI a*(1-r) I, C a*r O
    k2 = p3*np.exp(-p4*V) #deactivation rate I (1-r)*(1-a) CI, O r*(1-a) C
    k3 = p5*np.exp(p6*V) #inactivation rate O a*(1-r) I, C (1-r)*(1-a) CI
    k4 = p7*np.exp(-p8*V) #recovery rate I a*r O, CI r*(1-a) C
    ainf = k1/(k1 + k2)
    rinf = k4/(k3 + k4)
    taua = 1/(k1 + k2)
    taur = 1/(k3 + k4)
    da = ainf/taua*dt
    dr = rinf/taur*dt
#    da = ainf/taua - np.repeat(np.linspace(0, np.shape(V)[1]-1, np.shape(V)[1]),np.shape(V)[0]).reshape(np.shape(V))*(1/dt/taua +1/dt)    
#    dr = rinf/taur - np.repeat(np.linspace(0, np.shape(V)[1]-1, np.shape(V)[1]),np.shape(V)[0]).reshape(np.shape(V))*(1/dt/taur +1/dt)    
    if np.shape(V)[0] == 1:
#        print()
        exit('input one column of time series')
    elif np.size(V) == np.shape(V)[0]:  
        a = ainf -da*np.linspace(0, np.shape(V)[0]-1, np.shape(V)[0])/dt
        r = rinf -dr*np.linspace(0, np.shape(V)[0]-1, np.shape(V)[0])/dt
    else:      
        a = ainf - da*np.repeat(np.linspace(0, np.shape(V)[0]-1, np.shape(V)[0])/dt, np.shape(V)[1]).reshape(np.shape(V))
        r = rinf - dr*np.repeat(np.linspace(0, np.shape(V)[0]-1, np.shape(V)[0])/dt, np.shape(V)[1]).reshape(np.shape(V))

    if state == 1:
        Ikr = gkr*a*r*(V-EK)
    elif state == 2:
        Ikr = gkr*a*(1-r)*(V-EK) 
    elif state == 3:
        Ikr = gkr*(1-a)*(1-r)*(V-EK) 
    elif state == 4:
        Ikr = gkr*(1-a)*r*(V-EK)         
    return Ikr

def Calibrate_Validate(I0, calib, gkr0, tp, N, states, drifts, dt, p1, p2, p3, p4, p5, p6, p7, p8, EK):
    EXP = np.exp(I0)
    A = np.sort(np.linspace(min(I0), max(I0), N))
    B = np.sort(np.linspace(min(EXP), max(EXP), N)/max(EXP)/10)
    dVmax = 160
    p1 = 1/(450*dt)
#    p2 = 1/40/dVmax
    p2 = -1/40
    p3 = 1/(500*dt)
    p4 = -1/160
    p5 = 1/(500*dt)
    p6 = 1/120
    p7 = 1/(500*dt)
    p8 = -1/120
    IkrO = []
    GkrO = []
    ResiduleO = []
    Ikr = []
    Gkr = []
    Residule = []
    
    for i in range(len(tp)):
        V = calib[(tp[i]+drifts[i]):(tp[i]+drifts[i]+N)] 
        V1 = Val1(V)
        V2 = Val2(V)
        V3 = Val3(V)
        DAD = Val4(V)
        EAD = Val5(V)
        AP = Val6(V)
        state = states[i]
        I0temp = (I0[tp[i]:tp[i]+N]).ravel()
        gkr = gkr0
        Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkr, V, EK, state), dtype = float)
        gkr = I0temp/Itemp*gkr 
        Itempcorr = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkr, V, EK, state), dtype = float)

        residule = Itempcorr-I0temp
        gkrcor = I0temp/Itempcorr*gkr
        IkrO.append(Itempcorr)
        GkrO.append(gkrcor)
        ResiduleO.append(residule)
        
        Itempval = Itempcorr
        residuleval  = residule
        gkrval = gkrcor
        

        #Gkr.append(np.mean(gkrcor))
        #validation 1  
        if min(I0temp) <0:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, np.repeat(gkrcor[0:min(N, 1700)], np.shape(V1)[1]).reshape(min(N, 1700), np.shape(V1)[1]), -V1[0:min(N, 1700)], EK, state), dtype = float) #Val2(V[0:1700])
        else:  
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, np.repeat(gkrcor[0:min(N, 1700)], np.shape(V1)[1]).reshape(min(N, 1700), np.shape(V1)[1]), V1[0:min(N, 1700)], EK, state), dtype = float) #Val2(V[0:1700])
        gkrtemp = np.repeat(I0temp[0:min(N, np.shape(V1)[0])]/Itemp[:-1], np.shape(V1)[1]).reshape(min(N, np.shape(V1)[0]), np.shape(V1)[1])*np.repeat(gkr[0:min(N, np.shape(V1)[0])], np.shape(V1)[1]).reshape(min(N, np.shape(V1)[0]), np.shape(V1)[1])    
        
        if sum(abs(Itemp[:,-1]-I0temp[0:min(N, np.shape(V1)[0])]) <= residuleval[0:min(N, np.shape(V1)[0])])>0:
            idxtemp = abs(Itemp[:,-1]-I0temp[0:min(N, np.shape(V1)[0])]) <= residuleval[0:min(N, np.shape(V1)[0])]
            Itempval[idxtemp] = Itemp[idxtemp, -1]
            gkrval[idxtemp] = gkrtemp[idxtemp, -1] 
            residuleval = Itempval-I0temp
        
            plt.figure()
            plt.plot(Itempcorr,'r-')
            plt.plot(I0temp,'g--')
            plt.plot(Itempval, 'b*')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation1')
    
            plt.figure()
            plt.plot(residule,'r')
            plt.plot(residuleval, 'g')  
            plt.title(str(i)+' '+'th'+' '+'residule for validation1')
        #    if sum(residuleval <= residule)>=0:
        #        residule = residuleval
        #validation 2  
        if min(I0temp) <0:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, np.repeat(gkrcor[0:min(N, np.shape(V2)[0])], np.shape(V2)[1]).reshape(min(N, np.shape(V2)[0]), np.shape(V2)[1]), -V2[0:min(N, np.shape(V2)[0])], EK, state), dtype = float) #Val2(V[0:1200])
        else:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, np.repeat(gkrcor[0:min(N, np.shape(V2)[0])], np.shape(V2)[1]).reshape(min(N, np.shape(V2)[0]), np.shape(V2)[1]), V2[0:min(N, np.shape(V2)[0])], EK, state), dtype = float) #Val2(V[0:1200])
        gkrtemp = np.repeat(I0temp[0:min(N, np.shape(V2)[0])]/Itemp[:,-1], np.shape(V2)[1]).reshape(min(N, np.shape(V2)[0]), np.shape(V2)[1])*np.repeat(gkr[0:min(N, np.shape(V2)[0])], np.shape(V2)[1]).reshape(min(N, np.shape(V2)[0]), np.shape(V2)[1])    
        if sum(abs(Itemp[:,-1]-I0temp[0:min(N, np.shape(V2)[0])]) <= residuleval[0:min(N, np.shape(V2)[0])])>0:
            idxtemp = abs(Itemp[:,-1]-I0temp[0:min(N, np.shape(V2)[0])]) <= residuleval[0:min(N, np.shape(V2)[0])]
            for i in range(len(Itempval)):
                if i >= len(idxtemp):
                    break
                if idxtemp[i]== True:
                    Itempval[i,] = Itemp[i, -1]                
                    gkrval[i,] = gkrtemp[i, -1] 
            residuleval = Itempval-I0temp
            
            plt.figure()
            plt.plot(Itempcorr,'r-')
            plt.plot(I0temp,'g--')
            plt.plot(Itempval, 'b*')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation2')

            plt.figure()
            plt.plot(residule,'r')
            plt.plot(residuleval, 'g')
            plt.title(str(i)+' '+'th'+' '+'residule for validation2')
            
        #validation 3  
        if min(I0temp) <0:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(V3)[0])], -V3[0:min(N, np.shape(V3)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        else:    
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(V3)[0])], V3[0:min(N, np.shape(V3)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        gkrtemp = I0temp[0:min(N, np.shape(V3)[0])]/Itemp*gkr[0:min(N, np.shape(V3)[0])] 
        if sum(abs(Itemp-I0temp[0:min(N, np.shape(V3)[0])]) <= residuleval[0:min(N, np.shape(V3)[0])])>0:
            idxtemp = abs(Itemp-I0temp[0:min(N, np.shape(V3)[0])]) <= residuleval[0:min(N, np.shape(V3)[0])]
            for i in range(len(Itempval)):
                if i >= len(idxtemp):
                    break
                if idxtemp[i]== True:
                    Itempval[i] = Itemp[i]                
                    gkrval[i] = gkrtemp[i] 
            residuleval = Itempval-I0temp
            
            plt.figure()
            plt.plot(Itempcorr,'r-')
            plt.plot(I0temp,'g--')
            plt.plot(Itempval, 'b*')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation3')

            plt.figure()
            plt.plot(residule,'r')
            plt.plot(residuleval, 'g')
            plt.title(str(i)+' '+'th'+' '+'residule for validation 3')
            
        #validation 4  
        if min(I0temp) <0:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(DAD)[0])], -DAD[0:min(N, np.shape(DAD)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        else:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(DAD)[0])], DAD[0:min(N, np.shape(DAD)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        gkrtemp = I0temp[0:min(N, np.shape(DAD)[0])]/Itemp*gkr[0:min(N, np.shape(DAD)[0])] 
        if sum(abs(Itemp-I0temp[0:min(N, np.shape(DAD)[0])]) <= residuleval[0:min(N, np.shape(DAD)[0])])>0:
            idxtemp = abs(Itemp-I0temp[0:min(N, np.shape(DAD)[0])]) <= residuleval[0:min(N, np.shape(DAD)[0])]
            for i in range(len(Itempval)):
                if i >= len(idxtemp):
                    break
                if idxtemp[i]== True:
                    Itempval[i] = Itemp[i]                
                    gkrval[i] = gkrtemp[i] 
            residuleval = Itempval-I0temp
            
            plt.figure()
            plt.plot(Itempcorr,'r-')
            plt.plot(I0temp,'g--')
            plt.plot(Itempval, 'b*')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation4')

            plt.figure()
            plt.plot(residule,'r')
            plt.plot(residuleval, 'g')  
            plt.title(str(i)+' '+'th'+' '+'residule for validation4')
        
        #validation 5 
        if min(I0temp) <0:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(EAD)[0])], -EAD[0:min(N, np.shape(EAD)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        else:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(EAD)[0])], EAD[0:min(N, np.shape(EAD)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        gkrtemp = I0temp[0:min(N, np.shape(EAD)[0])]/Itemp*gkr[0:min(N, np.shape(EAD)[0])] 
        if sum(abs(Itemp-I0temp[0:min(N, np.shape(EAD)[0])]) <= residuleval[0:min(N, np.shape(EAD)[0])])>0:
            idxtemp = abs(Itemp-I0temp[0:min(N, np.shape(EAD)[0])]) <= residuleval[0:min(N, np.shape(EAD)[0])]
            for i in range(len(Itempval)):
                if i >= len(idxtemp):
                    break
                if idxtemp[i]== True:
                    Itempval[i] = Itemp[i]                
                    gkrval[i] = gkrtemp[i] 
            residuleval = Itempval-I0temp
            
            plt.figure()
            plt.plot(Itempcorr,'r-')
            plt.plot(I0temp,'g--')
            plt.plot(Itempval, 'b*')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation5')

            plt.figure()
            plt.plot(residule,'r')
            plt.plot(residuleval, 'g')  
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation5')
            
        #validation 6  
        if min(I0temp) <0:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(AP)[0])], -AP[0:min(N, np.shape(AP)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        else:
            Itemp = np.array(approximate(dt, p1, p2, p3, p4, p5, p6, p7, p8, gkrcor[0:min(N, np.shape(AP)[0])], AP[0:min(N, np.shape(AP)[0])], EK, state), dtype = float) #Val2(V[0:1700])
        gkrtemp = I0temp[0:min(N, np.shape(AP)[0])]/Itemp*gkr[0:min(N, np.shape(AP)[0])] 
        if sum(abs(Itemp-I0temp[0:min(N, np.shape(AP)[0])]) <= residuleval[0:min(N, np.shape(AP)[0])])>0:
            idxtemp = abs(Itemp-I0temp[0:min(N, np.shape(AP)[0])]) <= residuleval[0:min(N, np.shape(AP)[0])]
            for i in range(len(Itempval)):
                if i >= len(idxtemp):
                    break
                if idxtemp[i]== True:
                    Itempval[i] = Itemp[i]                
                    gkrval[i] = gkrtemp[i] 
            residuleval = Itempval-I0temp
            
            plt.figure()
            plt.plot(Itempcorr,'r-')
            plt.plot(I0temp,'g--')
            plt.plot(Itempval, 'b*')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation6')

            plt.figure()
            plt.plot(residule,'r')
            plt.plot(residuleval, 'g')
            plt.title(str(i)+' '+'th'+' '+'I reconstruction for validation6')
        
        Ikr.append(Itempval)
        Gkr.append(gkrval)
        Residule.append(residuleval)
        #
    return [IkrO, GkrO, ResiduleO, Ikr, Gkr, Residule]        

def Sensitivity(B1, B2):#B1: initiate, #B2: scaled
    dBMax = np.max([np.max(B1, 1), np.max(B2, 1)], 0)
    dBMin = np.min([np.min(B1, 1), np.min(B2, 1)], 0)
    S = (B2 -B1)/np.max(abs(dBMax)-abs(dBMin))
    return S

def main():
    E0 = pd.read_csv('D:/ECG/herg25oc1-staircaseramp-A04.csv', header = [0]) 
    E1 = pd.read_csv('D:/ECG/herg25oc1-staircaseramp-A01-after.csv', header = [0])
    E1Arr = np.array(E1, dtype = float)
    plt.figure()
    plt.plot(E1Arr)
    E1f = np.fft.fft(E1Arr, n = 2)
    E1f = np.mean(E1f, 1)    
    E1f = E1f-np.mean(E1f)
    plt.figure()    
    plt.plot(E1f)    
    E1_cor = np.fft.ifft(E1f)
    plt.figure()
    plt.plot(E1_cor)

    #check I
    a = 1
    i_f = 0.6
    i_s = 1-i_f
    xr_f = 0
    xr_s = 0.6
    g = 1
    gto = 0.192
    gkr = 0.0342
    gks = 0.0029
    gsus = 0.0301
    gk1 = 0.04555
    gf_Na = 0.0116
    gf_K = 0.0232
    fs = 4596 #Hz
#    V = E0_cor[0:fs]
#    V = np.real(E1_cor[0:round(fs/2)])
    V = E1Arr[0:round(fs/2)]
   
    
    #IKr Block
    CKpin = 110 #mM
    CKpout = 4  #mM
    CNapin = 8.23
    CNapout = 140
    Ca2p_sl = 1e-4
    T = 300.15 #K
    z = 1# fixed for K+
    EK = E(CKpin, CKpout, T, z)
    ENa = E(CNapin, CNapout, T, z)    
    Ito = I_to(V, a, i_s, i_f, g, gto, EK)    
    Isus = I_sus(V, gsus, EK)
    Ikr = I_Kr(V, CKpout, xr_s, xr_f, gkr, EK) #outward
    Iks =I_Ks(V, gks, Ca2p_sl, EK)
    If = I_f(V, gf_K, gf_Na, EK, ENa)
    Ik1 = I_K1(V, gk1, CKpout, EK) #inward
    I_stim = -40 #muA/muF
    dV = V[1:len(V)]-V[0:len(V)-1]
    dVtemp = []
    dVtemp.append(np.mean(V[1:len(V)])-np.mean(V[0:len(V)-1]))
    for dv in dV:
        dVtemp.append(dv)
    dV = np.array(dVtemp)
    dt = 1/fs
    Cm = -(Ik1+Ito+Isus+I_stim)*dt/(dV+0.00001)

    tp = [10, 9000, 15000, 68600]
    N = 1400
    states = [1, 4, 2, 3]
    drifts = [1100, -4500, -1000, 800]
    #calibrate and validate
    N = 1400
    I0 =  (E1Arr).ravel()
    calib = Calibration(V, fs)
    dVmax = 160
    p1 = 1/(450*dt)
#    p2 = 1/40/dVmax
    p2 = -1/40
    p3 = 1/(500*dt)
    p4 = -1/160
    p5 = 1/(500*dt)
    p6 = 1/120
    p7 = 1/(500*dt)
    p8 = -1/120
    output = Calibrate_Validate(I0, calib, gkr0, tp, N, states, drifts, dt, p1, p2, p3, p4, p5, p6, p7, p8, EK)
    IkrO = output[0]
    GkrO = output[1]
    ResiduleO = output[2]
    Ikr = output[0]
    Gkr = output[1]
    Residule = output[2]
    #simulation and retest the parameters for ion channel models
    #activate and recovery (do not consider after using E-4031 to eliminate IKr first doing correction)
    steps = 50
    N1 = 600
    N2 = 1500
    Vsim = []
    Isim = []
    Vsim.append(calib[899:900+N2])   
    Vsim.append(calib[249:250+N1])
    Isim.append(I0[4400:4400+N2])
    Isim.append(I0[100:100+N1])             
    EXP = np.exp(Isim)
    A = np.sort(np.linspace(min(I0), max(I0), N))
    B = np.sort(np.linspace(min(EXP), max(EXP), N)/max(EXP)/10)
#    dVsim = Vsim[1:len(Vsim)]-Vsim[0:len(Vsim)-1]
    dVsim = []
    for i in range(len(Vsim)): 
        dVsim.append(Vsim[i][1:len(Vsim[i])]-Vsim[i][0:len(Vsim[i])-1])                     
    Istim = 100 #pA
#    data is annotated when Cm = -(Ik1+Ito+Isus+I_Kr+I_stim)*dt/(dV+0.00001), I_others = 0
    sc = np.linspace(0.5, 1.5, 5)
    gk1sim = gk1*sc
    gtosim = gto*sc
    gsussim = gsus*sc
    gkrsim = gkr*sc
    gkssim = gks*sc
    
    Ito_0 = I_to(Vsim[0][1:len(Vsim[0])], a, i_s, i_f, g, gto, EK)    
    Isus_0 = I_sus(Vsim[0][1:len(Vsim[0])], gsus, EK)
    Ikr_0 = I_Kr(Vsim[0][1:len(Vsim[0])], CKpout, xr_s, xr_f, gkr, EK) #outward
    Iks_0 =I_Ks(Vsim[0][1:len(Vsim[0])], gks, Ca2p_sl, EK)
    Ik1_0 = I_K1(Vsim[0][1:len(Vsim[0])], gk1, CKpout, EK) #inward
    Ito_1 = I_to(Vsim[1][1:len(Vsim[1])], a, i_s, i_f, g, gto, EK)    
    Isus_1 = I_sus(Vsim[1][1:len(Vsim[1])], gsus, EK)
    Ikr_1 = I_Kr(Vsim[1][1:len(Vsim[1])], CKpout, xr_s, xr_f, gkr, EK) #outward
    Iks_1 =I_Ks(Vsim[1][1:len(Vsim[1])], gks, Ca2p_sl, EK)
    Ik1_1 = I_K1(Vsim[1][1:len(Vsim[1])], gk1, CKpout, EK) #inward    
    Cm0 = -(Ik1_0+Ito_0+Isus_0+np.array(Isim[0]))*dt/(np.array(dVsim[0])+0.00001)
    Cm1 = -(Ik1_1+Ito_1+Isus_1+np.array(Isim[1])+Istim)*dt/(np.array(dVsim[1])+0.00001)
    Lsigma = np.cov([np.ones((np.size(Ik1_0sim))),np.ones((np.size(Ik1_0sim))),np.ones((np.size(Ik1_0sim))),np.ones((np.size(Ik1_0sim)))])
    L = np.random.multivariate_normal([0,0,0,0], Lsigma, (np.size(Ik1_0sim), np.size(Ik1_0sim)))
    Params = []
    idx = 0
    for i in range(len(sc)):
            Ito_0sim = I_to(Vsim[0][1:len(Vsim[0])], a, i_s, i_f, g, gtosim[i], EK)    
            Isus_0sim = I_sus(Vsim[0][1:len(Vsim[0])], gsussim[i], EK)
#            Ikr_0 = I_Kr(Vsim[0][1:len(Vsim[0])], CKpout, xr_s, xr_f, gkrsim[i], EK) #outward
            Iks_0sim =I_Ks(Vsim[0][1:len(Vsim[0])], gkssim[i], Ca2p_sl, EK)
            Ik1_0sim = I_K1(Vsim[0][1:len(Vsim[0])], gk1sim[i], CKpout, EK) #inward
            
#            sci.norm.pdf(np.array(Ito_0sim)) 
            countIto0, centerIto0 = np.histogram(Ito_0sim, bins = 5)
            countIsus0, centerIsus0 = np.histogram(Isus_0sim, bins = 5)
            countIks0, centerIks0 = np.histogram(Iks_0sim, bins = 5)
            countIk10, centerIk10 = np.histogram(Ik1_0sim, bins = 5)
            Mu = [np.mean(Ito_0sim) , np.mean(Isus_0sim), np.mean(Iks_0sim), np.mean(Ik1_0sim)]
            SIGMA = np.cov([Ito_0sim , Isus_0sim, Iks_0sim, Ik1_0sim])
            Distr1 = np.random.normal(np.mean(Ito_0sim), np.std(Ito_0sim), np.size(Ito_0sim))*np.random.normal(np.mean(Isus_0sim), np.std(Isus_0sim), np.size(Isus_0sim))* np.random.normal(np.mean(Iks_0sim), np.std(Iks_0sim), np.size(Iks_0sim))*np.random.normal(np.mean(Ik1_0sim), np.std(Ik1_0sim), np.size(Ik1_0sim))
            Distr2 = np.random.multivariate_normal(Mu, SIGMA, (np.size(Ik1_0sim), np.size(Ik1_0sim)))*np.std(Iks_0sim)*np.std(Ik1_0sim)*np.std(Isus_0sim)*np.std(Ito_0sim)
            Ltemp = np.dot(Distr1,Distr2)
            if np.mean(Ltemp)/np.std(Ltemp) > np.mean(L)/np.std(L):
#            if np.mean(Isim[0] - Ikr_0sim) < np.mean(Res_0sim):
                L = Ltemp 
                idx = i
    Params = [sc[idx], gtosim[idx], gsussim[idx], gkssim[idx], gk1sim[idx]]
    Ikr_0sim = Cm0*(np.array(dVsim[0])+0.00001)/dt-Ito_0sim-Isus_0sim-Iks_0sim-Ik1_0sim
    Cm0sim = Cm0/(Ikr_0sim-Isim[0])
    Ikr_0simcor = Ikr_0sim*Cm0sim/(Ikr_0sim-Isim[0])
    Ikr_0simcorB = Ikr_0simcor-np.mean(Ikr_0simcor)

    plt.figure()
    plt.plot(Isim[0],'r-')
    plt.plot(Ikr_0,'g*')
    plt.plot(Ikr_0simcorB, 'b--')
    plt.legend(['origin', 'initiate', 'first simulation under best parameter'])
    plt.title('gkr:'+str(gkrsim[idx])+' '+'gives the best simulation of Ikr recovery I')
    
    plt.figure()
    plt.plot(abs(Ikr_0 - Isim[0]),'r')
    plt.plot(abs(Ikr_0simcorB - Isim[0])/np.mean(abs(Ikr_0simcorB - Isim[0])), 'g')  
    plt.legend(['Residule of initiate', ' Residule of first simulation under best parameter'])
    plt.title('accordding residule')

    Ito_0simO = Ito_0sim   
    Isus_0simO = Isus_0sim
    Iks_0simO = Iks_0sim
    Ik1_0simO = Ik1_0sim
    Ikr_0simO = Ikr_0sim    

    Res_Ikr_0sim = [] 
    Res_Ito_0sim = []
    Res_Isus_0sim = [] 
    Res_Iks_0sim = []    
    Res_Ik1_0sim = []
    Res_Ikr_0sim.append(np.mean(abs(Ikr_0simO - Isim[0])/np.mean(abs(Ikr_0simO - Isim[0]))))
    Res_Ito_0sim.append(np.mean(abs(Ito_0simO - Ito_0)/np.mean(abs(Ito_0simO - Ito_0))))
    Res_Isus_0sim.append(np.mean(abs(Isus_0simO - Isus_0)/np.mean(abs(Isus_0simO - Isus_0))))
    Res_Iks_0sim.append(np.mean(abs(Iks_0simO - Iks_0)/np.mean(abs(Iks_0simO - Iks_0))))
    Res_Ik1_0sim.append(np.mean(abs(Ik1_0simO - Ik1_0)/np.mean(abs(Ik1_0simO - Ik1_0))))

#    steps = 100
    N = 15
    c = 0
    NIto = np.size(Res_Ito_0sim)
    NIsus = np.size(Res_Isus_0sim)
    NIks = np.size(Res_Iks_0sim) 
    NIkr = np.size(Res_Ikr_0sim) 
    NIk1 = np.size(Res_Ik1_0sim)         
    while (NIk1 <= N | NIto <= N | NIsus <= N | NIks <= N | NIkr <= N) & steps >=0 & c>-20:
        Ik1_0sim_n = Ik1_0sim + np.random.normal(0, 1, np.size(Ik1_0sim))
        Ik1_0sim_sc = Cm0/(Ik1_0sim_n-Ik1_0)
        Ik1_0simcor = Ik1_0sim_n*Ik1_0sim_sc/(Ik1_0sim_n-Iks_0)        
        Ik1_0simcorB = Ik1_0simcor-np.mean(Iks_0simcor)
        if (np.mean(abs(Ik1_0simcorB - Ik1_0)/np.mean(abs(Ik1_0simcorB - Ik1_0))) <= Res_Ik1_0sim[-1])  & (sum(Ik1_0simcorB != Ik1_0sim)>0):
            Res_Ik1_0sim.append(np.mean(abs(Ik1_0simcorB - Ik1_0)/np.mean(abs(Ik1_0simcorB - Ik1_0))))        
            Ik1_0sim = Ik1_0simcorB
            c += 1

        Ito_0sim_n = Ito_0sim + np.random.normal(0, 1, np.size(Ito_0sim))
        Ito_0sim_sc = Cm0/(Ito_0sim_n-Ito_0)
        Ito_0simcor = Ito_0sim_n*Ito_0sim_sc/(Ito_0sim_n-Ito_0)        
        Ito_0simcorB = Ito_0simcor-np.mean(Ito_0simcor)
        if (np.mean(abs(Ito_0simcorB - Ito_0)/np.mean(abs(Ito_0simcorB - Ito_0))) <= Res_Ito_0sim[-1]) & (sum(Ito_0simcorB != Ito_0sim)>0):
            Res_Ito_0sim.append(np.mean(abs(Ito_0simcorB - Ito_0)/np.mean(abs(Ito_0simcorB - Ito_0))))        
            Ito_0sim = Ito_0simcorB
            c  += 1

        Isus_0sim_n = Isus_0sim + np.random.normal(0, 1, np.size(Isus_0sim))
        Isus_0sim_sc = Cm0/(Isus_0sim_n-Isus_0)
        Isus_0simcor = Isus_0sim_n*Isus_0sim_sc/(Isus_0sim_n-Isus_0)        
        Isus_0simcorB = Isus_0simcor-np.mean(Isus_0simcor)
        if (np.mean(abs(Isus_0simcorB - Isus_0)/np.mean(abs(Isus_0simcorB - Isus_0))) <= Res_Isus_0sim[-1]) & (sum(Isus_0simcorB != Isus_0sim)>0):
            Res_Isus_0sim.append(np.mean(abs(Isus_0simcorB - Isus_0)/np.mean(abs(Isus_0simcorB - Isus_0))))        
            Isus_0sim = Isus_0simcorB
            c += 1
        
        Iks_0sim_n = Iks_0sim + np.random.normal(0, 1, np.size(Iks_0sim))
        Iks_0sim_sc = Cm0/(Iks_0sim_n-Iks_0)
        Iks_0simcor = Iks_0sim_n*Iks_0sim_sc/(Iks_0sim_n-Iks_0)        
        Iks_0simcorB = Iks_0simcor-np.mean(Iks_0simcor)
        if (np.mean(abs(Iks_0simcorB - Iks_0)/np.mean(abs(Iks_0simcorB - Iks_0))) <= Res_Iks_0sim[-1])  & (sum(Iks_0simcorB != Iks_0sim)>0):
            Res_Iks_0sim.append(np.mean(abs(Iks_0simcorB - Iks_0)/np.mean(abs(Iks_0simcorB - Iks_0))))        
            Iks_0sim = Iks_0simcorB
            c += 1
            
        Ikr_0sim_n = Ikr_0sim + np.random.normal(0, 1, np.size(Ikr_0sim))
        Ikr_0sim_sc = Cm0/(Ikr_0sim_n-Ikr_0)
        Ikr_0simcor = Ikr_0sim_n*Ikr_0sim_sc/(Ikr_0sim_n-Ikr_0)        
        Ikr_0simcorB = Ikr_0simcor-np.mean(Ikr_0simcor)
        if (np.mean(abs(Ikr_0simcorB - Ikr_0)/np.mean(abs(Ikr_0simcorB - Ikr_0))) <= Res_Ikr_0sim[-1])  & (sum(Ikr_0simcorB != Ikr_0sim)>0):
            Res_Ikr_0sim.append(np.mean(abs(Ikr_0simcorB - Ikr_0)/np.mean(abs(Ikr_0simcorB - Ikr_0))))        
            Ikr_0sim = Ikr_0simcorB
            c += 1
        c -= 1
        NIto = np.size(Res_Ito_0sim)
        NIsus = np.size(Res_Isus_0sim)
        NIks = np.size(Res_Iks_0sim) 
        NIkr = np.size(Res_Ikr_0sim)
       
        steps-= 1
        
        
        

        dVmaxdt = []
        Sensitivity = [] 
 
        Res_ks_0sim.append(abs(Iks_0simcorB - Iks_0)/np.mean(abs(Iks_0simcorB - Iks_0)))        

        Ik1_0sim_n = Ik1_0sim + np.random.normal(0, 1, np.size(Ik1_0sim))
        Ik1_0sim_sc = Cm0/(Ik1_0sim_n-Ik1_0)
        Ik1_0simcor = Ik1_0sim_n*Ik1_0sim_sc/(Ik1_0sim_n-Ik1_0)        
        Ik1_0simcorB = Ik1_0simcor-np.mean(Ik1_0simcor)
        Res_Ik1_0sim.append(abs(Ik1_0simcorB - Ik1_0)/np.mean(abs(Ik1_0simcorB - Ik1_0)))        

        
    Ito_t0 = Ito_0simcorB
    Isus_t0 = Isus_0simcorB
    Iks_t0 = Iks_0simcorB
    Ikr_t0 = Ikr_0simcorB
    Ik1_t0 = Ik1_0simcorB
    dVmax = -(Ik1_t+Ito_t+Isus_t+Ikr_t+Iks_t)/Cm0
    Res_Ik1_t0 = Res_Ik1_0sim[0:14]
    Res_Ito_t0 = Res_Ito_0sim[0:14]
    Res_Isus_t0 = Res_Isus_0sim[0:14]
    Res_Iks_t0 = Res_Iks_0sim[0:14]    
    Res_Ikr_t0 = Res_Ikr_0sim[0:14]
    # Validate 3: DAD
    Ito_DAD = gtosim[idx]*(DAD-EK)    
    Isus_DAD = gsussim[idx]*(DAD-EK)
    Iks_DAD = gkssim[idx]*(DAD-EK)
    Ikr_DAD = gkrsim[idx]*(DAD-EK)
    Ik1_DAD = gk1sim[idx]*(DAD-EK)
    
    Ito_DAD = DAD- Ito_0simO/gtosim[idx]    
    Isus_DAD = gsussim[idx]*(DAD-EK)
    Iks_DAD = gkssim[idx]*(DAD-EK)
    Ikr_DAD = gkrsim[idx]*(DAD-EK)
    Ik1_DAD = gk1sim[idx]*(DAD-EK)
    # Validate 4: EAD
    Ito_EAD = gtosim[idx]*(EAD-EK)
    Isus_EAD = gsussim[idx]*(EAD-EK)
    Iks_EAD = gkssim[idx]*(EAD-EK)
    Ikr_EAD = gkrsim[idx]*(EAD-EK)
    Ik1_EAD = gk1sim[idx]*(EAD-EK)    
    # Validate 4: AP   
    Ito_AP = gtosim[idx]*(AP-EK)    
    Isus_AP = gsussim[idx]*(AP-EK)
    Iks_AP = gkssim[idx]*(AP-EK)
    Ikr_AP = gkrsim[idx]*(AP-EK)
    Ik1_AP = gk1sim[idx]*(AP-EK)

    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(DAD) 
    plt.title('protocal DAD') 
    plt.subplot(3,2,2)
    plt.plot(Ito_DAD)
    plt.ylabel('Gto') 
    plt.subplot(3,2,3)
    plt.plot(Isus_DAD)
    plt.ylabel('Gsus')
    plt.subplot(3,2,4)     
    plt.plot(Iks_DAD)
    plt.ylabel('Gks')
    plt.subplot(3,2,5)
    plt.plot(Ikr_DAD)
    plt.ylabel('Gkr')
    plt.subplot(3,2,6)
    plt.plot(Ik1_DAD)    
    plt.ylabel('Gk1')  
    
    plt.figure()    
    plt.subplot(3,2,1)
    plt.plot(EAD) 
    plt.title('protocal EAD') 
    plt.subplot(3,2,2)
    plt.plot(Ito_EAD)
    plt.ylabel('Gto') 
    plt.subplot(3,2,3)
    plt.plot(Isus_EAD)
    plt.ylabel('Gsus')
    plt.subplot(3,2,4)     
    plt.plot(Iks_EAD)
    plt.ylabel('Gks')
    plt.subplot(3,2,5)
    plt.plot(Ikr_EAD)
    plt.ylabel('Gkr')
    plt.subplot(3,2,6)
    plt.plot(Ik1_EAD)    
    plt.ylabel('Gk1') 

    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(AP) 
    plt.title('protocal AP') 
    plt.subplot(3,2,2)
    plt.plot(Ito_AP)
    plt.ylabel('Gto') 
    plt.subplot(3,2,3)
    plt.plot(Isus_AP)
    plt.ylabel('Gsus')
    plt.subplot(3,2,4)     
    plt.plot(Iks_AP)
    plt.ylabel('Gks')
    plt.subplot(3,2,5)
    plt.plot(Ikr_AP)
    plt.ylabel('Gkr')
    plt.subplot(3,2,6)
    plt.plot(Ik1_AP)    
    plt.ylabel('Gk1') 
    
    
    plt.figure()
    plt.subplot(5,2,1)
    plt.plot(Ito_t0[0:200])
    plt.ylabel('Ito')
    plt.subplot(5,2,2)
    plt.plot(Ito_t0[0:400])
    plt.ylabel('Ito')
    
    
    plt.subplot(5,2,3)
    plt.plot(Isus_t0[0:200])
    plt.ylabel('Isus')
    plt.subplot(5,2,4)
    plt.plot(Isus_t0[0:400])
    plt.ylabel('Isus') 
    
    plt.subplot(5,2,5)
    plt.plot(Iks_t0[0:200])
    plt.ylabel('Iks')     
    plt.subplot(5,2,6)
    plt.plot(Iks_t0[0:400])
    plt.ylabel('Iks')   
    
    plt.subplot(5,2,7)
    plt.plot(Ikr_t0[0:200])
    plt.ylabel('Ikr')     
    plt.subplot(5,2,8)
    plt.plot(Ikr_t0[0:400])
    plt.ylabel('Ikr')
    
    plt.subplot(5,2,9)
    plt.plot(Ik1_t0[0:200])
    plt.ylabel('Ik1')     
    plt.subplot(5,2,10)
    plt.plot(Ik1_t0[0:400])
    plt.ylabel('Ik1')
    
    
    plt.figure()
    plt.subplot(5,3,1)
    plt.plot(Ito_t0[0:200])
    plt.ylabel('Ito')
    plt.subplot(5,3,2)
    plt.plot(Ito_t0[0:200])
    plt.ylabel('Ito')     
    
    plt.subplot(5,3,3)
    plt.plot(Isus_t0[0:200])
    plt.ylabel('Isus')     
    plt.subplot(5,3,4)
    plt.plot(Isus_t0[0:400])
    plt.ylabel('Isus') 
    
    plt.subplot(5,2,5)
    plt.plot(Iks_t0[0:200])
    plt.ylabel('Iks')     
    plt.subplot(5,2,6)
    plt.plot(Iks_t0[0:400])
    plt.ylabel('Iks')   
    
    plt.subplot(5,2,7)
    plt.plot(Ikr_t0[0:200])
    plt.ylabel('Ikr')     
    plt.subplot(5,2,8)
    plt.plot(Ikr_t0[0:400])
    plt.ylabel('Ikr')
    
    plt.subplot(5,2,9)
    plt.plot(Ik1_t0[0:200])
    plt.ylabel('Ik1')     
    plt.subplot(5,2,10)
    plt.plot(Ik1_t0[0:400])
    plt.ylabel('Ik1')    
    
    

#    dVmaxdt =       

    mean_Res_Ik1 = []
    mean_Res_Ito = []
    mean_Res_Isus = []  
    mean_Res_Iks = []
    mean_Res_Ikr = []
    std_Res_Ik1 = []
    std_Res_Ito = []
    std_Res_Isus = []  
    std_Res_Iks = []
    std_Res_Ikr = []
    CV_Res_Ik1 = []
    CV_Res_Ito = []
    CV_Res_Isus = []  
    CV_Res_Iks = []
    CV_Res_Ikr = []
    
    for i in range(1,15):
        mean_Res_Ik1.append(np.mean(Res_Ik1_t0[0:i]))
        mean_Res_Ito.append(np.mean(Res_Ito_t0[0:i]))
        mean_Res_Isus.append(np.mean(Res_Isus_t0[0:i]))  
        mean_Res_Iks.append(np.mean(Res_Iks_t0[0:i]))
        mean_Res_Ikr.append(np.mean(Res_Ikr_t0[0:i]))
        std_Res_Ik1.append(np.std(Res_Ik1_t0[0:i]))
        std_Res_Ito.append(np.std(Res_Ito_t0[0:i]))
        std_Res_Isus.append(np.std(Res_Isus_t0[0:i])) 
        std_Res_Iks.append(np.std(Res_Iks_t0[0:i]))
        std_Res_Ikr.append(np.std(Res_Ikr_t0[0:i]))
        CV_Res_Ik1.append(mean_Res_Ik1[-1]/(std_Res_Ik1[-1]+0.000001))
        CV_Res_Ito.append(mean_Res_Ito[-1]/(std_Res_Ito[-1]+0.000001))
        CV_Res_Isus.append(mean_Res_Isus[-1]/(std_Res_Isus[-1]+0.000001)) 
        CV_Res_Iks.append(mean_Res_Iks[-1]/(std_Res_Iks[-1]+0.000001))
        CV_Res_Ikr.append(mean_Res_Ikr[-1]/(std_Res_Ikr[-1]+0.000001))        
        

    Sensitivity = Sensitivity(np.array([Ito_0, Isus_0, Iks_0, Ik1_0, Ikr_0]), np.array([Ito_0simO, Isus_0simO, Iks_0simO, Ik1_0simO, Ikr_0simO]))


    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(mean_Res_Ik1) 
    plt.title('Ik1')
    plt.ylabel('Res_mean')
    plt.subplot(2,2,2)
    plt.plot(std_Res_Ik1)
    plt.ylabel('Res_std') 
    plt.subplot(2,2,3)
    plt.plot(CV_Res_Ik1)
    plt.ylabel('Res_CV')
    plt.subplot(2,2,4)     
    plt.plot(Sensitivity[3,:])
    plt.ylabel('sensitivity')
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(mean_Res_Ito) 
    plt.title('Ito')
    plt.ylabel('Res_mean')
    plt.subplot(2,2,2)
    plt.plot(std_Res_Ito)
    plt.ylabel('Res_std') 
    plt.subplot(2,2,3)
    plt.plot(CV_Res_Ito)
    plt.ylabel('Res_CV')
    plt.subplot(2,2,4)     
    plt.plot(Sensitivity[0,:])
    plt.ylabel('sensitivity')

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(mean_Res_Iks) 
    plt.title('Iks')
    plt.ylabel('Res_mean')
    plt.subplot(2,2,2)
    plt.plot(std_Res_Iks)
    plt.ylabel('Res_std') 
    plt.subplot(2,2,3)
    plt.plot(CV_Res_Iks)
    plt.ylabel('Res_CV')
    plt.subplot(2,2,4)     
    plt.plot(Sensitivity[2,:])
    plt.ylabel('sensitivity')
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(mean_Res_Ikr) 
    plt.title('Isus')
    plt.ylabel('Res_mean')
    plt.subplot(2,2,2)
    plt.plot(std_Res_Isus)
    plt.ylabel('Res_std') 
    plt.subplot(2,2,3)
    plt.plot(CV_Res_Isus)
    plt.ylabel('Res_CV')
    plt.subplot(2,2,4)     
    plt.plot(Sensitivity[1,:])
    plt.ylabel('sensitivity')

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(mean_Res_Ikr) 
    plt.title('Ikr')
    plt.ylabel('Res_mean')
    plt.subplot(2,2,2)
    plt.plot(std_Res_Ikr)
    plt.ylabel('Res_std') 
    plt.subplot(2,2,3)
    plt.plot(CV_Res_Ikr)
    plt.ylabel('Res_CV')
    plt.subplot(2,2,4)     
    plt.plot(Sensitivity[4,:])
    plt.ylabel('sensitivity')

    muIkr_0sim = np.mean(Ikr_0sim)
    sigmaIkr_sim = np.std(Ikr_0sim)
#    plt.plot(Res_0sim,sci.norm.interval(alpha=0.05) )
    count0, center0 = np.histogram(Ikr_0sim)
    p0 = 1/1+np.exp(-count0/sum(count0))

    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Ik1_0sim, density = True)
    plt.title('Ik1')
    plt.subplot(2,1,2)
    plt.hist(Sensitivity[3,], density = True)
    plt.title('Ik1 Sensitivity')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Ito_0sim, density = True)
    plt.title('Ito')
    plt.subplot(2,1,2)
    plt.hist(Sensitivity[0,], density = True)
    plt.title('Ito Sensitivity')  
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Iks_0sim, density = True)
    plt.title('Iks')
    plt.subplot(2,1,2)
    plt.hist(Sensitivity[2,], density = True)
    plt.title('Iks Sensitivity')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Ikr_0sim, density = True)
    plt.title('Ikr')
    plt.subplot(2,1,2)
    plt.hist(Sensitivity[4,], density = True)
    plt.title('Ikr Sensitivity')   
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Isus_0sim, density = True)
    plt.title('Isus')
    plt.subplot(2,1,2)
    plt.hist(Sensitivity[1,], density = True)
    plt.title('Isus Sensitivity')      
    
    
    #EAD, BAD
    
    
    
    
    