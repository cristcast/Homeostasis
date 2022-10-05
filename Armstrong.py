#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:58:35 2022

@author: cristcast
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec

## FUNCIONES ##
def area(volumen):   # Cálculo de superficie de esfera [cm^2] a partir del volumen [lts]
    r = np.cbrt((volumen*0.001)*0.75/np.pi)*100 # lts to m^3 then radio [cm]
    area_cell = 4*np.pi*r**2 #[cm^2]
    return area_cell

def vol_cell(kin,nain,clin): # Cálculo de volumen de célula para equilibrio osmótico
    return(kin+nain+clin+Si_mol)/(K_o+Na_o+Cl_o)  # [Lts] [mol/(mol/lts)]

def Ipump_fun(Na_in):  # Cálculo de actividad de la bomba Na/K
    if pumpnak_bool:
        Pump_Na = 2.17*ATP/((1+Na_c/Na_in)**3)   #[mol/(s cm2)]
        Pump_K  = -Pump_Na/Pump_ratio            #[mol/(s cm2)]
        Ipump_t = (Pump_Na+Pump_K)*F             #[C/scm2]
        return(Pump_Na, Pump_K, Ipump_t) 
    else:
        return(0, 0, 0) 
           

def Pot_m(Ki,Nai,Cli):  # Cálculo de potencial de membrana
    top   = P_Nar*Na_o + P_Kr*K_o + P_Clr*Cli - scaleIpump*Ipump_fun(Nai)[2]/F   #[M]
    under = P_Nar*Nai  + P_Kr*Ki  + P_Clr*Cl_o                                   #[M]
    V_m   = ((R*T)/F)*np.log(top/under)                                          #[V]
    return(V_m)

def EDO_armstrong(yy, t):
    dy = [0, 0, 0]   
    k_i_mol  = yy[0]
    na_i_mol = yy[1]
    cl_i_mol = yy[2]
    
    if OSMOSIS: 
        Vol_t = vol_cell(k_i_mol,na_i_mol,cl_i_mol)  #Lts
    else:
        Vol_t = Vol_0
    
    # Calcula las nuevas concentraciones 
    K_i  = k_i_mol/Vol_t   #[M]
    Na_i = na_i_mol/Vol_t  #[M]
    Cl_i = cl_i_mol/Vol_t  #[M]
    
    # Pump Na/K
    Pump_Na, Pump_K, Ipump_t = Ipump_fun(Na_i)
    #Pump_Na = 2.17*ATP/((1+Na_c/Na_i)**3)   #[mol/(s cm2)]
    #Pump_K  = -Pump_Na/Pump_ratio           #[mol/(s cm2)]
    #Ipump_t = (Pump_Na+Pump_K)*F            #[C/s <cm2]
    
    # Potencial de membrana
    top   = P_Nar*Na_o + P_Kr*K_o + P_Clr*Cl_i - scaleIpump*Ipump_t/F #[M]
    under = P_Nar*Na_i + P_Kr*K_i + P_Clr*Cl_o                  #[M]
    V_m   = ((R*T)/F)*np.log(top/under)                         #[V]
    
    # Canales H, K, Na, Ca
    #      (1/0)       [cm/s]   1     [M]      [1]  (V/V)               [M]      [1e-3]  
    I_K  = ch_k_bool  * P_K  * z_K  * (K_i  * np.exp(V_m*z_K*F/(R*T)) - K_o)  * ltstocm3 #[mol/(s cm2)]
    I_Na = ch_na_bool * P_Na * z_Na * (Na_i * np.exp(V_m*z_Na*F/(R*T))- Na_o) * ltstocm3 #[mol/(s cm2)]
    I_Cl = ch_cl_bool * P_Cl * z_Cl * (Cl_o * np.exp(V_m*z_Cl*F/(R*T))- Cl_i) * ltstocm3 #[mol/(s cm2)
    
    # Diferenciales
           #Densidad de corriente   Área cell (¿Vol_0  o Vol_t?)
           #mol/(cm2*s)                (cm2)
    dKdt = (I_K + Pump_K)         * area(Vol_0)   #[mol/s]  
    dNadt = (I_Na + Pump_Na)      * area(Vol_0)   #[mol/s]
    dCldt = I_Cl                  * area(Vol_0)   #[mol/s]
    
    # Retorno de diferenciales
    dy = [dKdt, dNadt, dCldt]
    #print(dy)
    return dy

def plot_result():
    # Figuras
    fig = plt.figure(tight_layout=True, figsize=(16, 16))
    gs = gridspec.GridSpec(4, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(K_final)
    ax.set_ylabel('$[K]_i [M]$', fontsize='large')
    ax.set_xlabel('Tiempo[s]', fontsize='large')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Na_final)
    ax.set_ylabel('$[Na+]_i [M]$', fontsize='large')
    ax.set_xlabel('Tiempo[s]', fontsize='large')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(Cl_final)
    ax.set_ylabel('$[Cl^-]_i [M]$', fontsize='large')
    ax.set_xlabel('Tiempo[s]', fontsize='large')

    ax = fig.add_subplot(gs[2, :])
    ax.plot(Vol_final)
    ax.set_ylabel('Volumen [lts]', fontsize='large')
    ax.set_xlabel('Tiempo[s]', fontsize='large')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(psi_final)
    ax.set_ylabel('$\Delta\psi_L$ mV', fontsize='large')
    ax.set_xlabel('Tiempo[s]', fontsize='large')


# --------------------------------------------
#   PARÁMETROS DEL MODELO  ARMSTRONG         #
# --------------------------------------------
# Concentraciones externas
Na_o = 0.145  # M (145mM) Armstrong(2003)   (0.01M)Astaburuaga(2019)
K_o  = 0.005  # M (5mM)   Armstrong(2003)   (0.145M)Astaburuaga(2019)
Cl_o = 0.150  # M (150mM) Armstrong(2003)   (0.01)Astaburuaga(2019)
# Concentraciones internas
Na_i = 0.010  # M  (10 mM)  Armstrong(2003)  (0.02M) Astaburuaga(2019)  Ishida(0.145)
K_i  = 0.140  # M  (140 mM) Armstrong(2003)  (0.05M) Astaburuaga(2019)  Ishida(0.005)
Cl_i = 0.007  # M  (7  mM)  Armstrong(2003)  (0.001M)Astaburuaga(2019)  Ishida (0.110)
S_i  = 0.143  # M
# Permeabilidades (Relativas)
P_K   = 1e-6
P_Nar = 0.02
P_Kr  = 1
P_Clr = 2.0   
P_Na  = P_Nar*P_K
P_Cl  = P_Clr*P_K
# Constantes
NA  = 6.02e+23
F   = 96485
R   = 8.314
T   = 298
NA  = 6.02e+23
# Geometrías del lisosoma
Vol_0 = 1.646365952e-16 #[lts]
S = 1.45267584e-08   #Superficie [cm^2]
cap = 1.45267584e-14
cap_0 = 1e-6
# Z
z_Cl = -1
z_Na = 1
z_K  = 1
# V_m
psi_out = 0
# PUMP Na/K
Ipump = 0
Pump_ratio = 3/2
Na_c = 0.3
ATP = 0.003
# OTROS
ltstocm3 = 1e-3
Si_mol  = S_i*Vol_0 #[mol]


##---------------------------------------------------#
##         CONFIGURACIÓN DEL MODELO
##---------------------------------------------------#
# Canales activos
pumpnak_bool = 1
ch_k_bool = 1
ch_na_bool = 1
ch_cl_bool = 1
OSMOSIS = True
# Tiempo de integración
tiempo = 2000
t_start = 0.0
t_end = float(tiempo)
N_steps = int(t_end)
t = np.linspace(t_start, t_end, N_steps+1)

#Dudas
scaleIpump = 0  #¿Escalar Ipump? 0 o 1000 (ver notebook)
Na_c = 0.3      # [M] concentración para la mitad de la ocupación máxima de un sitio de unión de [Na+] en la bomba. No se detalla su valor en el paper.

#Iniciales
Ipump_t = 0
Vol_t = 1.646365952e-16 
#Potencial inicial
top   = P_Nar*Na_o + P_Kr*K_o + P_Clr*Cl_i - scaleIpump*Ipump_t/F
under = P_Nar*Na_i + P_Kr*K_i + P_Clr*Cl_o
V_m   = (R*T)/F*np.log(top/under)
# Vector inicial
init_K = K_i*Vol_t       #[mol]
init_Na = Na_i*Vol_t     #[mol]
init_Cl = Cl_i*Vol_t     #[mol]
y_init = [init_K, init_Na, init_Cl]

print("******************************")
print("VALORES INICIALES")
print("******************************")
print("Moles K+  inicial:", init_K,  " [mol]")
print("Moles Na+ inicial:", init_Na, " [mol]")
print("Moles Cl- inicial:", init_Cl, " [mol]")
print("Volumen Inicial:"  , Vol_t, " [lts]")
print("Potencial inicial:", V_m, " [V]")

#---------------------------------------------------
#CALCULO CON ODEINT
y = odeint(EDO_armstrong, y_init, t, hmax=1e-2)
#---------------------------------------------------

# Arreglos finales
K_f  = y[:, 0]    #[mol]
Na_f = y[:, 1]    #[mol]
Cl_f = y[:, 2]    #[mol]
Vol_final = vol_cell(K_f,Na_f,Cl_f) #[Lts]
K_final   = K_f/Vol_final  #[M]
Na_final  = Na_f/Vol_final #[M]
Cl_final  = Cl_f/Vol_final #[M]
psi_final = Pot_m(K_final,Na_final,Cl_final)/1000  #mV

# Valores Finales
print("")
print("******************************")
print("VALORES INICIALES")
print("******************************")
print("Valor final [K+]_L: ", K_final[-1], "[M]")
print("Valor final [Na+]_L: ",Na_final[-1], "[M]")
print("Valor final [Cl-]_L: ", Cl_final[-1], "[M]")
print("Valor final Volumen: ", Vol_final[-1], "[lts]")
print("******************************")
print("")

plot_result()