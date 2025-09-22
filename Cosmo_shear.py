import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp2d


import logging
# Basic registry settings
logging.basicConfig(level=logging.INFO)

import Cosmo_integration as ci
import Interpolation as int

# Parametros fiduciales

Omega_b0_fid = 0.05
Omega_m0_fid = 0.32
h_fid = 0.67
ns_fid = 0.96
sigma8_fid = 0.816
Omega_DE0_fid = 0.68
w0_fid = -1.0
wa_fid = 0.0
gamma_fid = 0.55

#c = 9.72 * 10 ** (-15) # en Mpc # 300000 en km/s
Aia = 1.72
Cia = 0.0134
nia = -0.41
bia = 2.17


class CosmicShear:
    def __init__(self, cosmic_paramss):
        self.z = cosmic_paramss['z']
        self.l = cosmic_paramss['l']
        self.universe = cosmic_paramss['type']
        self.model = cosmic_paramss['model']
        self.IA = cosmic_paramss['IA']
        self.epsilon = cosmic_paramss['epsilon']
        self.Nz = cosmic_paramss['Nz']
        self.c = cosmic_paramss['c']
        self.sigma_epsilon = cosmic_paramss['sigma_epsilon']
        self.n_gal = cosmic_paramss['n_gal']

    def E2(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        if self.model == 'ACDM_flat':
            Omega_k0 = 0 
            Omega_DE0 = 1 - (Omega_m0)
            w0, wa = -1, 0
        elif self.model == 'ACDM_non_flat':
            Omega_k0 = 1 - (Omega_m0 + Omega_DE0)
            w0, wa = -1, 0
        elif self.model == 'non_ACDM_flat':
            Omega_k0 = 0 
            Omega_DE0 = 1 - (Omega_m0)
        elif self.model == 'non_ACDM_non_flat':
            Omega_k0 = 1 - (Omega_m0 + Omega_DE0)
        elif self.model == 'non_ACDM_flat_gamma':
            Omega_k0 = 0
            Omega_DE0 = 1 - (Omega_m0)
        elif self.model == 'non_ACDM_non_flat_gamma':
            Omega_k0 = 0
        radicando = (Omega_m0 * (1 + z) ** 3) + (Omega_DE0 * ((1 + z)**(3 * (1 + wa + w0))) * np.exp(-3 * wa * (z / (1 + z)))) + (Omega_k0 * (1 + z)**2)
        return np.sqrt(radicando)

    def inverse_E2(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        return 1 / self.E2(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)

    def r(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        '''
        In Mpc
        '''
        H_0 = (100 * h)
        z_prime = np.linspace(0, z, 30)
        delta = z_prime[1] - z_prime[0]
        integrand = self.inverse_E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) * delta
        return np.sum(integrand) * (self.c / H_0) 
    
    def r_w(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        H_0 = (100 * h)
        z_prime = np.linspace(0, z, 30)
        delta = z_prime[1] - z_prime[0]
        integrand = self.inverse_E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) * delta
        return np.sum(integrand) * (self.c / H_0) * (H_0 / self.c)

    def SN(self, i, j): # unidades de area sr
        if i == j:
            return ((self.sigma_epsilon ** 2) / (self.n_gal * ((60 * 180 / np.pi)**2) / self.Nz))
        else: 
            return 0

    def D(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        z_prime = np.linspace(0, z, 30)
        E_array = self.E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        Omega_m = (Omega_m0  * (1 + z_prime)**3) / (E_array**2)
        delta = z_prime[1] - z_prime[0]
        integral = np.sum(((Omega_m**gamma) / (1 + z_prime)) * delta)
        return np.exp(-integral)
    
    def l_to_k(self, l, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        rz = self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        k = (10**l + 0.5) / rz 
        return k 

    
    ###
    # P y sus derivadas

    interp_func = int.interpolation('pkz-Fiducial.txt')
    dP_dk_interp = int.dPdk_interpolation('pkz-Fiducial.txt')

    der_dic = {
        "Omega_m0": {
            "pl": int.interpolation("pkz-Om_pl_eps_1p3E-2.txt"),
            "mn": int.interpolation("pkz-Om_mn_eps_1p3E-2.txt"),
        },
        "h": {
            "pl": int.interpolation("pkz-h_pl_eps_1p3E-2.txt"),
            "mn": int.interpolation("pkz-h_mn_eps_1p3E-2.txt"),
        },
        "Omega_b0": {
            "pl": int.interpolation("pkz-Ob_pl_eps_1p3E-2.txt"),
            "mn": int.interpolation("pkz-Ob_mn_eps_1p3E-2.txt"),
        },
        "ns": {
            "pl": int.interpolation("pkz-ns_pl_eps_1p3E-2.txt"),
            "mn": int.interpolation("pkz-ns_mn_eps_1p3E-2.txt"),
        },
        "sigma8": {
            "pl": int.interpolation("pkz-s8_pl_eps_1p3E-2.txt"),
            "mn": int.interpolation("pkz-s8_mn_eps_1p3E-2.txt"),
        },
    }

    fiduciales = {'Omega_m0': Omega_m0_fid, 'h': h_fid, 'Omega_b0': Omega_b0_fid, 'ns': ns_fid, 'sigma8': sigma8_fid}


    # Luminosity
    Lumo = int.Lumo()

    def PK(self, z, k):
        lnP = self.interp_func(z, np.log(k), grid=False)
        return np.exp(lnP) 
    
    def PPS(self, z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        k = self.l_to_k(l, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) / h
        P = self.PK(z, k) / (sigma8 ** 2)
        if self.universe == 'standard':
            return P 
        else:
            D_0 = self.D(0, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            D_array = self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            return P*((D_array/D_0)**2)
    
    def der_PPS_parametro(self, z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro):
        k = self.l_to_k(l, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        P = self.PPS(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)

        def der_PPS_k(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
            der = self.dP_dk_interp(z, np.log(k / h)) 
            return der
        
        def der_P_parametro(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro):
            if parametro == "h":
                P_plus = self.der_dic[parametro]["pl"](z, np.log(k / h ), grid=False) 
                P_minus = self.der_dic[parametro]["mn"](z, np.log(k / h ), grid=False) 
                return P * (P_plus - P_minus) / (2 * self.epsilon * self.fiduciales[parametro])
            elif parametro in self.der_dic:
                P_plus = self.der_dic[parametro]["pl"](z, np.log(k * h_fid)) 
                P_minus = self.der_dic[parametro]["mn"](z, np.log(k * h_fid)) 
                return P * (P_plus - P_minus) / (2 * self.epsilon * self.fiduciales[parametro])
            else:
                return 0
        
        def der_k_parametro(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro):
            if parametro == "h":
                k_pl = self.l_to_k(l, z, Omega_m0, h * (1 + self.epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                k_mn = self.l_to_k(l, z, Omega_m0, h * (1 - self.epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                return (k_pl - k_mn) / (2 * self.epsilon * self.fiduciales[parametro])
            elif parametro == "Omega_m0":
                k_pl = self.l_to_k(l, z, Omega_m0* (1 + self.epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                k_mn = self.l_to_k(l, z, Omega_m0* (1 - self.epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                return (k_pl - k_mn) / (2 * self.epsilon * self.fiduciales[parametro])
            else:
                return 0
        
        first = der_P_parametro(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro)
        second = der_PPS_k(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        third = der_k_parametro(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro)

        return first + (second * third)
    ###
    def K(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
        H_0 = (100 * h)
        z_prime= self.z

        params = {'z': z_prime, 'model': self.model, 'c': self.c}
    
        A = ci.CosmoIntegration(params)

        E_array = self.E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        Wi = np.array(A.Window2(i, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
        n_i_array = np.array([A.n_i_try(i, zs) for zs in z_prime])
        Wj = np.array(A.Window2(j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
        n_j_array = np.array([A.n_i_try(j, zs) for zs in z_prime])
        r_array = np.array([A.r_w(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])

        operador1 = ((3/2 * Omega_m0 * (1+z_prime)) **2) * ((H_0 / self.c) ** 3)
        operador2 = 3/2 * Omega_m0 * (1+z_prime) * ( (H_0 / self.c) ** 3)
        operador3 =  (H_0 / self.c) ** 3

        K_gg = operador1 * (Wi * Wj) / (E_array)
        K_Ig = operador2 * ((n_i_array * Wj) + (n_j_array * Wi)) / (r_array) 
        K_II = operador3 * (n_i_array * n_j_array * E_array) / ((r_array) ** 2)

        return K_gg, K_Ig, K_II

    ###
    def operando(self, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
        if self.IA == True:
            z_prime = self.z
            D_array = np.array([self.D(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])
            F = ((1 + z_prime) ** nia) * (self.Lumo(z_prime) ** bia)
            op = (-(Aia * Cia * Omega_m0 * F) / D_array) 
            return op
        else:
            return 1

    def Ps(self, l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k):
        z_prime = self.z
        operando = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)

        def lambda_k(i): 
            L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
            lambda_min = np.log10(10**L_array[0])
            lambda_max = np.log10(10**L_array[-1])
            delta_lambda = (lambda_max - lambda_min) / len(L_array)
            lambda_k = lambda_min + (k - 1)*delta_lambda
            return lambda_k
        
        ls = lambda_k(k)

        P_gg = np.array([self.PPS(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for z_primes in z_prime])
        P_Ig = operando * P_gg
        P_II = (operando ** 2) * P_gg

        return P_gg, P_Ig, P_II


    
    def Cosmic_Shear(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
        z_prime = self.z 
        delta = z_prime[1] - z_prime[0]
        SNs = self.SN(i, j)

        K_gg, K_Ig, K_II = self.K(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        
        result = []

        for k, l in enumerate(self.l):
            P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)

            integrand_1 = (K_gg * P_gg) * float(delta)
            integrand_2 = (K_Ig * P_Ig) * float(delta)
            integrand_3 = (K_II * P_II) * float(delta)

            integral_1 =  np.sum(integrand_1)
            integral_2 =  np.sum(integrand_2)
            integral_3 =  np.sum(integrand_3)

            integral_final = integral_1 + integral_2 + integral_3 + SNs

            result.append(integral_final)

        return np.array(result)
    
    # ----- Derivadas de C respecto a los parámetros -----
    
    def Der_C_parametro(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, parametro):
        z_prime = self.z
        delta = z_prime[1] - z_prime[0]

        op1 = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        K_gg, K_Ig, K_II = self.K(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        result = []

        def der_K_parametro(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
            if parametro == 'h':
                K_gg_pl, K_Ig_pl, K_II_pl = self.K(i ,j, Omega_m0, h * (1 + self.epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
                K_gg_mn, K_Ig_mn, K_II_mn = self.K(i ,j, Omega_m0, h * (1 - self.epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)

                der_K_gg = (K_gg_pl - K_gg_mn) / (2*self.epsilon*h)
                der_K_Ig = (K_Ig_pl - K_Ig_mn) / (2*self.epsilon*h)
                der_K_II = (K_II_pl - K_II_mn) / (2*self.epsilon*h)
            elif parametro == 'Omega_m0':
                K_gg_pl, K_Ig_pl, K_II_pl = self.K(i ,j, Omega_m0 * (1 + self.epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
                K_gg_mn, K_Ig_mn, K_II_mn = self.K(i ,j, Omega_m0 * (1 - self.epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)

                der_K_gg = (K_gg_pl - K_gg_mn) / (2*self.epsilon*Omega_m0)
                der_K_Ig = (K_Ig_pl - K_Ig_mn) / (2*self.epsilon*Omega_m0)
                der_K_II = (K_II_pl - K_II_mn) / (2*self.epsilon*Omega_m0)
            else:
                der_K_gg, der_K_Ig, der_K_II = 0, 0, 0
            return der_K_gg, der_K_Ig, der_K_II
        
        def der_IA_parametro(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
            if parametro == 'Aia': # Estas derivadas dan lo mismo que lo analitico
                op1_pl = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia * (1 + self.epsilon), nia, bia)
                op1_mn = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia * (1 - self.epsilon), nia, bia)
                der_op1 = (op1_pl - op1_mn) / (2*self.epsilon*Aia)
            elif parametro == 'nia':
                op1_pl = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia , nia * (1 + self.epsilon), bia)
                op1_mn = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia , nia * (1 - self.epsilon), bia)
                der_op1 = (op1_pl - op1_mn) / (2*self.epsilon*nia)
            elif parametro == 'bia':
                op1_pl = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia , nia , bia * (1 + self.epsilon))
                op1_mn = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia , nia , bia * (1 - self.epsilon))
                der_op1 = (op1_pl - op1_mn) / (2*self.epsilon*bia)
            elif parametro == 'Omega_m0':
                op1_pl = self.operando(Omega_m0 * (1 + self.epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia , nia , bia)
                op1_mn = self.operando(Omega_m0 * (1 - self.epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia , nia , bia)
                der_op1 = (op1_pl - op1_mn) / (2*self.epsilon*Omega_m0)
            else:
                der_op1 = 0
            return der_op1
        
        der_op1 = der_IA_parametro(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        der_K_gg, der_K_Ig, der_K_II = der_K_parametro(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)

        result = []
        for k, l in enumerate(self.l):
            op1 = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)
            der_P_gg_parametro = np.array([self.der_PPS_parametro(zs, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for zs in z_prime])
            integrand_1 = ((der_K_gg * P_gg) + (K_gg * der_P_gg_parametro)) * float(delta)
            integrand_2 = ((der_K_Ig * P_Ig) + (K_Ig * ((der_op1 * P_gg) + (op1 * der_P_gg_parametro)))) * float(delta)
            integrand_3 = ((der_K_II * P_II) + (K_II * ((2*op1*der_op1*P_gg) + ((op1**2) * der_P_gg_parametro)))) * float(delta)

            integral_1 =  np.sum(integrand_1)
            integral_2 =  np.sum(integrand_2)
            integral_3 =  np.sum(integrand_3)


            integral_final = integral_1 + integral_2 + integral_3

            result.append(integral_final)

        return np.array(result)

