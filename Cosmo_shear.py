import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import pandas as pd

# For interpolation
from scipy.interpolate import RectBivariateSpline, interp2d

import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divmax.*")

import logging
# Basic registry settings
logging.basicConfig(level=logging.INFO)

import Cosmo_util_data as cu
import Cosmo_integration as ci

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

c = 300000
Aia = 1
Cia = 0.0134
nia = -1
bia = 1.13


class CosmicShear:
    def __init__(self, cosmic_paramss):
        self.z = cosmic_paramss['z']
        self.l = cosmic_paramss['l']
        self.universe = cosmic_paramss['type']
        self.model = cosmic_paramss['model']

    def generate_interpolator(self, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8):
        params = {
            'Omega_m': Omega_m0,
            'Omega_b': Omega_b0,
            'h': h,
            'ns': ns,
            'sigma8': sigma8,
            'w0': w0,
            'wa': wa,
            'Omega_Lambda': Omega_DE0,  # o lo que use tu wrapper
            'model': self.model
        }
        return cu.inter_matter_power_spectrum(params)

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
        radicando = Omega_m0 * (1 + z)**3 + (Omega_DE0 * (1 + z)**(3 * (1 + wa + w0)) * np.exp(-3 * wa * (z / (1 + z)))) + (Omega_k0) * (1 + z)**2
        return np.sqrt(radicando)

    def inverse_E2(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        return 1 / self.E2(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)

    def r(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        H_0 = (100 * h)
        z_prime = np.linspace(0, z, 30)
        delta = z / len(z_prime)
        integrand = self.inverse_E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) * delta
        return np.sum(integrand) * (c / H_0)

    def r_w(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        H_0 = (100 * h)
        z_prime = np.linspace(0, z, 30)
        delta = z / len(z_prime)
        integrand = self.inverse_E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) * delta
        return np.sum(integrand)* (c / H_0) * (H_0 / c)

    def SN(self, i, j):
        if i == j:
            return (0.3**2) / 35454308.58
        else: 
            return 0
        #return (0.3**2) / 35454308.58 if i == j else 0

    def D(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        z_prime = np.linspace(0, z, 30)
        E_array = self.E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        Omega_m = (Omega_m0  * (1 + z_prime)**3) / (E_array**2)
        delta = z / len(z_prime)
        integral = np.sum((Omega_m**gamma / (1 + z_prime)) * delta)
        return np.exp(-integral)
    
    ###
    # P y sus derivadas

    interp_func = cu.inter_matter_power_spectrum()
    dP_dk_interp = cu.inter_k_matter_power_spectrum()

    der_P_inter_h = cu.inter_der_matter_power_spectrum('h', 0.01)
    der_P_inter_Omega_m0 = cu.inter_der_matter_power_spectrum('Omega_m0', 0.01) 
    der_P_inter_Omega_b0 = cu.inter_der_matter_power_spectrum('Omega_b0', 0.01) 
    der_P_inter_ns = cu.inter_der_matter_power_spectrum('ns', 0.01)
    der_P_inter_sigma8 = cu.inter_der_matter_power_spectrum('sigma8', 0.01)

    # Luminosity
    Lumo = cu.Lumo()

    def PK(self, z, k):
        return 10**(self.interp_func(z, np.log10(k), grid=False))  # tiene que ir el log10 del k original y el resultado es igual al esperado, porque la interpolacion se hace con log10(k)
    
    def PPS(self, z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
        P = self.PK(z, k)
        if self.universe == 'standard':
            return P
        else:
            D_0 = self.D(0, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            D_array = self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            return P*((D_array/D_0)**2)
    
    def der_PPS_parametro(self, z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro):
        epsilon = 0.01
        H_0 = 100 * h
        k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
        P = self.PK(z, k)
        def der_PPS_k(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
            k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
            P = self.dP_dk_interp(z, np.log10(k))
            if self.universe == 'standard':
                return float(P)
            else:
                D_0 = self.D(0, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                D_array = self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                return float(P * ((D_array/D_0)**2))
            
        k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
        third = der_PPS_k(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)

        if parametro == 'h':
            first = P * self.der_P_inter_h(z, np.log10(k), grid=False)
            k_pl = (10**l + 0.5)/(self.r(z, Omega_m0, h * (1 + epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            k_mn = (10**l + 0.5)/(self.r(z, Omega_m0, h * (1 - epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            second = (k_pl - k_mn) / (2 * epsilon * h)
            return first #+ (second * third)
        elif parametro == 'ns':
            first = P * self.der_P_inter_ns(z, np.log10(k), grid=False)
            return first
        elif parametro == 'Omega_b0':
            first = P * self.der_P_inter_Omega_b0(z, np.log10(k), grid=False)
            k_pl = (10**l + 0.5)/(self.r(z, Omega_m0, h, Omega_b0 * (1 + epsilon), Omega_DE0, w0, wa, ns, sigma8, gamma))
            k_mn = (10**l + 0.5)/(self.r(z, Omega_m0, h, Omega_b0 * (1 - epsilon), Omega_DE0, w0, wa, ns, sigma8, gamma))
            second = (k_pl - k_mn) / (2 * epsilon * Omega_b0)
            return first + (second * third)
        elif parametro == 'Omega_m0':
            first = P * self.der_P_inter_Omega_m0(z, np.log10(k), grid=False)
            k_pl = (10**l + 0.5)/(self.r(z, Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            k_mn = (10**l + 0.5)/(self.r(z, Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            second = (k_pl - k_mn) / (2 * epsilon * Omega_m0)
            return first  + (second * third)
        elif parametro == 'sigma8':
            first = self.der_P_inter_sigma8(z, np.log10(k), grid=False)
            return first  #/ (sigma8 ** 2) #- 2 * P / (sigma8 ** 3) * (sigma8 ** 2)
        else:
            print('We do not have the derivative of the power spectrum with respect to this parameter')
    ###
    def K(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):

        H_0 = (100 * h)
        z_prime= self.z

        params = {'z': z_prime, 'model': self.model}
    
        A = ci.CosmoIntegration(params)

        E_array = self.E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
        Wi = np.array(A.Window2(i, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
        n_i_array = np.array([A.n_i_try(i, zs) for zs in z_prime])
        Wj = np.array(A.Window2(j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
        n_j_array = np.array([A.n_i_try(j, zs) for zs in z_prime])
        r_array = np.array([A.r_w(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])

        operador1 = ((1.5 * Omega_m0 * (1+z_prime)) **2) * ((H_0 / c) ** 3)
        operador2 = 1.5 * Omega_m0 * (1+z_prime) * ( (H_0 / c) ** 3)
        operador3 =  (H_0 / c) ** 3

        K_gg = operador1 * (Wi * Wj / (E_array))
        K_Ig = operador2 * ((n_i_array * Wj) + (n_j_array * Wi)) / (r_array) 
        K_II = operador3 * (n_i_array * n_j_array * E_array) / ((r_array) ** 2)

        return K_gg, K_Ig, K_II

    ###
    def operando(self, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
        z_max, z_min, z0 = 2.5, 0.001, 0.62
        z_prime, delta = self.z, (z_max - z_min) / len(self.z)
        D_array = np.array([self.D(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])
        F = ((1 + z_prime) ** nia) * (self.Lumo(z_prime) ** bia)
        op = (-(Aia * Cia * Omega_m0 * F) / D_array) 
        return op

    def Ps(self, l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k):
        z_max, z_min, z0 = 2.5, 0.001, 0.62
        z_prime, delta = self.z, (z_max - z_min) / len(self.z)

        operando = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)

        def lambda_k(i): 
            L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
            lambda_min = np.log10(10**L_array[0])
            lambda_max = np.log10(10**L_array[-1])
            delta_lambda = (lambda_max - lambda_min) / len(L_array)
            lambda_k = lambda_min + (k - 1)*delta_lambda
            return lambda_k
        
        ls = lambda_k(k)

        P_gg = np.array([self.PPS(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for z_primes in z_prime])
        P_Ig = operando * P_gg
        P_II = (operando ** 2) * P_gg

        return P_gg, P_Ig, P_II


    
    def Cosmic_Shear(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia):
        z_max, z_min, z0 = 2.5, 0.001, 0.62
        z_prime, delta = self.z, (z_max - z_min) / len(self.z)
        SNs = self.SN(i, j)

        K_gg, K_Ig, K_II = self.K(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        
        result = []

        for k, l in enumerate(self.l):
            P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)

            integrand = ((K_gg * P_gg) + (K_Ig * P_Ig) + (K_II * P_II)) * float(delta)
            integral =  np.sum(integrand)
            integral_final = integral + SNs

            result.append(integral_final)

        return np.array(result)
    
    def Der_C_parametro(self, i ,j, epsilon, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, parametro):
        z_max, z_min, z0 = 2.5, 0.001, 0.62
        z_prime, delta = self.z, (z_max - z_min) / len(self.z)
        epsilon = 0.01
        SNs = self.SN(i, j)

        C = self.Cosmic_Shear(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        op1 = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        K_gg, K_Ig, K_II = self.K(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
        result = []

        if parametro == 'h':
            K_gg_pl, K_Ig_pl, K_II_pl = self.K(i ,j, Omega_m0, h * (1 + epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            K_gg_mn, K_Ig_mn, K_II_mn = self.K(i ,j, Omega_m0, h * (1 - epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)

            der_K_gg = (K_gg_pl - K_gg_mn) / (2*epsilon*h)
            der_K_Ig = (K_Ig_pl - K_Ig_mn) / (2*epsilon*h)
            der_K_II = (K_II_pl - K_II_mn) / (2*epsilon*h)

            for k, l in enumerate(self.l):
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)

                P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])
                der_P_Ig = (op1 * der_P_gg)
                der_P_II = ((op1 ** 2) * der_P_gg)

                integrand = ((der_K_gg * P_gg) + (K_gg * der_P_gg) + (der_K_Ig * P_Ig)  + (K_Ig * der_P_Ig) + (der_K_II * P_II) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        elif parametro == 'Omega_m0':
            K_gg_pl, K_Ig_pl, K_II_pl = self.K(i ,j, Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            K_gg_mn, K_Ig_mn, K_II_mn = self.K(i ,j, Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            der_K_gg = (K_gg_pl - K_gg_mn) / (2*epsilon*Omega_m0)
            der_K_Ig = (K_Ig_pl - K_Ig_mn) / (2*epsilon*Omega_m0)
            der_K_II = (K_II_pl - K_II_mn) / (2*epsilon*Omega_m0)
            op1_pl = self.operando(Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            op1_mn = self.operando(Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            op1_der = (op1_pl - op1_mn) / (2 * epsilon * Omega_m0)

            for k, l in enumerate(self.l):  
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)
                P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])

                der_P_Ig = (op1_der * P_gg) + (op1 * der_P_gg)
                der_P_II = ((op1 * 2 * op1_der) * P_gg) + ((op1 ** 2) * der_P_gg)

                integrand = ((der_K_gg * P_gg) + (K_gg * der_P_gg) + (der_K_Ig * P_Ig)  + (K_Ig * der_P_Ig) + (der_K_II * P_II) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'Aia':
            op1_pl = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia * (1 + epsilon), nia, bia)
            op1_mn = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia * (1 - epsilon), nia, bia)
            op1_der = (op1_pl - op1_mn) / (2 * epsilon * Aia)
            for k, l in enumerate(self.l):  
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)
                P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)

                der_P_Ig = (op1_der * P_gg) 
                der_P_II = ((op1 * 2 * op1_der) * P_gg) 

                integrand = ((K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)

        elif parametro == 'nia':
            op1_pl = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia * (1 + epsilon), bia)
            op1_mn = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia * (1 - epsilon), bia)
            op1_der = (op1_pl - op1_mn) / (2 * epsilon * nia)
            for k, l in enumerate(self.l):  
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)
                P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)

                der_P_Ig = (op1_der * P_gg) 
                der_P_II = ((op1 * 2 * op1_der) * P_gg) 

                integrand = ((K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        elif parametro == 'bia':
            op1_pl = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia * (1 + epsilon))
            op1_mn = self.operando(Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia * (1 - epsilon))
            op1_der = (op1_pl - op1_mn) / (2 * epsilon * bia)
            for k, l in enumerate(self.l):  
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)
                P_gg, P_Ig, P_II = self.Ps(l, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia, k)

                der_P_Ig = (op1_der * P_gg) 
                der_P_II = ((op1 * 2 * op1_der) * P_gg) 

                integrand = ((K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'sigma8_probe':
            '''
            for k, l in enumerate(self.l):  
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[k+1] + 10**self.l[k])/2) 
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])

                der_P_Ig = (op1 * der_P_gg)
                der_P_II = ((op1 ** 2) * der_P_gg)

                integrand = ((K_gg * der_P_gg)  + (K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)'''
            C = self.Cosmic_Shear(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, nia, bia)
            return 2 * (C - SNs) / sigma8
        elif parametro == 'sigma8':
            for k, l in enumerate(self.l):  
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[k+1] + 10**self.l[k])/2) 
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])

                der_P_Ig = (op1 * der_P_gg)
                der_P_II = ((op1 ** 2) * der_P_gg)

                integrand = ((K_gg * der_P_gg)  + (K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'ns':
            for k, l in enumerate(self.l):  
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])

                der_P_Ig = (op1 * der_P_gg)
                der_P_II = ((op1 ** 2) * der_P_gg)

                integrand = ((K_gg * der_P_gg)  + (K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        elif parametro == 'Omega_b0':
            for k, l in enumerate(self.l):  
                def lambda_k(i): 
                    L_array = np.log10(np.logspace(np.log10(10), np.log10(1500), 100))
                    lambda_min = np.log10(10**L_array[0])
                    lambda_max = np.log10(10**L_array[-1])
                    delta_lambda = (lambda_max - lambda_min) / len(L_array)
                    lambda_k = lambda_min + (k - 1)*delta_lambda
                    return lambda_k
                
                l = lambda_k(k)
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])

                der_P_Ig = (op1 * der_P_gg)
                der_P_II = ((op1 ** 2) * der_P_gg)

                integrand = ((K_gg * der_P_gg)  + (K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'Omega_DE0':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0 * (1 + epsilon), w0, wa, ns, sigma8, gamma, Aia, nia, bia))
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0 * (1 - epsilon), w0, wa, ns, sigma8, gamma, Aia, nia, bia))
            der = (C_pl - C_mn) / (2*epsilon*Omega_DE0)
            return der * C

        elif parametro == 'w0':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0 * (1 + epsilon), wa, ns, sigma8, gamma, Aia, nia, bia))
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0 * (1 - epsilon), wa, ns, sigma8, gamma, Aia, nia, bia))
            der = (C_pl - C_mn) / (2*epsilon*w0)
            return der * C
        
        elif parametro == 'wa':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa + epsilon, ns, sigma8, gamma, Aia, nia, bia))       
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa - epsilon, ns, sigma8, gamma, Aia, nia, bia))       
            der = (C_pl - C_mn) / (2*epsilon)
            return der * C
        
        elif parametro == 'gamma':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma * (1 + epsilon), Aia, nia, bia))
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma * (1 - epsilon), Aia, nia, bia))
            der = (C_pl - C_mn) / (2*epsilon*gamma)
            return der * C
    
        else:
            print('We do not have the derivative of the cosmic shear with respect to this parameter')