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

import Cosmo_util as cu
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
Aia = 1.72
Cia = 0.0134

class CosmicShear:
    def __init__(self, cosmic_paramss):
        self.z = cosmic_paramss['z']
        self.l = cosmic_paramss['l']
        self.universe = cosmic_paramss['type']
        self.model = cosmic_paramss['model']

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
        return np.sum(integrand) * (c / 67) 

    def r_w(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        H_0 = (100 * h)
        z_prime = np.linspace(0, z, 30)
        delta = z / len(z_prime)
        integrand = self.inverse_E2(z_prime, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) * delta
        return np.sum(integrand)* (c / 67) * (H_0 / c)

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
    
    def dD(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro):
        epsilon = 0.01
        if parametro == 'Omega_m0':
            D_pl= self.D(z, Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            D_mn= self.D(z, Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            return (D_pl - D_mn) / (2 * epsilon * Omega_m0)
        elif parametro == 'Omega_DE0':
            D_pl= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0 * (1 + epsilon), w0, wa, ns, sigma8, gamma)
            D_mn= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0 * (1 - epsilon), w0, wa, ns, sigma8, gamma)
            return (D_pl - D_mn) / (2 * epsilon * Omega_DE0)
        elif parametro == 'w0':
            D_pl= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0 * (1 + epsilon), wa, ns, sigma8, gamma)
            D_mn= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0 * (1 - epsilon), wa, ns, sigma8, gamma)
            return (D_pl - D_mn) / (2 * epsilon * w0)
        elif parametro == 'wa':
            D_pl= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa + epsilon, ns, sigma8, gamma)
            D_mn= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa - epsilon, ns, sigma8, gamma)
            return (D_pl - D_mn) / (2 * epsilon)
        elif parametro == 'gamma':
            D_pl= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma * (1 + epsilon))
            D_mn= self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma * (1 - epsilon))
            return (D_pl - D_mn) / (2 * epsilon * gamma)
        else: 
            return 1
    
    ###
    # P y sus derivadas

    interp_func = cu.inter_matter_power_spectrum()
    dP_dk_interp = cu.inter_k_matter_power_spectrum()

    der_P_inter_h = cu.inter_der_matter_power_spectrum('h', 0.01)
    der_P_inter_Omega_m0 = cu.inter_der_matter_power_spectrum('Omega_m0', 0.01) 
    der_P_inter_Omega_b0 = cu.inter_der_matter_power_spectrum('Omega_b0', 0.01) 
    der_P_inter_ns = cu.inter_der_matter_power_spectrum('ns', 0.01)
    der_P_inter_sigma8 = cu.inter_der_matter_power_spectrum('sigma8', 0.01)

    def PK(self, z, k):
        return 10**(self.interp_func(z, k, grid=False))  # tiene que ir el log10 del k original y el resultado es igual al esperado, porque la interpolacion se hace con log10(k)
    
    def PPS(self, z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
        P = float(self.PK(z, k)) # esta bien asi
        if self.universe == 'standard':
            return P
        else:
            D_0 = self.D(0, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            D_array = self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
            return P*((D_array/D_0)**2)
    
    def der_PPS_parametro(self, z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro):
        epsilon = 0.01
        H_0 = 100 * h
        def der_PPS_k(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
            k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
            P = self.dP_dk_interp(z, k)
            if self.universe == 'standard':
                return float(P)
            else:
                D_0 = self.D(0, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                D_array = self.D(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)
                return float(P * ((D_array/D_0)**2))
            
        k = ((10**l + 0.5) / (self.r(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)))
        third = der_PPS_k(z, l, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)

        if parametro == 'h':
            first = self.der_P_inter_h(z, k, grid=False)
            k_pl = (10**l + 0.5)/(self.r(z, Omega_m0, h * (1 + epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            k_mn = (10**l + 0.5)/(self.r(z, Omega_m0, h * (1 - epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            second = (k_pl - k_mn) / (2 * epsilon * h)
            return first + (second * third)
        elif parametro == 'ns':
            first = self.der_P_inter_ns(z, k, grid=False)
            return first
        elif parametro == 'Omega_b0':
            first = self.der_P_inter_Omega_b0(z, k, grid=False)
            k_pl = (10**l + 0.5)/(self.r(z, Omega_m0, h, Omega_b0 * (1 + epsilon), Omega_DE0, w0, wa, ns, sigma8, gamma))
            k_mn = (10**l + 0.5)/(self.r(z, Omega_m0, h, Omega_b0 * (1 - epsilon), Omega_DE0, w0, wa, ns, sigma8, gamma))
            second = (k_pl - k_mn) / (2 * epsilon * Omega_b0)
            return first + (second * third)
        elif parametro == 'Omega_m0':
            first = self.der_P_inter_Omega_m0(z, k, grid=False)
            k_pl = (10**l + 0.5)/(self.r(z, Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            k_mn = (10**l + 0.5)/(self.r(z, Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma))
            second = (k_pl - k_mn) / (2 * epsilon * Omega_m0)
            return first  + (second * third)
        elif parametro == 'sigma8':
            first = self.der_P_inter_sigma8(z, k, grid=False)
            return first
        else:
            print('We do not have the derivative of the power spectrum with respect to this parameter')
    ###
    def K(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia):

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

    
    def Cosmic_Shear(self, i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia):
        z_max, z_min, z0 = 2.5, 0.001, 0.62
        z_prime, delta = self.z, (z_max - z_min) / len(self.z)
        SNs = self.SN(i, j)
        n = -0.41
        
        result = []

        D_array = np.array([self.D(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])
        K_gg, K_Ig, K_II = self.K(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)

        for i, l in enumerate(self.l):
            if l == self.l[-1]: 
                ls = l
            else:
                ls = np.log10((10**self.l[i+1] + 10**self.l[i])/2) 

            P_gg = np.array([self.PPS(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for z_primes in z_prime])
            P_Ig = (-(Aia * Cia * Omega_m0) / D_array) * P_gg
            P_II = (((-(Aia * Cia * Omega_m0) / D_array)) ** 2) * P_gg

            integrand = ((K_gg * P_gg) + (K_Ig * P_Ig) + (K_II * P_II)) * float(delta)
            integral =  np.sum(integrand)
            integral_final = integral + SNs

            result.append(integral_final)

        return np.array(result)
    
    def Der_C_parametro(self, i ,j, epsilon, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia, parametro):
        z_max, z_min, z0 = 2.5, 0.001, 0.62
        z_prime, delta = self.z, (z_max - z_min) / len(self.z)
        result = []
        n = -0.41
        epsilon = 0.01

        D_array = np.array([self.D(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])
        K_gg, K_Ig, K_II = self.K(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)
        C = self.Cosmic_Shear(i ,j, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)

        if parametro == 'h':
            K_gg_pl, K_Ig_pl, K_II_pl = self.K(i ,j, Omega_m0, h * (1 + epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)
            K_gg_mn, K_Ig_mn, K_II_mn = self.K(i ,j, Omega_m0, h * (1 - epsilon), Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)

            der_K_gg = (K_gg_pl - K_gg_mn) / (2*epsilon*h)
            der_K_Ig = (K_Ig_pl - K_Ig_mn) / (2*epsilon*h)
            der_K_II = (K_II_pl - K_II_mn) / (2*epsilon*h)

            for i, l in enumerate(self.l):
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[i+1] + 10**self.l[i])/2) 

                op1 = (-Aia * Cia * Omega_m0 / D_array)
                P_gg = np.array([self.PPS(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for z_primes in z_prime])
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])
                P_Ig = op1 * P_gg
                der_P_Ig = (op1 * der_P_gg)
                P_II = (op1 ** 2) * P_gg
                der_P_II = ((op1 ** 2) * der_P_gg)

                integrand = ((der_K_gg * P_gg) + (K_gg * der_P_gg) + (der_K_Ig * P_Ig)  + (K_Ig * der_P_Ig) + (der_K_II * P_II) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'Omega_m0':
            D_array_pl = np.array([self.D(zs, Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])
            D_array_mn = np.array([self.D(zs, Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for zs in z_prime])
            der_D = (D_array_pl - D_array_mn) / (2 * epsilon * Omega_m0)
            K_gg_pl, K_Ig_pl, K_II_pl = self.K(i ,j, Omega_m0 * (1 + epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)
            K_gg_mn, K_Ig_mn, K_II_mn = self.K(i ,j, Omega_m0 * (1 - epsilon), h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, Aia, Cia)

            der_K_gg = (K_gg_pl - K_gg_mn) /(2*epsilon*Omega_m0)
            der_K_Ig = (K_Ig_pl - K_Ig_mn) /(2*epsilon*Omega_m0)
            der_K_II = (K_II_pl - K_II_mn) /(2*epsilon*Omega_m0)

            for i, l in enumerate(self.l):
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[i+1] + 10**self.l[i])/2) 
                op1 = (-Aia * Cia * Omega_m0 / D_array)
                der_op_1 = -Aia * Cia * ((1 / D_array) - (Omega_m0*(der_D / (D_array**2))))
                P_gg = np.array([self.PPS(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) for z_primes in z_prime])
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])
                P_Ig = op1 * P_gg
                der_P_Ig = (op1 * der_P_gg) + (der_op_1 * P_gg)
                P_II = (op1 ** 2) * P_gg
                der_P_II = ((op1 ** 2) * der_P_gg) + (2 * op1 * der_op_1 * der_P_gg)

                integrand = ((der_K_gg * P_gg) + (K_gg * der_P_gg) + (der_K_Ig * P_Ig)  + (K_Ig * der_P_Ig) + (der_K_II * P_II) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'Aia':
            for i, l in enumerate(self.l):
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[i+1] + 10**self.l[i])/2) 

                P_gg = np.array([self.PPS(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)  for z_primes in z_prime])
                P_Ig_pl = (-Aia * (1 + epsilon) * Cia * Omega_m0 / D_array) * P_gg
                P_Ig_mn = (-Aia * (1 - epsilon) * Cia * Omega_m0 / D_array) * P_gg
                P_II_pl = ((-Aia * (1 + epsilon) * Cia * Omega_m0 / D_array) ** 2) * P_gg
                P_II_mn = ((-Aia * (1 - epsilon) * Cia * Omega_m0 / D_array) ** 2) * P_gg
                der_P_Ig = (P_Ig_pl - P_Ig_mn) / (2 * epsilon * Aia)
                der_P_II = (P_II_pl - P_II_mn) / (2 * epsilon * Aia)

                integrand = ((K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)

        elif parametro == 'Cia':
            for i, l in enumerate(self.l):
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[i+1] + 10**self.l[i])/2) 

                P_gg = np.array([self.PPS(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)  for z_primes in z_prime])
                P_Ig_pl = (-Aia  * Cia * (1 + epsilon) * Omega_m0 / D_array) * P_gg
                P_Ig_mn = (-Aia * Cia * (1 - epsilon) * Omega_m0 / D_array) * P_gg
                P_II_pl = ((-Aia  * Cia * (1 + epsilon) * Omega_m0 / D_array) ** 2) * P_gg
                P_II_mn = ((-Aia * Cia * (1 - epsilon) * Omega_m0 / D_array) ** 2) * P_gg
                der_P_Ig = (P_Ig_pl - P_Ig_mn) / (2 * epsilon * Cia)
                der_P_II = (P_II_pl - P_II_mn) / (2 * epsilon * Cia)

                integrand = ((K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral

                result.append(integral_final)

            return np.array(result)
        
        elif parametro == 'Omega_DE0':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0 * (1 + epsilon), w0, wa, ns, sigma8, gamma, Aia, Cia))
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0 * (1 - epsilon), w0, wa, ns, sigma8, gamma, Aia, Cia))
            der = (C_pl - C_mn) / (2*epsilon*Omega_DE0)
            return der * C

        elif parametro == 'w0':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0 * (1 + epsilon), wa, ns, sigma8, gamma, Aia, Cia))
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0 * (1 - epsilon), wa, ns, sigma8, gamma, Aia, Cia))
            der = (C_pl - C_mn) / (2*epsilon*w0)
            return der * C
        
        elif parametro == 'wa':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa + epsilon, ns, sigma8, gamma, Aia, Cia))       
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa - epsilon, ns, sigma8, gamma, Aia, Cia))       
            der = (C_pl - C_mn) / (2*epsilon)
            return der * C
        
        elif parametro == 'gamma':
            C_pl = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma * (1 + epsilon), Aia, Cia))
            C_mn = np.log(self.Cosmic_Shear(i ,j, Omega_m0, h , Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma * (1 - epsilon), Aia, Cia))
            der = (C_pl - C_mn) / (2*epsilon*gamma)
            return der * C
        
        else:
            # ns, sigma8, Omega_b0
            for i, l in enumerate(self.l):
                if l == self.l[-1]: 
                    ls = l
                else:
                    ls = np.log10((10**self.l[i+1] + 10**self.l[i])/2) 

                op1 = (-Aia * Cia * Omega_m0 / D_array)
                der_P_gg = np.array([self.der_PPS_parametro(z_primes, ls, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma, parametro) for z_primes in z_prime])
                der_P_Ig = op1 * der_P_gg
                der_P_II = (op1 ** 2) * der_P_gg

                integrand = ((K_gg * der_P_gg) + (K_Ig * der_P_Ig) + (K_II * der_P_II)) * float(delta)
                integral =  np.sum(integrand)
                integral_final = integral 

                result.append(integral_final)

            return np.array(result)