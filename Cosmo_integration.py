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

# ---- ANOTHER PARAMETERS

f_out = 0.1
c_b = 1.0
z_b = 0.0
sigma_b = 0.05
c_o = 1.0
z_o = 0.1
sigma_o = 0.05


class CosmoIntegration:
    def __init__(self, params):
        self.z = params['z']
        self.model = params['model']

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

    def n_t(self, z):
        z_m, z_0 = 0.9, 0.9 / np.sqrt(2)
        return ((z / z_0)**2) * np.exp(-(z / z_0)**(3 / 2))

    def p_ph(self, z_p, z):
        first = ((1 - f_out) / (np.sqrt(2 * np.pi) * sigma_b * (1+z))) * np.exp(- 0.5 * (((z - c_b*z_p - z_b) / (sigma_b * (1+z)))**2))
        second = (f_out / (np.sqrt(2 * np.pi) * sigma_o * (1+z))) * np.exp(- 0.5 * (((z - c_o*z_p - z_o) / (sigma_o * (1+z)))**2))
        return first + second

    def r(self, z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        '''
        In Mpc
        '''
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
        return np.sum(integrand) * (c / H_0) * (H_0 / c)

    def n_i_try(self, i, z):
        '''
        This function calculates the numerator for the window function for bin i.
        It uses the redshift bins defined in the original code and the n_t and p_ph functions.
        The function is normalized
        '''
        z_bins = [0.001, 0.42, 0.56, 0.68, 0.79, 0.9, 1.02, 1.15, 1.32, 1.58, 2.5]
        denominators = np.array([0.04690055617199938, 0.041209323920287824, 0.04169211292551454, 0.040191768918692396, 0.03953241118138398, 0.040135711830953276, 0.038169468867739365, 0.04019519620236196, 0.04114271877029161, 0.039251552948857606])

        def numerator_n_i(i, z):
            z_prime = np.linspace(z_bins[i], z_bins[i + 1], 50)
            delta = (z_bins[i + 1] - z_bins[i]) / len(z_prime)
            multiplication_array = self.n_t(z) * self.p_ph(z_prime, z)
            result = np.sum(multiplication_array * delta)
            return float(result)
        #def denominator(i):
            z_prime = np.linspace(z_bins[0], z_bins[-1], 30)
            delta = (z_prime[-1] - z_prime[0]) / len(z_prime)
            num = np.array([numerator_n_i(i, z_p) for z_p in z_prime])
            deno = float(np.sum(num) * delta)
            print(deno)
            return deno
        return numerator_n_i(i, z) / denominators[i]

    def Window2(self, i, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma):
        H_0 = (100 * h)
        result = []
        for z in self.z:
            z_max = 2.5
            z_prime = np.linspace(z, z_max, 30)
            delta = (z_max - z) / len(z_prime)
            r_true = self.r_w(z, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma) 
            n_array = np.array([self.n_i_try(i, zs) for zs in z_prime])
            r_array = np.array([(self.r_w(zs, Omega_m0, h, Omega_b0, Omega_DE0, w0, wa, ns, sigma8, gamma)) for zs in z_prime])
            integrand = n_array * (1 - (r_true  / r_array)) * delta
            result.append(np.sum(integrand))
        return np.array(result)
    