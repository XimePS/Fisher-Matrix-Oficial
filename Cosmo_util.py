import camb
from camb import model, initialpower
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d

# ----- PARÁMETROS FIDUCIALES -----
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

# ----- REDSHIFTS Y K -----
redshifts = np.linspace(0.001, 2.5, 250)
kmin = 5e-4
kmax = 35
npoints_k = 800


def matter_power_spectrum(z_list = redshifts, k_min = kmin, k_max = kmax, npoints_k = npoints_k, Omega_b0_fid=Omega_b0_fid, Omega_m0_fid=Omega_m0_fid, h_fid=h_fid, ns_fid=ns_fid, sigma8_fid=sigma8_fid): 
    '''
    Entrega k, P(z,k) en unidades fisicas (h/Mpc y Mpc^3 / h^3). P esta normalizado en sigma8_fid^2.
    z_list: lista de redshifts para los que se quiere calcular el espectro de potencias.
    k_min: valor mínimo de k (en unidades de 1/Mpc).
    k_max: valor máximo de k (en unidades de 1/Mpc).
    npoints_k: número de puntos en el rango de k.
    '''
    # ----- CONVERSIÓN A LOS QUE CAMB NECESITA -----
    ombh2 = Omega_b0_fid * h_fid**2
    omch2 = (Omega_m0_fid - Omega_b0_fid) * h_fid**2


    # ----- CONFIGURACIÓN INICIAL -----
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h_fid*100, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=ns_fid, As=2e-9)  # valor temporal

    # Inicial: calcular sigma8 para escalar As
    pars.set_matter_power(redshifts=[0.], kmax=kmax)
    results = camb.get_results(pars)
    sigma8_now = results.get_sigma8()
    As_scaled = 2e-9 * (sigma8_fid / sigma8_now)**2
    pars.InitPower.set_params(As=As_scaled)

    # Definir redshifts y rango de k
    pars.set_matter_power(redshifts=redshifts, kmax=kmax)

    # Ejecutar CAMB
    results = camb.get_results(pars)

    # Obtener el espectro de potencias: pk tiene shape (len(z), len(k))
    k_array_h, z_array, P_array_2d_h = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints_k) #pk esta en dim de z y k
    k_array = k_array_h # / h_fid 
    P_array_2d = P_array_2d_h # * h_fid**3


    # pk[z_index][k_index] = P(k, z)
    #print(f"k shape: {len(k_array)}, z shape: {len(z_array)}, pk shape: {P_array_2d.shape}")
    
    return k_array, z_array, P_array_2d

def der_k_matter_power_spectrum():
    k_array, z_array, P_array_2d = matter_power_spectrum()
    dP_dk = np.diff(P_array_2d, axis=1) / np.diff(k_array)
    return k_array, z_array, dP_dk

def pl_matter_power_spectrum(parametro, epsilon):
    if parametro == 'Omega_b0':
        Omega_b0 = Omega_b0_fid + epsilon
        k, z, P = matter_power_spectrum(Omega_b0_fid=Omega_b0)
    elif parametro == 'Omega_m0':
        Omega_m0 = Omega_m0_fid + epsilon
        k, z, P = matter_power_spectrum(Omega_m0_fid=Omega_m0)
    elif parametro == 'h':
        h = h_fid + epsilon
        k, z, P = matter_power_spectrum(h_fid=h)
    elif parametro == 'ns':
        ns = ns_fid + epsilon
        k, z, P = matter_power_spectrum(ns_fid=ns)
    elif parametro == 'sigma8':
        sigma8 = sigma8_fid + epsilon
        k, z, P = matter_power_spectrum(sigma8_fid=sigma8)
    else:
        raise ValueError("Parámetro no reconocido. Debe ser uno de: 'Omega_b0_fid', 'Omega_m0_fid', 'h_fid', 'ns_fid', 'sigma8_fid'.")
    
    return P

def mn_matter_power_spectrum(parametro, epsilon):
    if parametro == 'Omega_b0':
        Omega_b0 = Omega_b0_fid - epsilon
        k, z, P = matter_power_spectrum(Omega_b0_fid=Omega_b0)
    elif parametro == 'Omega_m0':
        Omega_m0 = Omega_m0_fid - epsilon
        k, z, P = matter_power_spectrum(Omega_m0_fid=Omega_m0)
    elif parametro == 'h':
        h = h_fid - epsilon
        k, z, P = matter_power_spectrum(h_fid=h)
    elif parametro == 'ns':
        ns = ns_fid - epsilon
        k, z, P = matter_power_spectrum(ns_fid=ns)
    elif parametro == 'sigma8':
        sigma8 = sigma8_fid - epsilon
        k, z, P = matter_power_spectrum(sigma8_fid=sigma8)
    else:
        raise ValueError("Parámetro no reconocido. Debe ser uno de: 'Omega_b0_fid', 'Omega_m0_fid', 'h_fid', 'ns_fid', 'sigma8_fid'.")
    
    return P

def der_matter_power_spectrum(parametro, epsilon):
    """
    Calcula la derivada del espectro de potencia de materia con respecto a un parámetro dado.
    """
    k, z, P = matter_power_spectrum()
    P_plus = np.log(pl_matter_power_spectrum(parametro, epsilon))
    P_minus = np.log(mn_matter_power_spectrum(parametro, epsilon))
    num = (P_plus - P_minus)

    if parametro == 'Omega_b0':
        den = (2 * epsilon * Omega_b0_fid)
    elif parametro == 'Omega_m0':
        den = (2 * epsilon * Omega_m0_fid)
    elif parametro == 'h':
        den = (2 * epsilon * h_fid)
    elif parametro == 'ns':
        den = (2 * epsilon * ns_fid)
    elif parametro == 'sigma8':
        den = (2 * epsilon * sigma8_fid)
    else:
        raise ValueError("Parámetro no reconocido. Debe ser uno de: 'Omega_b0_fid', 'Omega_m0_fid', 'h_fid', 'ns_fid', 'sigma8_fid'.")
    
    return P * (num / den)

## INTERPOLACIONES

def inter_matter_power_spectrum(z_list = redshifts, k_min = kmin, k_max = kmax, npoints_k = npoints_k, Omega_b0_fid=Omega_b0_fid, Omega_m0_fid=Omega_m0_fid, h_fid=h_fid, ns_fid=ns_fid, sigma8_fid=sigma8_fid):
    k_array, z_array, P_array_2d = matter_power_spectrum()
    k_array = k_array  # Convertir a logaritmo para mejor interpolación
    P_array_2d = np.log10(P_array_2d)  # Convertir a logaritmo para mejor interpolación
    interp_func = RectBivariateSpline(z_array, k_array, P_array_2d)
    print('Interpolation of matter power spectrum created.')
    return interp_func

def inter_der_matter_power_spectrum(parametro, epsilon):
    k_array, z_array, P_array_2d = matter_power_spectrum()
    der_array = der_matter_power_spectrum(parametro, epsilon)
    interp_func = RectBivariateSpline(z_array, k_array, der_array)
    print(f'Interpolation of derivative of matter power spectrum with respect to {parametro} created.')
    return interp_func

def inter_k_matter_power_spectrum():
    k_array, z_array, dP_dk = der_k_matter_power_spectrum()
    dP_dk_interp = RectBivariateSpline(z_array, k_array[0:-1], dP_dk)
    print('Interpolation of derivative of matter power spectrum with respect to k created.')
    return dP_dk_interp

