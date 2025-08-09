import camb
from camb import model, initialpower
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.interpolate import griddata

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
Aia = 1
Cia = 0.0134
nia = -1
bia = 1.13


# ----- REDSHIFTS Y K -----
redshifts = np.linspace(0.001, 2.5, 250)
kmin = 5e-4
kmax = 35
npoints_k = 800
k_array_21 = np.logspace(5e-4, 35, 800)


def matter_power_spectrum(): 
    data = np.loadtxt("pkz-Fiducial.txt")  # ajusta el nombre

    z_all = data[:, 0]
    k_all = data[:, 1]
    P_all = data[:, 2]  # o la columna que te interese

    # Obtener arrays únicos de z y k
    z_unique = np.unique(z_all)
    k_unique = np.unique(k_all)

    # Crear una malla 2D de P(z, k)
    # Asumimos que por cada z hay el mismo número de k's (o viceversa)
    P_matrix = np.empty((len(z_unique), len(k_unique)))

    for i, z_val in enumerate(z_unique):
        # Filtrar las filas correspondientes a este z
        mask = z_all == z_val
        k_vals_z = k_all[mask]
        P_vals_z = P_all[mask]

        # Ordenar por k, por si acaso
        sorted_indices = np.argsort(k_vals_z)
        k_vals_z = k_vals_z[sorted_indices]
        P_vals_z = P_vals_z[sorted_indices]

        # Insertar en la matriz
        P_matrix[i, :] = P_vals_z
    return z_unique, k_vals_z , P_matrix #/ (h_fid**3)) * (sigma8_fid ** 2)

def inter_matter_power_spectrum():
    z_unique, k_vals_z, P_array_2d = matter_power_spectrum()
    k_array = np.log10(k_vals_z)  # Convertir a logaritmo para mejor interpolación
    P_array_2d = np.log10(P_array_2d)  # Convertir a logaritmo para mejor interpolación
    interp_func = RectBivariateSpline(z_unique, k_array, P_array_2d)
    print('Interpolation of matter power spectrum created.')
    return interp_func

def der_k_matter_power_spectrum():
    z_array, k_array, P_array_2d = matter_power_spectrum()

    # Usamos gradient en el eje de k (axis=1)
    # Es importante pasar el espaciado k_array explícitamente
    dP_dk = np.gradient(P_array_2d, k_array, axis=1)

    return k_array, z_array, dP_dk

def pl_matter_power_spectrum(parametro, epsilon):
    
    if parametro == 'h':
        data = np.loadtxt("pkz-h_pl_eps_1p3E-2.txt")  # o "\t" para tabulaciones
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z

    elif parametro == 'Omega_b0':
        data = np.loadtxt("pkz-Ob_pl_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    elif parametro == 'Omega_m0':
        data = np.loadtxt("pkz-Om_pl_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    elif parametro == 'ns':
        data = np.loadtxt("pkz-ns_pl_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    elif parametro == 'sigma8':     
        data = np.loadtxt("pkz-s8_pl_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    else:
        raise ValueError("Parámetro no reconocido. Debe ser uno de: 'Omega_b0_fid', 'Omega_m0_fid', 'h_fid', 'ns_fid', 'sigma8_fid'.")  
    
    return z_unique, k_vals_z , P_matrix 

def mn_matter_power_spectrum(parametro, epsilon):
    if parametro == 'h':
        data = np.loadtxt("pkz-h_mn_eps_1p3E-2.txt")  # o "\t" para tabulaciones
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z

    elif parametro == 'Omega_b0':
        data = np.loadtxt("pkz-Ob_mn_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    elif parametro == 'Omega_m0':
        data = np.loadtxt("pkz-Om_mn_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    elif parametro == 'ns':
        data = np.loadtxt("pkz-ns_mn_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    elif parametro == 'sigma8':     
        data = np.loadtxt("pkz-s8_mn_eps_1p3E-2.txt")
        z_all = data[:, 0]
        k_all = data[:, 1]
        P_all = data[:, 2]  # o la columna que te interese

        # Obtener arrays únicos de z y k
        z_unique = np.unique(z_all)
        k_unique = np.unique(k_all)

        # Crear una malla 2D de P(z, k)
        # Asumimos que por cada z hay el mismo número de k's (o viceversa)
        P_matrix = np.empty((len(z_unique), len(k_unique)))

        for i, z_val in enumerate(z_unique):
            # Filtrar las filas correspondientes a este z
            mask = z_all == z_val
            k_vals_z = k_all[mask]
            P_vals_z = P_all[mask]

            # Ordenar por k, por si acaso
            sorted_indices = np.argsort(k_vals_z)
            k_vals_z = k_vals_z[sorted_indices]
            P_vals_z = P_vals_z[sorted_indices]

            # Insertar en la matriz
            P_matrix[i, :] = P_vals_z
            
    else:
        raise ValueError("Parámetro no reconocido. Debe ser uno de: 'Omega_b0_fid', 'Omega_m0_fid', 'h_fid', 'ns_fid', 'sigma8_fid'. 'o'")  
    
    return z_unique, k_vals_z , P_matrix 

def der_matter_power_spectrum(parametro, epsilon):
    """
    Calcula la derivada del espectro de potencia de materia con respecto a un parámetro dado.
    """
    z_list, k_list, P_plus = pl_matter_power_spectrum(parametro, epsilon) 
    z_list, k_list, P_minus = mn_matter_power_spectrum(parametro, epsilon)
    P_plus = np.log(P_plus)
    P_minus = np.log(P_minus)

    if parametro == 'Omega_b0':
        num = (P_plus - P_minus)
        den = (2 * epsilon * Omega_b0_fid)
    elif parametro == 'Omega_m0':
        P_minus = P_minus[:, 1:]
        num = (P_plus - P_minus)
        den = (2 * epsilon * Omega_m0_fid)
    elif parametro == 'h':
        num = (P_plus - P_minus)
        den = (2 * epsilon * h_fid)
    elif parametro == 'ns':
        num = (P_plus - P_minus)
        den = (2 * epsilon * ns_fid)
    elif parametro == 'sigma8':
        num = (P_plus - P_minus)
        den = (2 * epsilon * sigma8_fid)
    else:
        raise ValueError("Parámetro no reconocido. Debe ser uno de: 'Omega_b0_fid', 'Omega_m0_fid', 'h_fid', 'ns_fid', 'sigma8_fid'.")
    
    return z_list, k_list * h_fid, (num / den) * (h_fid ** 3)


def luminosity():
    data = np.loadtxt("scaledmeanlum-E2Sa.dat")  # o "\t" para tabulaciones

    # Filtrar por un redshift específico, por ejemplo z = 2.5

    # Extraer columnas
    z_list = data[:, 0]       # columna de k
    Lum = data[:, 1]   # columna de derivadas con respecto a h

    return z_list, Lum

## INTERPOLACIONES

def inter_der_matter_power_spectrum(parametro, epsilon):    
    z_list, k_list, der = der_matter_power_spectrum(parametro, epsilon)
    if parametro == 'Omega_m0':
        interp_func = RectBivariateSpline(z_list, np.log10(k_list[1:]), der)
    else:
        interp_func = RectBivariateSpline(z_list, np.log10(k_list), der)
    return interp_func

def inter_k_matter_power_spectrum():
    k_array, z_array, dP_dk = der_k_matter_power_spectrum()
    k_array = np.log10(k_array)  # Convertir a logaritmo para mejor interpolación
    dP_dk = dP_dk  # Convertir a
    dP_dk_interp = RectBivariateSpline(z_array, k_array, dP_dk)
    print('Interpolation of derivative of matter power spectrum with respect to k created.')
    return dP_dk_interp

def Lumo():
    z, L = luminosity()
    Lumo = interp1d(z, L, fill_value='extrapolate')
    print('Interpolation of luminosity function created.')
    return Lumo