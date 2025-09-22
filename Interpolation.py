import camb
from camb import model, initialpower
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
import matplotlib.pyplot as plt
import random

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

#c = 9.72 * 10 ** (-15) # en Mpc # 300000 en km/s
c = 300000 #en km/s
Aia = 1.72
Cia = 0.0134
nia = -0.41
bia = 2.17

# ----- INTERPOLATION -----

def interpolation(path, lineal = False):
    if lineal:
        idx_p = 2
    else:
        idx_p = 3
    data = np.loadtxt(path)

    z_vals = np.unique(data[:,0]) # devuelve z ordenados de abajo hacia arriba
    k_vals = np.unique(data[:,1]) # devuelve k ordenador de abajo hacia arriba
    P_vals = np.flip(data[:,idx_p]) # devuelve P ordenado de abajo hacia arriba -> esto implica que para cada z, P esta ordenado de mayor a menor k

    # Se revisa que los k sean iguales para todos los z
    k_all = data[:,1]
    l = len(k_vals)

    #print(len(z_vals)-1)

    for i in range(0, len(z_vals)-1): # va de 0 a len(z_vals)-2
        l = len(k_vals)
        A = k_all[i*l : l*(i+1)]
        B = k_all[(i+1)*l : l*(i+2)]
        if not np.array_equal(A, B):
            print("Los arrays no son iguales, se detiene el for")
            break

    P_new = []

    for i, z in enumerate(z_vals):
        A = P_vals[i*l : l*(i+1)]
        #print(np.flip(A))
        #print(k_vals)
        P_new.append(list(np.flip(A))) # tengo que voltear P para alinearlo luego con el orden de k_vals (de menor a mayor)
    
    P_inter = RectBivariateSpline(z_vals, np.log(k_vals), np.log(P_new))
    print('Interpolation of ' + path + ' done')
    return P_inter

# ----- CHECK INTERPOLATION -----
def check_interpolation(path):
    data = np.loadtxt(path)
    
    z_vals = np.unique(data[:,0]) # devuelve z ordenados de abajo hacia arriba
    k_vals = np.unique(data[:,1]) # devuelve k ordenador de abajo hacia arriba
    P_vals = np.flip(data[:,3]) # devuelve P ordenado de abajo hacia arriba -> esto implica que para cada z, P esta ordenado de mayor a menor k

    k_list = np.logspace(np.log10(0.00001), np.log10(30), 700) # va de 0.00001 a 30 en espacios logaritmicos
    i = random.randint(0, 303) 
    z = z_vals[i]
    l = len(k_vals)


    P_kk = interpolation(path)
    P_kk_evaluated = np.exp(P_kk(z, np.log(k_list)))

    plt.plot(k_vals, np.flip(P_vals[l*(i): l*(i+1)]), '-', color = 'orange', label = 'Data')
    plt.plot(k_list, P_kk_evaluated[0,:], '--', color = 'green', label = 'Our interp')
    plt.xscale('log')
    plt.yscale('log')   
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)^3]')#
    plt.title(path + ' at z=' + str(z))
    plt.grid()
    plt.legend()
    plt.show()

# ----- DERIVATIVE OF P(k) IN K -----
def dPdk_interpolation(path):

    data = np.loadtxt(path)

    z_vals = np.unique(data[:,0]) # devuelve z ordenados de abajo hacia arriba
    k_vals = np.unique(data[:,1]) # devuelve k ordenador de abajo hacia arriba
    P_vals = np.flip(data[:,3]) # devuelve P ordenado de abajo hacia arriba -> esto implica que para cada z, P esta ordenado de mayor a menor k

    # Se revisa que los k sean iguales para todos los z
    k_all = data[:,1]
    l = len(k_vals)

    #print(len(z_vals)-1)

    for i in range(0, len(z_vals)-1): # va de 0 a len(z_vals)-2
        l = len(k_vals)
        A = k_all[i*l : l*(i+1)]
        B = k_all[(i+1)*l : l*(i+2)]
        if not np.array_equal(A, B):
            print("Los arrays no son iguales, se detiene el for")
            break

    P_new = []

    for i, z in enumerate(z_vals):
        A = P_vals[i*l : l*(i+1)]
        P_new.append(list(np.flip(A))) # tengo que voltear P para alinearlo luego con el orden de k_vals (de menor a mayor)  

    dP_dk = []

    for i, z in enumerate(z_vals):
        Pz = np.array(P_new[i])        # P(k) para un z fijo
        dP = np.gradient(Pz, k_vals)   # derivada respecto a k
        dP_dk.append(dP)

    dP_dk = np.array(dP_dk)  # shape: (len(z_vals), len(k_vals))

    dP_dk_inter = RectBivariateSpline(z_vals, np.log(k_vals), dP_dk)
    return dP_dk_inter

# ----- CHECK DERIVATIVE OF P(k) IN K -----
def check_dPdk_interpolation(path):
    data = np.loadtxt(path)

    z_vals = np.unique(data[:,0]) # devuelve z ordenados de abajo hacia arriba
    k_vals = np.unique(data[:,1]) # devuelve k ordenador de abajo hacia arriba
    P_vals = np.flip(data[:,3]) # devuelve P ordenado de abajo hacia arriba -> esto implica que para cada z, P esta ordenado de mayor a menor k

    l = len(k_vals)
    i = random.randint(0, 303) 
    z = z_vals[i]
    k_list = np.logspace(np.log10(0.00001), np.log10(30), 700) # va de 0.00001 a 30 en espacios logaritmicos

    P_new = []

    for i, z in enumerate(z_vals):
        A = P_vals[i*l : l*(i+1)]
        P_new.append(list(np.flip(A))) # tengo que voltear P para alinearlo luego con el orden de k_vals (de menor a mayor)  

    dP_dk = []

    for i, z in enumerate(z_vals):
        Pz = np.array(P_new[i])        # P(k) para un z fijo
        dP = np.gradient(Pz, k_vals)   # derivada respecto a k
        dP_dk.append(dP)

    dP_dk = np.array(dP_dk)  # shape: (len(z_vals), len(k_vals))


    dP_dk_inter = dPdk_interpolation(path)
    dP_dk_evaluated = dP_dk_inter(z, np.log(k_list))

    plt.plot(k_vals, dP_dk[i,:], '-', color = 'purple', label = 'dP/dk Data')
    plt.plot(k_list, dP_dk_evaluated[0,:], '--', color = 'blue', label = 'dP/dk interp')
    plt.xscale('log')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('dP/dk [(Mpc/h)^3]')#
    plt.title('Derivative of ' + path + ' at z=' + str(z))
    plt.grid()
    plt.legend()
    plt.show()


# ----- LUMINOSITY DISTANCE -----
def luminosity():
    data = np.loadtxt("scaledmeanlum-E2Sa.dat")  # o "\t" para tabulaciones

    # Filtrar por un redshift específico, por ejemplo z = 2.5

    # Extraer columnas
    z_list = data[:, 0]       # columna de k
    Lum = data[:, 1]   # columna de derivadas con respecto a h

    return z_list, Lum

def Lumo():
    '''
    k -> normal
    L -> normal
    '''
    z, L = luminosity()
    Lumo = interp1d(z, L, fill_value='extrapolate')
    print('Interpolation of luminosity function created.')
    return Lumo
