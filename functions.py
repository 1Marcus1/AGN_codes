import numpy as np
import matplotlib.pyplot as plt

gamma_min = 8.0e2 #lorentz factor
gamma_max = 1.0e8 #lorentz factor
gamma_br  = np.array([5.0e4, 3.9e5])
gamma_cut = np.array([1.0e5])

p = np.array([2.2,2.7,4.7]) #pot - power law

#Power laws
powerlaw_S = "Simple"
powerlaw_B = "Broken"
powerlaw_L = "Log-Parabola"
powerlaw_E = "Exponential-Cutoff"
powerlaw_2B = "2breaks"

ke = 5.0e2 #normalization factor

raio = 5.2e16 #cm - radius of emitting region
GAMMA = 20.0 #Lorentz factor

deltaD = 21.0 #doppler boosting

B = 3.8e-2 #gauss - magnetic field

CD = 1.0 #compton dominance

#Redshift
z =  0.031

distance = 3.2e26 #cm - Luminosity distance (distance to center of emitting region)

me = 9.1094e-28 #g - electron mass
c = 2.99792458e10 #cm/s - speed of light
hp = 6.62607015e-27 #erg s - planck constant
e = 4.803204e-10 #esu - elementary charge
sigma_T = 6.652459e-25 #cm^2 - Thomson cross-section


#frequency distribution
nu = np.logspace(8,28, 200) #Hz - frequency for Synchrotron
nu2 = np.logspace(5,28, 200) #Hz - frequency to integrate

def R(x):
    #Eq. 7.45 in Reference, using approximation Eq. D7 of https://arxiv.org/abs/1006.1045
    t1 = (1.808 * np.power(x, 1 / 3)) / (np.sqrt(1 + 3.4 * np.power(x, 2 / 3)))
    t2 = (1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)) / (1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3))
    t3 = np.exp(-x)
    return t1 * t2 * t3

def axes_reshaper(*args):
    """reshape 1-dimensional arrays of different lengths in order for them to be
    broadcastable in multi-dimensional operations
    the rearrangement scheme for a list of n arrays is the following:
    `args[0]` is reshaped as `(args[0].size, 1, 1, ..., 1)` -> axis 0
    `args[1]` is reshaped as `(1, args[1].size, 1, ..., 1)` -> axis 1
    `args[2]` is reshaped as `(1, 1, args[2].size ..., 1)` -> axis 2
        .
        .
        .
    `args[n-1]` is reshaped as `(1, 1, 1, ..., args[n-1].size)` -> axis n-1
    Parameters
    ----------
    args: 1-dimensional `~numpy.ndarray`s to be reshaped
    """
    n = len(args)
    dim = (1,) * n
    reshaped_arrays = []
    for i, arg in enumerate(args):
        reshaped_dim = list(dim)  # the tuple is copied in the list
        reshaped_dim[i] = arg.size
        reshaped_array = np.reshape(arg, reshaped_dim)
        reshaped_arrays.append(reshaped_array)
    return reshaped_arrays

################################################ Flux SED for Synchrotron ################################################

x_synch = np.array([gamma_min, gamma_max, gamma_br, gamma_cut, ke, p, me, c, hp, e, raio, GAMMA, deltaD, distance, B, z])

def SED_synchrotron(x_synch, nu, powerlaw):
    gamma_min = x_synch[0]
    gamma_max = x_synch[1]
    gamma_br = x_synch[2]
    gamma_cut = x_synch[3]
    ke = x_synch[4]
    p = x_synch[5]
    me = x_synch[6]
    c = x_synch[7]
    hp = x_synch[8]
    e = x_synch[9]
    raio = x_synch[10]
    GAMMA = x_synch[11]
    deltaD = x_synch[12]
    distance = x_synch[13]
    B = x_synch[14]
    z = x_synch[15]

    #adimensional frequency distribution
    nu1 = nu*hp/(me*(c**2))

    #volume of sphere
    Ve = (4/3) * np.pi * (raio**3)

    #relativistic velocity
    Beta = (1-(1/(GAMMA**2)))**(1/2)

    #jet viewing angle
    #mu = (1 - (1/(GAMMA*deltaD)))/Beta
    #theta = np.rad2deg(np.arccos(mu))

    #electron distribution
    gammadist = np.logspace(np.log10(gamma_min), np.log10(gamma_max), 200)
    epsilon = (1+z)*nu1/deltaD
    ne = np.zeros((len(gammadist)))
    
    #modify shapes of frequency and lorentz factor distributions to integrate
    gammadist1 = axes_reshaper(gammadist)
    epsilon1 = axes_reshaper(epsilon)

   
    #electrons density
    if (powerlaw=="Simple"):
        ne = (np.power(gammadist1, -p[0]))*ke #electron/cm^3
    
    elif (powerlaw=="Broken"):
        ne = np.where(gammadist1 < gamma_br[0],
                          (np.power(gammadist1, -p[0]))*ke, 
                          (np.power(gamma_br[0], p[1]-p[0]))*(np.power(gammadist1, -p[1]))*ke)
    elif (powerlaw=="Log-Parabola"):
        ne = (np.power(gammadist1, -p[0]-p[1]*np.log(gammadist1)))*ke
    elif (powerlaw=="Exponential-Cutoff"):
        ne = (np.power(gammadist1, -p[0]))*ke*np.exp(-np.power(gamma_cut,p[1]))
    elif (powerlaw=="2breaks"):
        ne = np.where(gammadist1 < gamma_br[0],
                       np.power(gammadist1, -p[0])*ke,
                    np.where(gammadist1 > gamma_br[1], ke*np.power(gamma_br[0], p[1]-p[0])*np.power(gamma_br[1], p[2]-p[1])*np.power(gammadist1, -p[2]),
                                ke*np.power(gamma_br[0], p[1]-p[0])*np.power(gammadist1, -p[1]) ))
    
    #electrons quantity    
    Ne = Ve*ne
   
    #SED
    x = np.zeros(( len(epsilon1[0]), len(gammadist1[0])))
    for i in range(0,len(epsilon1[0])):
        for j in range(0,len(gammadist1[0])):
            x[i,j] = (4*np.pi*np.array(epsilon1[0][i])*(me**2)*(c**3))/(3*e*B*hp*np.power(gammadist1[0][j],2))

    gx = R(x)

    #emissivity
    integrand = (Ne * np.sqrt(3) * (e**3) * B / hp) * gx
    j = np.zeros((len(integrand)))
    for i in range(0,len(integrand)):
        j[i] = np.trapz(integrand[i], gammadist,axis=0)

    prefactor = (deltaD**4)/(4*np.pi*(distance**2))
   
    #Flux for Synchrotron radiation - Eq. 7.116 in Reference, Eq. 21 in https://arxiv.org/abs/0802.1529
    SED = prefactor*epsilon*j

    return SED
##################################################################################################################

x_ssc = np.array([gamma_min, gamma_max, gamma_br, gamma_cut, ke, p, me, c, hp, e, raio, GAMMA, deltaD, distance, B, z, nu2])

############################################# Flux SED for SSC ###################################################
def SED_ssc(x_ssc, nu, powerlaw):
    gamma_min = x_ssc[0]
    gamma_max = x_ssc[1]
    gamma_br = x_ssc[2]
    gamma_cut = x_ssc[3]
    ke = x_ssc[4]
    p = x_ssc[5]
    me = x_ssc[6]
    c = x_ssc[7]
    hp = x_ssc[8]
    e = x_ssc[9]
    raio = x_ssc[10]
    GAMMA = x_ssc[11]
    deltaD = x_ssc[12]
    distance = x_ssc[13]
    B = x_ssc[14]
    z = x_ssc[15]
    nu2 = x_ssc[16]
    
    
    #volume of sphere
    Ve = (4/3) * np.pi * (raio**3)

    #SSC part
    integrate_nu = nu2
    synch_ssc = SED_synchrotron(x_ssc, nu2, powerlaw)
   
    #adimensional frequencies
    nu1_ssc = nu*hp/(me*(c**2))
    integrate_nu1_ssc = integrate_nu*hp/(me*(c**2))
    epsilon = (1+z)*integrate_nu1_ssc/deltaD
    epsilon_s = (1+z)*nu1_ssc/deltaD
    
    nu_ssc1 = nu*hp
    
    sigma_KN = (3/4)*sigma_T*( ((1+nu_ssc1)/(nu_ssc1**3))*((2*nu_ssc1*(1 + nu_ssc1))/(1 + 2*nu_ssc1) - np.log(1 + 2*nu_ssc1)) + (np.log(1 + 2*nu_ssc1))/(2*nu_ssc1) - (1 + 3*nu_ssc1)/((1 + 2*nu_ssc1)**2) )

    #radiation energy density - Eq. 8 in https://arxiv.org/pdf/0802.1529.pdf  
    u_synch = (3 * np.power(distance, 2) * synch_ssc) / (c * np.power(raio, 2) * np.power(deltaD, 4) * epsilon)
    _u_synch = np.reshape(u_synch, (1, u_synch.size, 1))
   
    #multidimensional integration
    gammadist = np.logspace(np.log10(gamma_min), np.log10(gamma_max), 200)

    _gamma, _epsilon, _epsilon_s = axes_reshaper(gammadist, epsilon, epsilon_s)

   
    #electrons density
    ne = np.zeros((len(_gamma)))
    
    if (powerlaw=="Simple"):
        ne = (np.power(_gamma, -p[0]))*ke #electron/cm^3
    elif (powerlaw=="Broken"):
        ne = np.where(_gamma < gamma_br[0],
                          (np.power(_gamma, -p[0]))*ke, #gamma < gamma_br
                          (np.power(gamma_br[0], p[1]-p[0]))*(np.power(_gamma, -p[1]))*ke)
    elif (powerlaw=="Log-Parabola"):
        ne = (np.power(_gamma, -p[0]-p[1]*np.log(_gamma)))*ke
    elif (powerlaw=="Exponential-Cutoff"):
        ne = (np.power(_gamma, -p[0]))*ke*np.exp(-np.power(gamma_cut,p[1]))
    
    elif (powerlaw=="2breaks"):
        ne = np.where(_gamma < gamma_br[0],
                       np.power(_gamma, -p[0])*ke,
                    np.where(_gamma > gamma_br[1], ke*np.power(gamma_br[0], p[1]-p[0])*np.power(gamma_br[1], p[2]-p[1])*np.power(_gamma, -p[2]),
                                ke*np.power(gamma_br[0], p[1]-p[0])*np.power(_gamma, -p[1]) ))

    
    Ne = Ve*ne
       
    GAMMA_e = 4 * _gamma * _epsilon #Eq. 11 in https://arxiv.org/pdf/0802.1529.pdf
    q = (_epsilon_s / _gamma) / (GAMMA_e * (1 - _epsilon_s / _gamma)) #Eq. 11 in https://arxiv.org/pdf/0802.1529.pdf
    q_min = 1 / (4 * np.power(_gamma, 2)) #Eq. 12 in https://arxiv.org/pdf/0802.1529.pdf
   
    #Isotropic Compton Kernel - Eq. 6.75 in reference, Eq. 10 in https://arxiv.org/pdf/0802.1529.pdf
    F_c = 2 * q * np.log(q) + (1 + 2 * q) * (1 - q) + 1 / 2 * np.power(GAMMA_e * q, 2) / (1 + GAMMA_e * q) * (1 - q)
    
    kernel = np.where((q_min <= q) * (q <= 1), F_c, 0)

    #SSC integrals of Eq. 7.117 in reference
    integrand = (_u_synch / np.power(_epsilon, 2) * Ne / np.power(_gamma, 2) * kernel)
    integral_gamma = np.trapz(integrand, gammadist, axis=0)
    integral_epsilon = np.trapz(integral_gamma, epsilon, axis=0)
    emissivity = np.where(((nu_ssc1)<((8.1*me*c**2))),(3 / 4 * c * sigma_T * np.power(epsilon_s, 2) * integral_epsilon),
                          (3 / 4 * c * sigma_KN * np.power(epsilon_s, 2) * integral_epsilon))
    prefactor = np.power(deltaD, 4) / (4 * np.pi * np.power(distance, 2))
    
    #Flux for Synchrotron radiation - Eq. 7.118 in Reference, Eq. 23 in https://arxiv.org/abs/0802.1529
    sed = prefactor * emissivity

    return sed
####################################################################################################################
