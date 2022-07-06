import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps

#Constants
mp = 1.6726219e-24 #g - proton mass
mpi0 = 2.4063e-28 #g - pi0 meson
c = 2.99792458e10 #cm/s - speed of light
mp   = 0.9382720813#; //[GeV]
mpi0 = 0.1349770#;    //[GeV]
mpip = 0.13957061#;   //[GeV]

#Parameters
doppler = 30.0
Gamma = doppler / 2.0
r_b = 1.0e+17 #cm
z = 0.3365
d_l = 5.643309e+27 #cosmology.luminosity_distance(z).to(u.cm)
b = 6.900000e+01
B = 6.900000e+01


############################################## functions ################################################################

SMALLD = 1.0e-200

NEUTRINO_OSCILLATION_FACTOR =0.33333333
KELNER_KOEFFICIENT =1.1634328 * 1.3


def proton_spectrum(E_p_eV, p_p, E_cut_p):

    E_ref = 1.0 #eV
    if (E_cut_p < 0):
        return np.power(E_p_eV/E_ref, -p_p)

    else:
        return (np.power(E_p_eV/E_ref, -p_p) * np.exp(-E_p_eV / E_cut_p))


def interpolate_param(particle,eta,etav,eta0,sv,dv,Bv):        
        nv = len(etav)
        r_eta = eta/eta0
        
        if r_eta>etav[0] and r_eta<etav[nv-1]:
            int_s = interp1d(etav,sv,kind='cubic')
            st = int_s(r_eta)
                
            int_d = interp1d(etav,dv,kind='cubic')
            dt = int_d(r_eta)
                
            int_b = interp1d(etav,Bv,kind='cubic')
            Bt = int_b(r_eta)
                
        if (r_eta<=etav[0]):
            st= sv[0]
            dt= dv[0]
            Bt= Bv[0]
            if particle == 'electron' or particle == 'anti-nu-e':
                rho= r_eta/eta0;
                rho0= etav[0]/eta0;
                Bt= Bv[0]*(rho-2.14)/(rho0-2.14)
        if (r_eta>=etav[nv-1]):
            st= sv[nv-1]
            dt= dv[nv-1]
            Bt= Bv[nv-1]

        return st,dt,Bt

def calculate_F(particle,eta,x):
   
    if particle=='gamma':
        #parameters
        mpi= mpi0
        M= mp
        r= mpi/mp
        eta0= 2.0*r+r*r
        xt1 = 2.0*(1.0+eta)
        xt2 = eta+r*r
        xt31= eta-r*r-2.0*r
        xt32= eta-r*r+2.0*r
        xt3 = xt31*xt32
        xp= (1.0/xt1)*(xt2+np.sqrt(xt3))
        xm= (1.0/xt1)*(xt2-np.sqrt(xt3))
       
        #Interpolate parameters
        param = np.loadtxt('gammaparam.txt',unpack=True,delimiter=';')
        etag = param[0,:]
        sg = param[1,:]
        dg = param[2,:]
        Bg = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etag,eta0,sg,dg,Bg)
       
        #Calculation of F
        p= 2.5+0.4*np.log(eta/eta0);
        if (x<=xm):
            F= Bt*np.power(np.log(2.0),p);
        elif ((x>xm) and (x<xp)):
            y= (x-xm)/(xp-xm)
            t11 = np.log(x/xm)
            t1= np.exp(-st*np.power(t11,dt))
            t21= np.log(2.0/(1.0+y*y))
            t2= np.power(t21,p)
            F= Bt*t1*t2
        elif (x>=xp):
            F= 0.0
        return F
   
    if particle=='positron':
        mpi= mpip
        M= mp
        r= mpi/mp
        R= M/mp
        eta0= 2.0*r+r*r
       
        xt1 = 2.0*(1.0+eta);
        xt2 = eta+r*r;
        xt31= eta-r*r-2.0*r;
        xt32= eta-r*r+2.0*r;
        xt3 = xt31*xt32;
        xp= 0.0;
        xm= 0.0;
        f = 0;
        if (xt3>SMALLD):
            xp= (1.0/xt1)*(xt2+np.sqrt(xt3))
            xm= (1.0/xt1)*(xt2-np.sqrt(xt3))
        else:
            f = 1
        xps= xp
        xms= xm/4.0
       
        #Interpolate parameters
        param = np.loadtxt('positronparam.txt',unpack=True,delimiter=';')
        etap = param[0,:]
        sp = param[1,:]
        dp = param[2,:]
        Bp = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etap,eta0,sp,dp,Bp)
       
        #Calculation of F
        p= 2.5+1.4*np.log(eta/eta0);
        if (x<=xms):
            F= Bt*np.power(np.log(2.0),p)
        elif ((x>xms) and (x<xps)):
            ys= (x-xms)/(xps-xms)
            t11= np.log(x/xms)
            t1= np.exp(-st*np.power(t11,dt))
            t21= np.log(2.0/(1.0+ys*ys))
            t2= np.power(t21,p)
            F= Bt*t1*t2
        elif (x>=xps):
            F= 0.0
        return F
       
    if particle=='electron':
        f = 0
        mpi= mpip
        M= mp
        r= mpi/mp
        R= M/mp
        eta0= 2.0*r+r*r
       
        xt1 = 2.0*(1.0+eta)
        xt2 = eta-2.0*r
        xt3 = eta*(eta-4.0*r*(1.0+r))
        xmax= 0.0
        xmin= 0.0
       
        if (xt3>SMALLD):
            xmax= (1.0/xt1)*(xt2+np.sqrt(xt3));
            xmin= (1.0/xt1)*(xt2-np.sqrt(xt3));
        else:
            f = 1
        xps= xmax;
        xms= xmin/2.0;
       
        #Interpolate parameters
        param = np.loadtxt('electronparam.txt',unpack=True,delimiter=';')
        etae = param[0,:]
        se = param[1,:]
        de = param[2,:]
        Be = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etae,eta0,se,de,Be)
       
        #Calculation of F
        rho= eta/eta0;
        p= 6.0*(1.0-np.exp(1.5*(4.0-rho)))
        if (rho<4.0):
            p= 0.0
   
        if (x<=xms):
            F= Bt*np.power(np.log(2.0),p)
        elif ((x>xms) and (x<xps)):
            ys= (x-xms)/(xps-xms)
            t11= np.log(x/xms)
            t1= np.exp(-st*np.power(t11,dt))
            t21= np.log(2.0/(1.0+ys*ys))
            t2= np.power(t21,p)
            F= Bt*t1*t2
        elif (x>=xps):
            F= 0.0
        if (rho<2.14):
            F= 0.0
        return F
   
    if particle=='nu-mu':
        mpi= mpip;
        M= mp;
        r= mpi/mp;
        eta0= 2.0*r+r*r;
        xt1 = 2.0*(1.0+eta)
        xt2 = eta+r*r
        xt31= eta-r*r-2.0*r
        xt32= eta-r*r+2.0*r
        xt3 = xt31*xt32

        f = 0
        if (xt3<0.0):
            xp= (1.0/xt1)*xt2; #verify
            xm= (1.0/xt1)*xt2; #verify
            f= 1
   
        if (xt3>0.0):
            xp= (1.0/xt1)*(xt2+np.sqrt(xt3))
            xm= (1.0/xt1)*(xt2-np.sqrt(xt3))
       
        rho= eta/eta0;
        if (rho<2.14):
            xps= 0.427*xp
        elif (rho<10.0):
            xps= (0.427+0.0729*(rho-2.14))*xp
        else:
            xps= xp
        
        xms= 0.427*xm
       
        #Interpolate parameters
        param = np.loadtxt('numuparam.txt',unpack=True,delimiter=';')
        etanm = param[0,:]
        snm = param[1,:]
        dnm = param[2,:]
        Bnm = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etanm,eta0,snm,dnm,Bnm)
       
        #Calculate F
        p= 2.5+1.4*np.log(eta/eta0);
        if (x<=xms):
            F= Bt*np.power(np.log(2.0),p)
        elif ((x>xms) and (x<xps)):
            ys= (x-xms)/(xps-xms)
            t11= np.log(x/xms)
            t1= np.exp(-st*np.power(t11,dt))
            t21= np.log(2.0/(1.0+ys*ys))
            t2= np.power(t21,p)
            F= Bt*t1*t2
        elif (x>=xps):
            F= 0.0
        return F
       
    if particle=='anti-nu-mu':
        mpi= mpip;
        M= mp;
        r= mpi/mp;
        R= M/mp;
        eta0= 2.0*r+r*r;
        antif = 0;
        xt1 = 2.0*(1.0+eta);
        xt2 = eta+r*r;
        xt31= eta-r*r-2.0*r;
        xt32= eta-r*r+2.0*r;
        xt3 = xt31*xt32;
        xp= 0.0;
        xm= 0.0;
        if (xt3>SMALLD):
            xp= (1.0/xt1)*(xt2+np.sqrt(xt3));
            xm= (1.0/xt1)*(xt2-np.sqrt(xt3))
        else:
            antif = 1;
        xps= xp
        xms= xm/4.0
       
        #Interpolate parameters
        param = np.loadtxt('antinumuparam.txt',unpack=True,delimiter=';')
        etaanm = param[0,:]
        sanm = param[1,:]
        danm = param[2,:]
        Banm = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etaanm,eta0,sanm,danm,Banm)
       
        p= 2.5+1.4*np.log(eta/eta0);
        if (x<=xms):
            F= Bt*np.power(np.log(2.0),p)
        elif ((x>xms) and (x<xps)):
            ys= (x-xms)/(xps-xms)
            t11= np.log(x/xms)
            t1= np.exp(-st*np.power(t11,dt))
            t21= np.log(2.0/(1.0+ys*ys))
            t2= np.power(t21,p)
            F= Bt*t1*t2
        elif (x>=xps):
            F= 0.0
        return F
       
       
    if particle=='nu-e':  
        mpi = mpip;
        M = mp;
        r = mpi/mp;
        R = M/mp;
        eta0 = 2.0*r+r*r;
        f = 0
        xt1 = 2.0*(1.0+eta);
        xt2 = eta+r*r;
        xt31= eta-r*r-2.0*r;
        xt32= eta-r*r+2.0*r;
        xt3 = xt31*xt32;
        if (xt3>0):
            xp= (1.0/xt1)*(xt2+np.sqrt(xt3));
            xm= (1.0/xt1)*(xt2-np.sqrt(xt3));
        else:
            f = 1;
            xp = 0.0 #verify
            xm = 0.0 #verify
        xps= xp;
        xms= xm/4.0;
       
        #Interpolate parameters
        param = np.loadtxt('nueparam.txt',unpack=True,delimiter=';')
        etanue = param[0,:]
        snue = param[1,:]
        dnue = param[2,:]
        Bnue = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etanue,eta0,snue,dnue,Bnue)
       
        #Calculate F
        p= 2.5+1.4*np.log(eta/eta0);
        if (x<=xms):
            F= Bt*np.power(np.log(2.0),p);
        elif ((x>xms) and (x<xps)):
            ys= (x-xms)/(xps-xms);
            t11= np.log(x/xms);
            t1= np.exp(-st*np.power(t11,dt));
            t21= np.log(2.0/(1.0+ys*ys));
            t2= np.power(t21,p);
            F= Bt*t1*t2;
        elif (x>=xps):
            F= 0.0
        return F
       
    if particle=='anti-nu-e':  
        mpi = mpip;
        M = mp;
        r = mpi/mp;
        R = M/mp;
        eta0 = 2.0*r+r*r;
        antif = 0
        xt1 = 2.0*(1.0+eta);
        xt2 = eta-2.0*r;
        xt3= eta*(eta-4.0*r*(1.0+r));
        if (xt3>0):
            xmax= (1.0/xt1)*(xt2+np.sqrt(xt3));
            xmin= (1.0/xt1)*(xt2-np.sqrt(xt3));
        else:
            antif = 1;
            xmax = 0.0 #verify
            xmin = 0.0 #verify
        xps= xmax;
        xms= xmin/2.0;
       
        #Interpolate parameters
        param = np.loadtxt('antinueparam.txt',unpack=True,delimiter=';')
        etaanue = param[0,:]
        sanue = param[1,:]
        danue = param[2,:]
        Banue = param[3,:]
        st,dt,Bt = interpolate_param(particle,eta,etaanue,eta0,sanue,danue,Banue)
       
        #Calculate F
        rho= eta/eta0;
        p= 6.0*(1.0-np.exp(1.5*(4.0-rho)))
        if (rho<4.0):
            p= 0.0

        if (x<=xms):
            F= Bt*np.power(np.log(2.0),p)
        elif ((x>xms) and (x<xps)):
            ys= (x-xms)/(xps-xms)
            t11= np.log(x/xms)
            t1= np.exp(-st*np.power(t11,dt))
            t21= np.log(2.0/(1.0+ys*ys))
            t2= np.power(t21,p)
            F= Bt*t1*t2
        elif (x>=xps):
            F= 0.0
        if (rho<2.14):
            F= 0.0
        return F
       
    return 'wrong particle definition'




def Integrate(energy_proton, proton_energy_number, args):

    N_PROTON = args[0]
    SIZE_ENERGY_NEUTRINO = args[1]
    SIZE_ENERGY_ELECTRON = args[2]
    SIZE_ENERGY_GAMMA = args[3]
    MINIMAL_FRACTION = args[4]
    SIZE_X = args[5]
    SIZE_N_PHOTONS_SYNCHRO = args[6]

    a_frac = np.log10(1.0/MINIMAL_FRACTION) / SIZE_X

    SED_e_init = np.zeros((SIZE_X))
    SED_n_init = np.zeros((SIZE_X))
    SED_g_init = np.zeros((SIZE_X))

    Ep = energy_proton[proton_energy_number]/1.0e+09 # eV -> GeV
    pi = 3.141592654
    GeV= 1.0e-9#				//[eV]

    mpi= mpi0
    M= mp
    r= mpi/mp
    eta0= 2.0*r+r*r
    eps0= 1.0e9*(eta0*mp*mp)/(4.0*Ep)#;		//energy threshold [eV]

    synchro = np.loadtxt('field.txt',unpack=True,delimiter=';')
    epsilon_synchro = synchro[0,:]
    nph_synchro = synchro[1,:]

    for j in range(1,SIZE_X):
        sg  = 0.0
        sp  = 0.0
        se  = 0.0
        snm = 0.0
        sanm= 0.0
        sne = 0.0
        sane= 0.0

        x = MINIMAL_FRACTION*pow(10.0,a_frac*np.array(j))#; //x= (E_{gamma}/E_{p}); (E_e/E_{p}), (...)
        for i in range(1,SIZE_N_PHOTONS_SYNCHRO):

            if (epsilon_synchro[i] > eps0):

                eps = epsilon_synchro[i]
                f = nph_synchro[i]#; //number density of photon field [1/(cm^{3}*eV)]

                eta= (4.0*GeV*eps*Ep)/(mp*mp)#;		 //approximation parameter: Phi(eta,x) -> here F(eta,x)

                deps = epsilon_synchro[i] - epsilon_synchro[i-1]

                f_g = calculate_F('gamma',eta,x)
                f_e = calculate_F('electron',eta,x)
                f_p = calculate_F('positron',eta,x)
                f_nm = calculate_F('nu-mu',eta,x)
                f_anm = calculate_F('anti-nu-mu',eta,x)
                f_ne = calculate_F('nu-e',eta,x)
                f_ane = calculate_F('anti-nu-e',eta,x)

                sg = sg + f * f_g * deps
                sp = sp + f * f_p * deps
                se = se + f * f_e * deps
                snm = snm + f * f_nm * deps
                sanm = sanm + f * f_anm * deps
                sne = sne + f * f_ne * deps
                sane = sane + f * f_ane * deps

        temp = x*x*energy_proton[proton_energy_number]#//*1.0e+09;
        SED_e_init[j] = temp*(sp + se) * KELNER_KOEFFICIENT;
        SED_n_init[j] = temp*(snm + sanm + sne + sane) * KELNER_KOEFFICIENT;
        SED_g_init[j] = temp*sg * KELNER_KOEFFICIENT;

    SEDS_init = []
    SEDS_init.append(SED_e_init)
    SEDS_init.append(SED_n_init)
    SEDS_init.append(SED_g_init)

    return SEDS_init



def SEDIntermediate(energy_proton, a_frac, args, final_energy, SEDS_initial):

    N_PROTON = args[0]
    SIZE_ENERGY_NEUTRINO = args[1]
    SIZE_ENERGY_ELECTRON = args[2]
    SIZE_ENERGY_GAMMA = args[3]
    MINIMAL_FRACTION = args[4]
    SIZE_X = args[5]

    energy_neutrino_final = final_energy[0]
    energy_electron_final = final_energy[1]
    energy_gamma_final = final_energy[2]

    SED_neutrino_init = SEDS_initial[0]
    SED_electron_init = SEDS_initial[1]
    SED_gamma_init = SEDS_initial[2]

    SED_neutrino_intermediate = np.zeros((SIZE_ENERGY_NEUTRINO,N_PROTON))
    SED_electron_intermediate = np.zeros((SIZE_ENERGY_ELECTRON,N_PROTON))
    SED_gamma_intermediate = np.zeros((SIZE_ENERGY_GAMMA,N_PROTON))

    #neutrinos
    for j in range(0,N_PROTON):
        for l in range(0,SIZE_ENERGY_NEUTRINO):
            x = energy_neutrino_final[l] / energy_proton[j]#;// * 1.0e-09;
            k = int(np.log10(x/MINIMAL_FRACTION)/a_frac)#;
            #if ((k>0) and (k<SIZE_ENERGY_NEUTRINO)):
            if ((k>0) and (k<SIZE_X)):
                SED_neutrino_intermediate[l][j] = SED_neutrino_init[k][j];

    #electrons (+ positrons)
    for j in range(0,N_PROTON):
        for l in range(0,SIZE_ENERGY_ELECTRON):
            x = energy_electron_final[l] / energy_proton[j]#;// * 1.0e-09;
            k = int(np.log10(x/MINIMAL_FRACTION)/a_frac)#;
            #if ((k>0) and (k<SIZE_ENERGY_ELECTRON)):
            if ((k>0) and (k<SIZE_X)):
                SED_electron_intermediate[l][j] = SED_electron_init[k][j]

    #gamma-rays
    for j in range(0,N_PROTON):
        for l in range(0,SIZE_ENERGY_GAMMA):
            x = energy_gamma_final[l] / energy_proton[j]#;// * 1.0e-09;
            k = int(np.log10(x/MINIMAL_FRACTION)/a_frac)#;
            #if ((k>0) and (k<SIZE_ENERGY_GAMMA)):
            if ((k>0) and (k<SIZE_X)):
                SED_gamma_intermediate[l][j] = SED_gamma_init[k][j];

    SEDS_intermediate = []
    SEDS_intermediate.append(SED_neutrino_intermediate)
    SEDS_intermediate.append(SED_electron_intermediate)
    SEDS_intermediate.append(SED_gamma_intermediate)

    return SEDS_intermediate


def FinalSED(energy_proton, p_p, E_cut, args, final_energy, SEDS_intermediate):

    N_PROTON = args[0]
    SIZE_ENERGY_NEUTRINO = args[1]
    SIZE_ENERGY_ELECTRON = args[2]
    SIZE_ENERGY_GAMMA = args[3]
    MINIMAL_FRACTION = args[4]
    SIZE_X = args[5]

    energy_neutrino_final = final_energy[0]
    energy_electron_final = final_energy[1]
    energy_gamma_final = final_energy[2]

    SED_neutrino_final = np.zeros((SIZE_ENERGY_NEUTRINO))
    SED_electron_final = np.zeros((SIZE_ENERGY_ELECTRON))
    SED_gamma_final = np.zeros((SIZE_ENERGY_GAMMA))

    SED_neutrino_intermediate = SEDS_intermediate[0]
    SED_electron_intermediate = SEDS_intermediate[1]
    SED_gamma_intermediate = SEDS_intermediate[2]

    #neutrinos
    for l in range(0, SIZE_ENERGY_NEUTRINO):
        for j in range(1,N_PROTON):
            SED_neutrino_final[l] += SED_neutrino_intermediate[l][j] * proton_spectrum(energy_proton[j], p_p, E_cut)*(energy_proton[j] - energy_proton[j-1])

    maxn = 0.0
    for l in range(0, SIZE_ENERGY_NEUTRINO):
        SED_neutrino_final[l] = SED_neutrino_final[l] * NEUTRINO_OSCILLATION_FACTOR;
        if (maxn < SED_neutrino_final[l]):
            maxn = SED_neutrino_final[l]

    np.savetxt('neutrino_SED.txt',np.c_[energy_neutrino_final,SED_neutrino_final])

    #electrons (+ positrons)
    for l in range(0, SIZE_ENERGY_ELECTRON):
        for j in range(1,N_PROTON):
            SED_electron_final[l] += SED_electron_intermediate[l][j] * proton_spectrum(energy_proton[j], p_p, E_cut)*(energy_proton[j] - energy_proton[j-1]);

    maxe = 0.0
    for l in range(0, SIZE_ENERGY_ELECTRON):
        if (maxe < SED_electron_final[l]):
            maxe = SED_electron_final[l]

    np.savetxt('electron_SED.txt',np.c_[energy_electron_final,SED_electron_final])



    #gamma-rays
    for l in range(0, SIZE_ENERGY_GAMMA):
        for j in range(1,N_PROTON):
            SED_gamma_final[l] += SED_gamma_intermediate[l][j] * proton_spectrum(energy_proton[j], p_p, E_cut)*(energy_proton[j] - energy_proton[j-1]);

    maxg = 0.0
    for l in range(0, SIZE_ENERGY_GAMMA):
        if (maxg < SED_gamma_final[l]):
            maxg = SED_gamma_final[l]

    np.savetxt('gamma_SED.txt',np.c_[energy_gamma_final,SED_gamma_final])

    return 0


def axes_reshaper(*args):
    n = len(args)
    dim = (1,) * n
    reshaped_arrays = []
    for i, arg in enumerate(args):
        reshaped_dim = list(dim)  # the tuple is copied in the list
        reshaped_dim[i] = arg.size
        reshaped_array = np.reshape(arg, reshaped_dim)
        reshaped_arrays.append(reshaped_array)
    return reshaped_arrays


def power_law(en, gamma, norm):
    en_ref = 1.0 #eV
    f = norm * (en / en_ref) ** (-gamma)
    return f


def process(energy_proton_min_ext, energy_proton_max_ext, p_p_ext, E_cut_ext, args):

    print("Performing calculations of proton-photon meson production secondaries SED")
    print('')

    N_PROTON = args[0]
    SIZE_ENERGY_NEUTRINO = args[1]
    SIZE_ENERGY_ELECTRON = args[2]
    SIZE_ENERGY_GAMMA = args[3]
    MINIMAL_FRACTION = args[4]
    SIZE_X = args[5]
    SIZE_N_PHOTONS_SYNCHRO = args[6]

    SED_electron_init = np.zeros((SIZE_X,N_PROTON))
    SED_neutrino_init = np.zeros((SIZE_X,N_PROTON))
    SED_gamma_init = np.zeros((SIZE_X,N_PROTON))
    SED_electron_final = np.zeros((SIZE_ENERGY_ELECTRON))
    SED_neutrino_final = np.zeros((SIZE_ENERGY_NEUTRINO))
    SED_gamma_final = np.zeros((SIZE_ENERGY_GAMMA))
    energy_electron_final = np.zeros((SIZE_ENERGY_ELECTRON))
    energy_neutrino_final = np.zeros((SIZE_ENERGY_NEUTRINO))
    energy_gamma_final = np.zeros((SIZE_ENERGY_GAMMA))
    energy_proton = np.zeros((N_PROTON))

    energy_proton_min = energy_proton_min_ext
    energy_proton_max = energy_proton_max_ext
    p_p = p_p_ext
    E_cut = E_cut_ext

    a_p = np.log10(energy_proton_max / energy_proton_min) / N_PROTON

    energy_neutrino_min = 1.0e+12#; //eV
    energy_neutrino_max = energy_proton_max * 0.75#; // eV
    a_neu = np.log10(energy_neutrino_max/energy_neutrino_min)/SIZE_ENERGY_NEUTRINO#;

    energy_electron_min = 1.0e+12#; // eV
    energy_electron_max = energy_proton_max * 0.75#; // eV
    a_e = np.log10(energy_electron_max/energy_electron_min)/SIZE_ENERGY_ELECTRON#;

    energy_gamma_min = 1.0e+12#; // eV
    energy_gamma_max = energy_proton_max * 0.75#; // eV
    a_g = np.log10(energy_gamma_max/energy_gamma_min)/SIZE_ENERGY_GAMMA#;

    #neutrinos
    for k in range(0,SIZE_ENERGY_NEUTRINO):
        SED_neutrino_final[k] = 0.0;

    for k in range(0,SIZE_ENERGY_NEUTRINO):
        energy_neutrino_final[k] = energy_neutrino_min*np.power(10.0, a_neu*k);

    #electrons (+ positrons)
    for k in range(0,SIZE_ENERGY_ELECTRON):
        SED_electron_final[k] = 0.0;

    for k in range(0,SIZE_ENERGY_ELECTRON):
        energy_electron_final[k] = energy_electron_min*np.power(10.0, a_e*k);

    #gamma-rays
    for k in range(0,SIZE_ENERGY_GAMMA):
        SED_gamma_final[k] = 0.0;

    for k in range(0,SIZE_ENERGY_GAMMA):
        energy_gamma_final[k] = energy_gamma_min*np.power(10.0, a_g*k);

    #primary protons
    for k in range(0,N_PROTON):
        energy_proton[k] = energy_proton_min*np.power(10.0, a_p*k);

    for k in range(0,N_PROTON):
        SEDS_init = Integrate(energy_proton, k, args)
        SED_electron_init[:,k] = SEDS_init[0]
        SED_neutrino_init[:,k] = SEDS_init[1]
        SED_gamma_init[:,k] = SEDS_init[2]
        print ('Integrating proton energy=',k,'/',N_PROTON)

    SEDS_initial = []
    SEDS_initial.append(SED_neutrino_init)
    SEDS_initial.append(SED_electron_init)
    SEDS_initial.append(SED_gamma_init)

    final_energy = []
    final_energy.append(energy_neutrino_final)
    final_energy.append(energy_electron_final)
    final_energy.append(energy_gamma_final)

    a_frac = np.log10(1.0/MINIMAL_FRACTION) / SIZE_X

    SEDS_intermediate = SEDIntermediate(energy_proton, a_frac, args, final_energy, SEDS_initial)


    FinalSED(energy_proton, p_p, E_cut, args, final_energy, SEDS_intermediate)

    return 0


def photohadron(energy_proton_min, energy_proton_max, p_p, E_cut):

    #constants
    IntegrationLogFlag=0
    B01SSCFlag=0
    SIZE_N_PHOTONS_SYNCHRO=100
    SIZE_X=500
    SIZE_ENERGY_NEUTRINO=5000
    SIZE_ENERGY_ELECTRON=5000
    SIZE_ENERGY_GAMMA=5000
    N_PROTON =50 #//100
    NEUTRINO_OSCILLATION_FACTOR =0.33333333
    MINIMAL_FRACTION =1.0e-04
    KELNER_KOEFFICIENT =1.1634328 * 1.3#// now CMB normalization is fixed!
    PROTON_REST_ENERGY =9.38272e+08 #// eV

    E_ref = 1.0 #eV

    a_frac = np.log10(1.0/MINIMAL_FRACTION) / SIZE_X

    args = []
    args.append(N_PROTON)
    args.append(SIZE_ENERGY_NEUTRINO)
    args.append(SIZE_ENERGY_ELECTRON)
    args.append(SIZE_ENERGY_GAMMA)
    args.append(MINIMAL_FRACTION)
    args.append(SIZE_X)
    args.append(SIZE_N_PHOTONS_SYNCHRO)
    #process(energy_proton_min_ext, energy_proton_max_ext, p_p_ext, E_cut_ext, sizes)
    process(energy_proton_min, energy_proton_max, p_p, E_cut, args)

    return 0


def kelner_pgamma_calculate( field , energy_proton_min , energy_proton_max , p_p , e_cut_p=-1 , C_p=1.0 ):

    if type(field) == type(''):
        field = np.loadtxt(field)
        field[:, 0] = field[:, 0]# * energy_coef
        field[:, 1] = field[:, 1]# * dens_coef
    elif type(field) == type(np.array(([2, 1], [5, 6]))):
        field[:, 0] = field[:, 0]# * energy_coef
        field[:, 1] = field[:, 1]# * dens_coef

    photon_field_path = 'field.txt'
    np.savetxt(photon_field_path, field, delimiter=';', fmt='%.6e')

    ###########################################################################
    e_cut_value = e_cut_p
    ###########################################################################
    photohadron(energy_proton_min, energy_proton_max, p_p, e_cut_p)
    ###########################################################################
    neutrino = np.loadtxt('neutrino_SED.txt')
    neutrino_e = neutrino[0,:] #* u.eV
    neutrino_sed = neutrino[1,:] #* (u.eV * u.s**(-1))
    #neutrino_sed = neutrino_sed * C_p# / (1.0 / u.eV)
    ###########################################################################
    electron = np.loadtxt('electron_SED.txt')
    electron_e = electron[0,:] #* u.eV
    electron_sed = electron[1,:] #* (u.eV * u.s**(-1))
    #electron_sed = electron_sed * C_p# / (1.0 / u.eV)
    ###########################################################################
    gamma = np.loadtxt('gamma_SED.txt')
    gamma_e = gamma[0,:] #* u.eV
    gamma_sed = gamma[1,:] #* (u.eV * u.s**(-1))
    #gamma_sed = gamma_sed * C_p #/ (1.0 / u.eV)
    ###########################################################################
    
    return neutrino_e, neutrino_sed, electron_e, electron_sed, gamma_e, gamma_sed


##############################################################################################################


#External field definition
en_ext = np.logspace(np.log10(1.33), np.log10(10.0), 100) #eV
en_ext_blob = 4.0 / 3.0 * Gamma * en_ext
boost = 2.0 * 4.0 / 3.0 * Gamma**2
alpha = 2.0
K = 3e+03 #/ (u.eV * u.cm**3)
n_ext = power_law(en_ext, alpha, norm=K)
n_ext_blob = n_ext * boost
field_ext = np.concatenate((en_ext_blob.reshape(en_ext_blob.shape[0], 1), n_ext_blob.reshape(n_ext_blob.shape[0], 1)), axis=1)

#Proton parameters
energy_proton_min = 1.0e14 # u.eV  # 3.0e+14 * u.eV
energy_proton_max = 6.0e14 # u.eV
en_p = np.logspace(np.log10(energy_proton_min), np.log10(energy_proton_max), 100)
p_p = 2.0
C_p = 2.5e65 #* u.eV**(-1) 1/eV
proton_spectrum1 = power_law(en_p, p_p, norm=C_p)
u_p = (simps(proton_spectrum1 * en_p, en_p) / (4.0 / 3.0 * np.pi * r_b**3))
u_p = u_p * 1.60218e-12 #convert from eV/cm3 to erg/cm3
print("proton energy density in the blob = {:.6e}".format(u_p),"erg/cm3")
L_p = np.pi * r_b**2 * c * (doppler / 2.0)**2 * u_p
L_p = L_p * 1.60218e-12 #convert from eV/s to erg/s
print("observable proton luminosity in the lab frame = {:.6e}".format(L_p),"erg/s")
u_b = (b**2 / (8.0 * np.pi))#.to(u.erg / u.cm**3)
print("magnetic field density in the blob = {:.6e}".format(u_b),"erg/cm3")

(neutrino_e, neutrino_sed, electron_e, electron_sed, gamma_e, gamma_sed) = kelner_pgamma_calculate(field_ext, energy_proton_min, energy_proton_max, p_p, e_cut_p=-1, C_p=C_p)


###################### plot ########################

dfn = np.loadtxt('neutrino_SED.txt',unpack=True,delimiter=' ')
dfe = np.loadtxt('electron_SED.txt',unpack=True,delimiter=' ')
dfg = np.loadtxt('gamma_SED.txt',unpack=True,delimiter=' ')

fig, ax = plt.subplots()
plt.plot(dfn[0,:], dfn[1,:],'-.',label='Neutrino + Anti-neutrino',linewidth=2,color='red')
plt.plot(dfe[0,:], dfe[1,:],'--',label='Electron + Positron',linewidth=2,color='blue')
plt.plot(dfg[0,:], dfg[1,:],'-',label='Gamma-rays',linewidth=2,color='black')
plt.legend(loc='best',numpoints=1)
plt.grid(linestyle = '--',linewidth = 0.5)
plt.xlabel(r'E (eV)')
plt.ylabel(r'E/t (eV/s)')
ax.set_yscale('log')
ax.set_xscale('log')
plt.savefig('photohadronic_sed.png')
plt.show()
plt.close()
