import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.optimize as optimization
import functions as f
from scipy.optimize import minimize

#remove 'gamma_min' parameter from fit

import time
from datetime import timedelta
start_time = time.monotonic()
#end_time = time.monotonic()
#print(timedelta(seconds=end_time - start_time))


#parameters

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

distance = 3.2e26 #cm - distance to center of emitting region

me = 9.1094e-28 #g - electron mass
c = 2.99792458e10 #cm/s - speed of light
hp = 6.62607015e-27 #erg s - planck constant
e = 4.803204e-10 #esu - elementary charge
sigma_T = 6.652459e-25 #cm^2 - Thomson cross-section


#frequency distribution
nu = np.logspace(8,28, 200) #Hz - frequency for Synchrotron
nu2 = np.logspace(5,28, 200) #Hz - frequency to integrate


#type of powerlaw
type_powerlaw = '2breaks'


#Figure
x_synch = np.array([gamma_min, gamma_max, gamma_br, gamma_cut, ke, p, me, c, hp, e, raio, GAMMA, deltaD, distance, B, z])
synch = f.SED_synchrotron(x_synch, nu, type_powerlaw)

x_ssc = np.array([gamma_min, gamma_max, gamma_br, gamma_cut, ke, p, me, c, hp, e, raio, GAMMA, deltaD, distance, B, z, nu2])
ssc = f.SED_ssc(x_ssc, nu, type_powerlaw)

synch_ssc = synch + ssc


######################################## FIT ####################################################
# Experimental data
#MRK 421
file = np.loadtxt("mrk421.csv",delimiter=";")
xdata = file[:,0]*2.417989e14
xdata1 = []
ydata = file[:,1]
ydata1 = []
error_data1 = file[:,2]
error_data = []

for i in range(0,len(xdata)):
    if xdata[i]>1.0e11:
        xdata1.append(xdata[i])
        ydata1.append(ydata[i])
        error_data.append(error_data1[i])


# Initial guess for the parameters
x0 = []
#x0.append(np.log10(gamma_min))
x0.append(np.log10(gamma_max))

if type_powerlaw=='Simple':
    x0.append(p[0])

elif type_powerlaw=='Broken':
    #x0.append(gamma_br[0])
    x0.append(p[0])
    x0.append(p[1])

elif type_powerlaw=='Log-Parabola':
    x0.append(p[0])
    x0.append(p[1])

elif type_powerlaw=='Exponential-Cutoff':
    x0.append(np.log10(gamma_cut[0]))
    x0.append(p[0])
    x0.append(p[1])

elif type_powerlaw=='2breaks':
    #x0.append(np.log10(gamma_br[0]))
    #x0.append(np.log10(gamma_br[1]))
    x0.append(p[0])
    x0.append(p[1])
    x0.append(p[2])

x0.append(deltaD)
x0.append(B)


#Log interpolation
def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

#Interpolation function
def func_interp(x1):
    x_model = np.copy(x_ssc)
    #x_model[0] = np.power(10,x1[0])
    x_model[1] = np.power(10,x1[0])
    
    if type_powerlaw=='Simple':
        x_model[5] = [x1[1]]
        x_model[12] = x1[2]
        x_model[14] = x1[3]
    elif type_powerlaw=='Broken':
        #x_model[2] = [x1[1]]
        x_model[5] = [x1[1], x1[2]]
        x_model[12] = x1[3]
        x_model[14] = x1[4]
    elif type_powerlaw=='Log-Parabola':
        x_model[5] = [x1[1],x1[2]]
        x_model[12] = x1[3]
        x_model[14] = x1[4]
    elif type_powerlaw=='Exponential-Cutoff':
        x_model[3] = [np.power(10,x1[1])]
        x_model[5] = [x1[2],x1[3]]
        x_model[12] = x1[4]
        x_model[14] = x1[5]
    elif type_powerlaw=='2breaks':
        #x_model[2] = [np.power(10,x1[1]), np.power(10,x1[2])]
        x_model[5] = [x1[1],x1[2],x1[3]]
        x_model[12] = x1[4]
        x_model[14] = x1[5]
    
    y1 = f.SED_synchrotron(x_model, nu, type_powerlaw)
    y2 = f.SED_ssc(x_model, nu, type_powerlaw)
    y_model = y1 + y2
    
    #Create interpolation model
    f_model = log_interp1d(nu, y_model)
    #Interpolate the model
    y_interp = f_model(xdata1)
    
    error = np.sum(abs(y_interp-ydata1)/ydata1)*100
    
    if type_powerlaw=='Simple':
        print('gamma_max',10**x1[0],'p[0]',x1[1],'deltaD',x1[2],'B',x1[3],'error',error)
    elif type_powerlaw=='Broken':
        print('gamma_max',10**x1[1],'p[0]',x1[2],'p[1]',x1[3],'deltaD',x1[4],'B',x1[5],'error',error)
    elif type_powerlaw=='Log-Parabola':
        print('gamma_max',10**x1[0],'p[0]',x1[1],'p[1]',x1[2],'deltaD',x1[3],'B',x1[4],'error',error)
    elif type_powerlaw=='Exponential-Cutoff':
        print('gamma_max',10**x1[0], 'gamma_cut',10**x1[1],'p[0]',x1[2],'p[1]',x1[3],'deltaD',x1[4],'B',x1[5],'error',error)
    elif type_powerlaw=='2breaks':
        print('gamma_max',10**x1[0], 'p[0]',x1[1], 'p[1]',x1[2], 'p[2]',x1[3], 'deltaD',x1[4], 'B',x1[5], 'error',error)

    return error



#Optimize
if type_powerlaw=='Simple':
    cons = ({'type': 'ineq', 'fun': lambda x:  5.1 - x[1]},
            {'type': 'ineq', 'fun': lambda x:  x[1] - 2.0})

elif type_powerlaw=='Broken':
    cons = ({'type': 'ineq', 'fun': lambda x:  5.1 - x[2]},
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]},
            {'type': 'ineq', 'fun': lambda x:  x[1] - 2.0})

elif type_powerlaw=='Log-Parabola':
    cons = ({'type': 'ineq', 'fun': lambda x:  5.1 - x[2]},
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]},
            {'type': 'ineq', 'fun': lambda x:  x[1] - 2.0})

elif type_powerlaw=='Exponential-Cutoff':
    cons = ({'type': 'ineq', 'fun': lambda x:  5.6 - x[3]},
            {'type': 'ineq', 'fun': lambda x:  x[3] - x[2]},
            {'type': 'ineq', 'fun': lambda x:  x[2] - 1.4})

elif type_powerlaw=='2breaks':
    cons = ({'type': 'ineq', 'fun': lambda x:  5.1 - x[3]},
            {'type': 'ineq', 'fun': lambda x:  x[3] - x[2]},
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]})
            #{'type': 'ineq', 'fun': lambda x:  x[1] - 2.0},#)
            #{'type': 'ineq', 'fun': lambda x:  x[2] - x[1]})

res = minimize(func_interp, x0, method='SLSQP', constraints=cons)
print ('Normal optimization',res.x)

x_res = np.copy(x_ssc)

if type_powerlaw=='Simple':
    #gamma_min = 10**res.x[0]
    gamma_max = 10**res.x[0]
    p = [res.x[1]]
    deltaD = res.x[2]
    B = res.x[3]
    
    #x_res[0] = gamma_min
    x_res[1] = gamma_max
    x_res[5] = [p[0]]
    x_res[12] = deltaD
    x_res[14] = B

elif type_powerlaw=='Broken':
    #gamma_min = 10**res.x[0]
    gamma_max = 10**res.x[0]
    #gamma_br = [10**res.x[1]]
    p = [res.x[1], res.x[2]]
    deltaD = res.x[3]
    B = res.x[4]
    
    #x_res[0] = gamma_min
    x_res[1] = gamma_max
    #x_res[2] = gamma_br[0]
    x_res[5] = [p[0],p[1]]
    x_res[12] = deltaD
    x_res[14] = B

elif type_powerlaw=='Log-Parabola':
    #gamma_min = 10**res.x[0]
    gamma_max = 10**res.x[0]
    p = [res.x[1], res.x[2]]
    deltaD = res.x[3]
    B = res.x[4]
    
    #x_res[0] = gamma_min
    x_res[1] = gamma_max
    x_res[5] = [p[0],p[1]]
    x_res[12] = deltaD
    x_res[14] = B

elif type_powerlaw=='Exponental-Cutoff':
    #gamma_min = 10**res.x[0]
    gamma_max = 10**res.x[0]
    gamma_cut = [10**res.x[1]]
    p = [res.x[2], res.x[3]]
    deltaD = res.x[4]
    B = res.x[5]
    
    #x_res[0] = gamma_min
    x_res[1] = gamma_max
    x_res[3] = gamma_cut
    x_res[5] = [p[0],p[1]]
    x_res[12] = deltaD
    x_res[14] = B

elif type_powerlaw=='2breaks':
    #gamma_min = 10**res.x[0]
    gamma_max = 10**res.x[0]
    #gamma_br = [10**res.x[1], 10**res.x[2]]
    p = [res.x[1],res.x[2],res.x[3]]
    deltaD = res.x[4]
    B = res.x[5]
    
    x_res[1] = gamma_max
    #x_res[2] = [gamma_br[0],gamma_br[1]]
    x_res[5] = [p[0],p[1],p[2]]
    x_res[12] = deltaD
    x_res[14] = B


Y1 = f.SED_synchrotron(x_res, nu, type_powerlaw)
Y2 = f.SED_ssc(x_res, nu, type_powerlaw)
y_res = Y1 + Y2
y_1 = f.SED_synchrotron(x_synch, nu, type_powerlaw)
y_2 = f.SED_ssc(x_ssc, nu, type_powerlaw)
y_old = y_1 + y_2



###################################### Plot ############################################
fig, ax = plt.subplots()

plt.plot(nu, y_res,'-',label='fit',linewidth=2,color='black')
plt.plot(nu, y_old,'.',label='old',linewidth=2,color='grey')
plt.plot(xdata1, ydata1,'o',label='Exp. MrK 421',mec='black',mfc='none')
plt.errorbar(xdata1, ydata1, error_data, fmt='none')

plt.legend(loc='best',numpoints=1)
plt.grid(linestyle = '--',linewidth = 0.5)
plt.xlabel(r'$\nu ~(Hz)$')
plt.ylabel(r'$\nu F_{\nu} ~(erg ~cm^{-2} s^{-1})$')
plt.ylim([1e-15,1e-9])
ax.set_yscale('log')
ax.set_xscale('log')

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

plt.savefig('Fit_2_Mrk421.png')
plt.show()
plt.close()
