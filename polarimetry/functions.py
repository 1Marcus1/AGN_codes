import numpy as np
import matplotlib.pyplot as plt
import math as m


# contant parameters
p_cons     = [4.0, 1.0]
theta_cons = [130, 10]
I_cons     = [20, 20, 20, 20, 20, 20]

param_cons = [p_cons, theta_cons, I_cons]

#print(I_cons[0])

# variable parameters
p_var = [[12.5, 12.5, 12.5, 12.5, 12.5, 12.5], [1.3, 1.3, 1.3, 1.3, 1.3, 1.3]]
theta_var = [[84.9, 84.9, 84.6, 84.6, 84.6, 84.6], [5.6, 5.6, 5.6, 5.6, 5.6, 5.6]]
I_var = [[2.3, 2.3, 2.3, 2.3, 2.3, 2.3], [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]]
#print(len(theta_var[0]))
param_var  = [p_var, theta_var, I_var]


# functions
def p(param_cons, param_var):
    p_cons     = param_cons[0]
    theta_cons = param_cons[1]
    I_cons     = param_cons[2]
    #print(param_cons)
    p_var      = param_var[0]
    theta_var  = param_var[1]
    I_var      = param_var[2]
    
    I_vc = []
    I_vc_err = []
    for i in range(0,len(I_cons)):
        I_vc.append(I_var[0][i]/I_cons[i])
        I_vc_err.append(I_var[1][i]/I_cons[i])

    epsilon = []
    epsilon_err = []
    for i in range(0,len(theta_var[0])):
        epsilon.append(np.deg2rad((theta_cons[0]-theta_var[0][i])) )
        epsilon_err.append(np.deg2rad(np.sqrt( ( np.power(theta_cons[1], 2) + np.power(theta_var[1][i], 2) ) )))
    
    # calculate
    num = []
    dem = []
    p   = []
    #print((I_cons[0]))
    #print((I_cons[0]))
    #print(len(p_cons[0]))
    for i in range(0,len(I_cons)):
        num.append(np.power(p_cons[0], 2) + ((np.power(p_var[0][i], 2)) * np.power(I_vc[i], 2)) + (2*p_cons[0]*p_var[0][i]*I_vc[i]*np.cos( 2*epsilon[i] ) ))
        dem.append((1+I_vc[i])**2)
    for j in range(0,len(I_var[0])):
        p.append(np.sqrt(num[j]/dem[j]))

    p_err = []
    for i in range(0,len(I_cons)):
        dem_err = 2*np.sqrt( ((p_cons[0]**2)+((p_var[0][i]**2)*(I_vc[i]**2))+(2*p_cons[0]*p_var[0][i]*I_vc[i]*np.cos(2*epsilon[i])) ) / ((I_vc[i])**2) )
    
        a = (( ((I_vc[i]+1)**(-2))*( (2*p_cons[0]) + ( 2*np.cos(2*epsilon[i])*I_vc[i]*p_var[0][i] ) ) )*p_var[1][i]/dem_err)
        b = (( ((I_vc[i]+1)**(-2))*((2*p_var[0][i]*(I_vc[i]**2))+2*np.cos(2*epsilon[i])*I_vc[i]*p_cons[0]) )*p_var[1][i]/dem_err)
        c = ((( (( (2*I_vc[i]*(p_var[0][i]**2))+(2*np.cos(2*epsilon[i])*p_cons[0]*p_var[0][i]) )*((I_vc[i]+1)**2))-(2*( (p_cons[0]**2)+((p_var[0][i]**2)*(I_vc[i]**2))+(2*p_cons[0]*p_var[0][i]*I_vc[i]*np.cos(2*epsilon[i])) )*(I_vc[i]+1)) )/dem_err)*I_vc[i]/( (I_vc[i]+1)**4 ))
        d = (( ((I_vc[i]+1)**(-2))*4*p_cons[0]*p_var[0][i]*I_vc[i]*np.sin(2*epsilon[i]) )*epsilon_err[i]/dem_err)
    
    #for j in range(0,len(I_cons)):
        p_err.append(np.sqrt( (a**2)+(b**2)+(c**2)+(d**2) ))
    #print(p_err[0])
    p1 = []
    p_err1 = []
    for i in range(0,len(p_err)):
        p1.append(p[i])
        p_err1.append(p_err[i])
    #for k in range(len(p_err))
    #print(p1)
    
    return p1, p_err1


def theta(param_cons, param_var):
    p_cons     = param_cons[0]
    theta_cons = param_cons[1]
    I_cons     = [25.438470588235294, 23.755814814814816, 23.104719999999997, 24.741312499999996, 21.197599999999998, 23.688428571428574] #param_cons[2]
    
    p_var      = param_var[0]
    theta_var  = param_var[1]
    I_var      = param_var[2]
    
    
    #I_vc = [I_var[0]/I_cons, I_var[1]/I_cons]
    I_vc = []
    I_vc_err = []
    for i in range(0,len(I_var[0])):
        I_vc.append(I_var[0][i]/I_cons[i])
        I_vc_err.append(I_var[1][i]/I_cons[i])

    frac = []
    for i in range(0,len(I_var[0])):
        a = (p_cons[0] * np.sin(np.deg2rad(2*theta_cons[0]))) + (p_var[0][i] * I_vc[i] * np.sin(np.deg2rad(2*theta_var[0][i])))
        b = (p_cons[0] * np.cos(np.deg2rad(2*theta_cons[0]))) + (p_var[0][i] * I_vc[i] * np.cos(np.deg2rad(2*theta_var[0][i])))
    
        frac.append(a/b)# = a/b
        
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    if isinstance(theta_var[0], (list, np.ndarray)) == True:
        theta = []
        for i in range(0, len(frac)):
            theta.append( ( np.rad2deg( m.atan(frac[i]) ) / 2.0 ) + 90)

        for k in range(0,len(theta_var[0])):
            
            if 0<abs(theta_cons[0]-theta_var[0][k])<=65 or 155<abs(theta_cons[0]-theta_var[0][k])<=180:
                for i in range(0,len(theta)):
                    if abs((theta[i]-theta[i-1])/theta[i-1]) >= 0.6:
                        t1 = i
                for j in range(t1, len(theta)):
                    theta[j] -= 0
                
            elif 65<abs(theta_cons[0]-theta_var[0][k])<=90:
                for i in range(0,len(theta)):
                    if abs((theta[i]-theta[i-1])/theta[i-1]) >= 0.6:
                        t2 = i
                for j in range(t2, len(theta)):
                    theta[j] -= 90
        
            elif 90<abs(theta_cons[0]-theta_var[0][k])<=155:
                for i in range(0,len(theta)):
                    if abs((theta[i]-theta[i-1])/theta[i-1]) >= 0.6:
                        t3 = i
                for j in range(t3, len(theta)):
                    theta[j] += 90
    
    else:
        theta = []
        for i in range(0, len(frac)):
            theta.append( ( np.rad2deg( m.atan(frac[i]) ) / 2.0 ) + 0)
        
        if 0<(theta_cons[0]-theta_var[0])<=65 or 155<(theta_cons[0]-theta_var[0])<=180:
            for i in range(0,len(theta)):
                if abs((theta[i]-theta[i-1])/theta[i-1]) >= 0.6:
                    t1 = i
            for k in range(t1, len(theta)):
                theta[k] -= 0
                
        elif 65<(theta_cons[0]-theta_var[0])<=90:
            for i in range(0,len(theta)):
                if abs((theta[i]-theta[i-1])/theta[i-1]) >= 0.6:
                    t2 = i
            for k in range(t2, len(theta)):
                theta[k] -= 90
        
        elif 90<(theta_cons[0]-theta_var[0])<=155:
            for i in range(0,len(theta)):
                if abs((theta[i]-theta[i-1])/theta[i-1]) >= 0.6:
                    t3 = i
            for k in range(t3, len(theta)):
                theta[k] += 90
    
    
    #uncertainty
    epsilon_1 = ( ( b*np.sin(2*np.deg2rad(theta_cons[0])) ) - ( a*np.cos(2*np.deg2rad(theta_cons[0])) ) )*p_cons[1] / ( (a**2)+(b**2) )
    epsilon_2 = ( (a+b)/((a**2)+(b**2)) ) * ( p_cons[0]*np.cos(4*np.deg2rad(theta_cons[0])) )*theta_cons[1]
    epsilon_3 = ( I_vc/((a**2)+(b**2)) ) * ( (b*np.sin(2*np.deg2rad(theta_var[0])))-(a*np.cos(2*np.deg2rad(theta_var[0]))) )*p_var[1]
    epsilon_4 = ( (a+b)/((a**2)+(b**2)) ) * ( p_var[0]*I_vc*np.sin(4*np.deg2rad(theta_var[0])) )*theta_var[1]
    epsilon_5 = ( p_var[0]/((a**2)+(b**2)) ) * ( (b*np.sin(2*np.deg2rad(theta_var[0])))-(a*np.cos(2*np.deg2rad(theta_var[0]))) )*I_vc[1]
    
    theta_err = np.sqrt( (epsilon_1**2)+(epsilon_2**2)+(epsilon_3**2)+(epsilon_4**2)+(epsilon_5**2) ) / 2.0
    
    return theta, theta_err
