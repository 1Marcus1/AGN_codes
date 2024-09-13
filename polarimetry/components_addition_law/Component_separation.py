import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy import interpolate
import scipy.optimize as optimization
from scipy.optimize import minimize
from scipy import stats

from HagenTorn import HagenTorn as HT
from Addition_Law import Addition_Law as add


home = os.getcwd()

# data
file2 = pd.read_csv('data_pks2155304_complete.txt', sep='  ')
file1 = file2#file2[file2['filter'] == 'R']
time = file1['#MJD'].to_numpy()
I2 = file1['I'].to_numpy()
I2_err = file1['Ierr'].to_numpy()
p2 = file1['p'].to_numpy()
p2_err = file1['p_err'].to_numpy()
theta2 = file1['theta'].to_numpy()
theta2_err = file1['theta_err'].to_numpy()

day1_pandas = file1[file1['#MJD'] <= 713].to_numpy().transpose()
day2_pandas = file1[(file1['#MJD'] >= 713) & (file1['#MJD'] <= 714)].to_numpy().transpose()
day3_pandas = file1[(file1['#MJD'] >= 714) & (file1['#MJD'] <= 715)].to_numpy().transpose()
day4_pandas = file1[(file1['#MJD'] >= 715) & (file1['#MJD'] <= 716)].to_numpy().transpose()
day5_pandas = file1[(file1['#MJD'] >= 716) & (file1['#MJD'] <= 717)].to_numpy().transpose()
day6_pandas = file1[file1['#MJD'] >= 717].to_numpy().transpose()

# Store the arrays in a list
days = [day1_pandas, day2_pandas, day3_pandas, day4_pandas, day5_pandas, day6_pandas]

p_mean = [np.average(day[4], weights=day[5]) for day in days]
chi_mean = [np.average(day[6], weights=day[7]) for day in days]
time_mean = [np.average(day[0]) for day in days]
I_mean = [np.average(day[2], weights=day[3]) for day in days]


# Calculate the Variable Component
theta_var = []
theta_var_err = []
p_var = []
p_var_err = []
I_var = []
I_var_err = []
I_cons = []
I_cons_err = []


for j in range(len(days)):
    time = np.array(days[j][0], dtype=float)
    pd   = np.array(days[j][4], dtype=float)
    pa   = np.array(days[j][6], dtype=float)
    I    = np.array(days[j][2], dtype=float)
    pd_err = np.array(days[j][5], dtype=float)
    pa_err = np.array(days[j][7], dtype=float)
    I_err  = np.array(days[j][3], dtype=float)
    
    I = pd * I / 100
    
    mask = np.isfinite(pd) & np.isfinite(pa) & np.isfinite(I) & np.isfinite(pd_err) & np.isfinite(pa_err) & np.isfinite(I_err)
    pd, pa, I, pd_err, pa_err, I_err = pd[mask], pa[mask], I[mask], pd_err[mask], pa_err[mask], I_err[mask]

    aaa = HT(pd, pa, I, pd_err, pa_err, I_err, is_SP=False)
    
    p_var.append(aaa.p_var)
    p_var_err.append(aaa.p_var_err)
    theta_var.append(aaa.theta_var + 90)
    theta_var_err.append(aaa.theta_var_err)
    I_var.append(aaa.I_var)
    I_var_err.append(aaa.I_var_err)
    I_cons.append(aaa.I_cons)
    I_cons_err.append(aaa.I_cons_err)
    

p_var = np.array(p_var)
p_var_err = np.array(p_var_err)
theta_var = np.array(theta_var)
theta_var_err = np.array(theta_var_err)
I_var = np.array(I_var)
I_var_err = np.array(I_var_err)
I_cons = np.array(I_cons)
I_cons_err = np.array(I_cons_err)


# Fit Constant Component
cons = ({'type': 'ineq', 'fun': lambda x:  8. - x[0]},
        {'type': 'ineq', 'fun': lambda x:  x[0] - 2.4},
        {'type': 'ineq', 'fun': lambda x:  145. - x[1]},
        {'type': 'ineq', 'fun': lambda x:  x[1] - 92.})


p_cons_new1     = [] ; p_cons_new2 = [] ; p_cons_new2_err = []
theta_cons_new1 = [] ; theta_cons_new2 = [] ; theta_cons_new2_err = []

for i in range(0,len(days)): # in each day
    for j in range(0,500): # getting errors
        # Initial guess
        param_cons = [np.random.normal(4.0, 1.0), np.random.normal(130,1.0)]

        x0 = [param_cons[0], param_cons[1]]
        param_var = [p_var[i], theta_var[i], I_var[i]]


        p_err = np.array(p_var_err[i])
        theta_err = np.array(theta_var_err[i])
        I_err = np.array(I_var_err[i])

        weights_p = 100**2 / p_err**2
        weights_theta = 1 / theta_err**2

        def func_interp(x1):
            x_model = np.copy(param_cons)
            x_model[0] = x1[0]
            x_model[1] = x1[1]
            
            b = add.Addition_law([x_model[0], x_model[1], I_cons[i]], param_var, is_SP=False)
            
            y_model1 = b.theta_tot
            y_model2 = b.p_tot

            error1 = np.sum(((y_model1 - days[i][6])**2) * weights_theta)
            error2 = np.sum(((y_model2 - days[i][4])**2) * weights_p)
            
            return error1 + error2

        res = minimize(func_interp, x0, method='SLSQP', constraints = cons)
        p_cons_new1.append(res.x[0]) ; theta_cons_new1.append(res.x[1])
    
    p_cons_new2.append(np.average(p_cons_new1)) ; p_cons_new2_err.append(np.std(p_cons_new1))
    theta_cons_new2.append(np.average(theta_cons_new1)) ; theta_cons_new2_err.append(np.std(theta_cons_new1))
    p_cons_new1.clear() ; theta_cons_new1.clear()


# constant component result
p_cons_new = np.array(p_cons_new2) ; p_cons_new_err = np.array(p_cons_new2_err)
theta_cons_new = np.array(theta_cons_new2) ; theta_cons_new_err = np.array(theta_cons_new2_err)

#p_cons_new = np.array([np.mean(p_cons_new2)]*len(days)) ; p_cons_new_err = np.array([np.std(p_cons_new2)]*len(days))
#theta_cons_new = np.array([np.mean(theta_cons_new2)]*len(days)) ; theta_cons_new_err = np.array([np.std(theta_cons_new2)]*len(days))

#print(np.mean(p_cons_new), np.std(p_cons_new), np.mean(theta_cons_new), np.std(theta_cons_new))
#print(np.mean(p_cons_new)+np.std(p_cons_new) , np.mean(p_cons_new)-np.std(p_cons_new), np.mean(theta_cons_new)+np.std(theta_cons_new), np.mean(theta_cons_new)-np.std(theta_cons_new))


# spline interpolation of variable component
def p_spline_var(x):
    x_points = time_mean
    y_points = p_var

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def theta_spline_var(x):
    x_points = time_mean
    y_points = theta_var

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def p_spline_var_min(x):
    x_points = time_mean
    y_points = p_var-p_var_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def theta_spline_var_min(x):
    x_points = time_mean
    y_points = theta_var-theta_var_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def p_spline_var_max(x):
    x_points = time_mean
    y_points = p_var+p_var_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def theta_spline_var_max(x):
    x_points = time_mean
    y_points = theta_var+theta_var_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

# spline interpolation of constant component
def p_spline_cons(x):
    x_points = time_mean
    y_points = p_cons_new

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def theta_spline_cons(x):
    x_points = time_mean
    y_points = theta_cons_new

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def p_spline_cons_min(x):
    x_points = time_mean
    y_points = p_cons_new-p_cons_new_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def theta_spline_cons_min(x):
    x_points = time_mean
    y_points = theta_cons_new-theta_cons_new_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def p_spline_cons_max(x):
    x_points = time_mean
    y_points = p_cons_new+p_cons_new_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def theta_spline_cons_max(x):
    x_points = time_mean
    y_points = theta_cons_new+theta_cons_new_err

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)



c = add.Addition_law([p_cons_new, theta_cons_new, I_cons], param_var, is_SP=False)


# results
p_result = c.p_tot
p_result_err = c.p_total_err(p_cons_new, p_var, theta_cons_new, theta_var, I_cons, I_var, p_cons_new_err, p_var_err, theta_cons_new_err, theta_var_err, I_cons_err, I_var_err)
theta_result = c.theta_tot
theta_result_err = c.theta_total_err(p_cons_new/100, p_var/100, theta_cons_new, theta_var, I_cons, I_var, p_cons_new_err/100, p_var_err/100, theta_cons_new_err, theta_var_err, I_cons_err, I_var_err)


print('p_cons: ', p_cons_new, p_cons_new_err)
print('p_var: ', p_var, p_var_err)

print('\ntheta_cons: ', theta_cons_new, theta_cons_new_err)
print('theta_var: ', theta_var, theta_var_err)

print('\nI_cons: ', I_cons, I_cons_err)
print('I_var: ', I_var, I_var_err)

print('\np_total: ', p_result, p_result_err)
print('theta_total: ', theta_result, theta_result_err)

# Plot results of separation components
time = file1['#MJD'].to_numpy().tolist()
p   = p2
chi = theta2
I   = I2

p_cons_new = np.array([np.mean(p_cons_new2)]*len(days)) ; p_cons_new_err = np.array([np.std(p_cons_new2)]*len(days))
theta_cons_new = np.array([np.mean(theta_cons_new2)]*len(days)) ; theta_cons_new_err = np.array([np.std(theta_cons_new2)]*len(days))

print(p_cons_new, p_cons_new_err)
print(theta_cons_new, theta_cons_new_err)

x_spline = np.linspace(time[0], time[-1])


fig, axs = plt.subplots(2, sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].errorbar(x=time, y=p, yerr=p2_err, fmt='o', label='data', linewidth=2, color='blue', markersize=3, capsize=3)                                    # data

axs[0].errorbar(x = time_mean, y = p_result, yerr = p_result_err, fmt='o', color='black' , capsize=5, label='result')                                   # result

axs[0].plot(x_spline, p_spline_cons(x_spline), '--', label='constant component', linewidth=1.2, color='black')                                          # constant
axs[0].fill_between(x_spline, p_spline_cons_min(x_spline), p_spline_cons_max(x_spline), alpha=0.2)#, label='error constant')                              # constant error shade

axs[0].plot(x_spline, p_spline_var(np.linspace(time[0], time[-1])), '-', label='variable component', linewidth=1, color='black')                            # variable
axs[0].fill_between(x_spline, p_spline_var_min(x_spline), p_spline_var_max(x_spline), alpha=0.2)#, label='error var')                                     # variable error shade

axs[0].set_ylabel(r'polarization ($\%$)')
axs[0].grid(linestyle = '--',linewidth = 0.5)
axs[0].legend(loc='best', ncols=2)
axs[0].set_title('Components separation - PKS 2155-304')


axs[1].errorbar(x=time, y=chi, yerr=theta2_err, fmt='o', label='data', linewidth=2, color='blue', markersize=3, capsize=3)                              # data

axs[1].errorbar(x = time_mean, y = theta_result, yerr = theta_result_err, fmt='o', color='black' , capsize=5, label='result')                           # result

axs[1].plot(x_spline, theta_spline_cons(x_spline), '--', label='constant component', linewidth=1.2, color='black')                                      # constant
axs[1].fill_between(x_spline, theta_spline_cons_min(x_spline), theta_spline_cons_max(x_spline), alpha=0.2)#, label='error constant')                    # constant error shade

axs[1].plot(x_spline, theta_spline_var(x_spline), '-', label='variable component', linewidth=1, color='black')                                          # variable
axs[1].fill_between(x_spline, theta_spline_var_min(x_spline), theta_spline_var_max(x_spline), alpha=0.2)#, label='error var')                           # variable error shade

axs[1].set_ylabel(r'position angle (degree)')
axs[1].grid(linestyle = '--',linewidth = 0.5)


plt.xlabel(r'time (days)')

#plt.savefig('model_fit_theta_cons.png')
plt.show()
