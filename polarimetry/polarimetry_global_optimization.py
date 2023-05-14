import numpy as np
import matplotlib.pyplot as plt
import functions as f


# variable parameters
file = np.loadtxt('data_var.txt', delimiter=' ')
time = file[:,0]-54710
p_var = [file[:,1], file[:,2]]
theta_var = [file[:,3], file[:,4]]
I_var = [file[:,5], file[:,6]]

param_var = [p_var, theta_var, I_var]


# data
file1 = np.loadtxt('data_pks2155304.txt', delimiter=' ')
time2 = file1[:,0]
I2     = [file1[:,1], file1[:,2]]
p2     = [file1[:,3], file1[:,4]]
theta2 = [file1[:,5], file1[:,6]]


I2_1 = []
I2_2 = []
I2_3 = []
I2_4 = []
I2_5 = []
I2_6 = []

for i in range(0,len(time2)):
    if 712<=time2[i]<713:
        I2_1.append(I2[0][i])
    elif 713<=time2[i]<714:
        I2_2.append(I2[0][i])
    elif 714<=time2[i]<715:
        I2_3.append(I2[0][i])
    elif 715<=time2[i]<716:
        I2_4.append(I2[0][i])
    elif 716<=time2[i]<717:
        I2_5.append(I2[0][i])
    elif 717<=time2[i]<718:
        I2_6.append(I2[0][i])


I2_total = [np.mean(I2_1), np.mean(I2_2), np.mean(I2_3), np.mean(I2_4), np.mean(I2_5), np.mean(I2_6)]



# contant parameters
I_cons = []
for i in range(0,len(I_var[0])):
    I_cons.append(I2_total[i]-I_var[0][i])


############# function variable component #############
# p

matrix     = [[], [], [], [], [], []]
matrix_sol = []
for i in range(0,6):
    for j in range(0,6):
        matrix[i].append(time[i]**j)
    matrix_sol.append(p_var[0][i])

param_pol = np.linalg.solve(matrix, matrix_sol)[::-1]

# theta
matrix_     = [[], [], [], [], [], []]
matrix_sol_ = []
for i in range(0,6):
    for j in range(0,6):
        matrix_[i].append(time[i]**j)
    matrix_sol_.append(theta_var[0][i])

param_pol_ = np.linalg.solve(matrix_, matrix_sol_)[::-1]

def func(param_pol, x):
    return param_pol[0]*(x**5) + param_pol[1]*(x**4) + param_pol[2]*(x**3) + param_pol[3]*(x**2) + param_pol[4]*x + param_pol[5]

x = np.linspace(1, 6,200)

function_p     = func(param_pol, x)
function_theta = func(param_pol_, x)
######################################################


# article results for "p" and "theta"
reference_time = [1.6013986013986015, 2.6013986013986017, 3.6013986013986017, 4.601398601398602, 5.601398601398602, 6.601398601398602]
reference_p = [9.267015706806282, 4.083769633507853, 6.910994764397904, 9.541884816753925, 8.992146596858637, 6.086387434554972]
reference_theta = [83.54978354978354, 93.93939393939394, 104.32900432900433, 116.7965367965368, 118.52813852813853, 128.57142857142856]

yerr_reference_p = [10.183706070287538-9.273162939297123, 4.8881789137380185-4.0734824281150175, 7.595846645367411-6.924920127795527, 10.495207667731629-9.536741214057507, 9.87220447284345-8.985623003194888, 6.685303514376997-6.086261980830672]
yerr_reference_theta = [87.81774580335733-83.5971223021583, 93.95683453237412-89.35251798561154, 104.50839328537171-99.13669064748203, 122.92565947242208-116.97841726618707, 118.70503597122304-112.75779376498802, 128.8729016786571-122.15827338129498]


#################### plot ####################
# optimization (with constraints)
p_cons1 = [5, 1.0]
p_cons2 = [2.56030056, 1.0]
p_cons3 = [5, 1.0]
p_cons4 = [5, 1.0]
p_cons5 = [5, 1.0]
p_cons6 = [5, 1.0]

theta_cons1 = [110, 10.0]
theta_cons2 = [110, 10.0]
theta_cons3 = [110.80238682, 10.0]
theta_cons4 = [114.23178426, 10.0]
theta_cons5 = [114.89010382, 10.0]
theta_cons6 = [132.95949186, 10.0]



# p
a1 = f.p( [p_cons1, theta_cons1, I_cons], param_var )
b1 = f.p( [p_cons2, theta_cons2, I_cons], param_var )
c1 = f.p( [p_cons3, theta_cons3, I_cons], param_var )
d1 = f.p( [p_cons4, theta_cons4, I_cons], param_var )
e1 = f.p( [p_cons5, theta_cons5, I_cons], param_var )
g1 = f.p( [p_cons6, theta_cons6, I_cons], param_var )
# theta
a2 = f.theta( [p_cons1, theta_cons1, I_cons], param_var )
b2 = f.theta( [p_cons2, theta_cons2, I_cons], param_var )
c2 = f.theta( [p_cons3, theta_cons3, I_cons], param_var )
d2 = f.theta( [p_cons4, theta_cons4, I_cons], param_var )
e2 = f.theta( [p_cons5, theta_cons5, I_cons], param_var )
g2 = f.theta( [p_cons6, theta_cons6, I_cons], param_var )

p_result     = np.mean([p_cons1[0], p_cons2[0], p_cons3[0], p_cons4[0], p_cons5[0], p_cons6[0]])
theta_result = np.mean([theta_cons1[0], theta_cons2[0], theta_cons3[0], theta_cons4[0], theta_cons5[0], theta_cons6[0]])

y1    = [a1[0][0], b1[0][1], c1[0][2], d1[0][3], e1[0][4], g1[0][5]]
yerr1 = np.std([p_cons1[0], p_cons2[0], p_cons3[0], p_cons4[0], p_cons5[0], p_cons6[0]])

y2    = [a2[0][0], b2[0][1], c2[0][2], d2[0][3], e2[0][4], g2[0][5]]
yerr2 = np.std([theta_cons1[0], theta_cons2[0], theta_cons3[0], theta_cons4[0], theta_cons5[0], theta_cons6[0]])

time2 = time2+54000-1.5-54710

print('p: {} +- {}'.format(p_result, yerr1))
print('theta: {} +- {}'.format(theta_result, yerr2))



fig, axs = plt.subplots(2, sharex=True)
fig.subplots_adjust(hspace=0)


axs[0].plot(time2, p2[0], 'o', label='data', linewidth=2, color='blue', markersize=3)

axs[0].plot(time, y1, 'o', label='result', linewidth=2, color='black')
axs[0].errorbar(x = time, y = y1, yerr = yerr1, fmt='o', color='black' , capsize=5)

axs[0].plot([time2[0],time2[-1]], [np.mean(y1),np.mean(y1)], '--', label='constant component', linewidth=0.8, color='black')
axs[0].plot(x, function_p, '-', label='variable component', linewidth=1, color='black')

axs[0].plot(time, reference_p, 'o', label='reference', linewidth=2, color='red')
axs[0].errorbar(x = time, y = reference_p, yerr = yerr_reference_p, fmt='o', color='red', capsize=5)


axs[0].set_ylabel(r'polarization ($\%$)')
#axs[0].set_ylim([0,15])
axs[0].grid(linestyle = '--',linewidth = 0.5)


axs[1].plot(time2, theta2[0], 'o', label='data', linewidth=2, color='blue', markersize=3)

axs[1].plot(time, y2, 'o', label='result', linewidth=2, color='black',markersize=5)
axs[1].errorbar(x = time, y = y2, yerr = yerr2, fmt='o', color='black' , capsize=5)

axs[1].plot([time2[0],time2[-1]], [np.mean(y2),np.mean(y2)], '--', label='constant component', linewidth=0.8, color='black')
axs[1].plot(x, function_theta, '-', label='variable component', linewidth=1, color='black')

axs[1].plot(time, reference_theta, 'o', label='reference', linewidth=2, color='red')
axs[1].errorbar(x = time, y = reference_theta, yerr = yerr_reference_theta, fmt='o', color='red', capsize=5)


axs[1].set_ylabel(r'position angle (degree)')
#axs[1].set_ylim([70,170])
axs[1].grid(linestyle = '--',linewidth = 0.5)

plt.legend(loc='lower right', ncols=2, fontsize='x-small')
plt.xlabel(r'time (days)')

plt.savefig('model_global_fit_cons.png')
plt.show()
plt.close()
