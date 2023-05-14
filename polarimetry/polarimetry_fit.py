import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.optimize as optimization
from scipy.optimize import minimize
import functions_fit as f


# data
file1 = np.loadtxt('data_pks2155304.txt', delimiter=' ')
time2 = file1[:,0]
I2     = [file1[:,1], file1[:,2]]
p2     = [file1[:,3], file1[:,4]]
theta2 = [file1[:,5], file1[:,6]]

p2_1 = []
p2_2 = []
p2_3 = []
p2_4 = []
p2_5 = []
p2_6 = []

for i in range(0,len(time2)):
    if 712<=time2[i]<713:
        p2_1.append(p2[0][i])
    elif 713<=time2[i]<714:
        p2_2.append(p2[0][i])
    elif 714<=time2[i]<715:
        p2_3.append(p2[0][i])
    elif 715<=time2[i]<716:
        p2_4.append(p2[0][i])
    elif 716<=time2[i]<717:
        p2_5.append(p2[0][i])
    elif 717<=time2[i]<718:
        p2_6.append(p2[0][i])

P2 = [np.mean(p2_1), np.mean(p2_2), np.mean(p2_3), np.mean(p2_4), np.mean(p2_5), np.mean(p2_6)]
     
theta2_1 = []
theta2_2 = []
theta2_3 = []
theta2_4 = []
theta2_5 = []
theta2_6 = []

for i in range(0,len(time2)):
    if 712<=time2[i]<713:
        theta2_1.append(theta2[0][i])
    elif 713<=time2[i]<714:
        theta2_2.append(theta2[0][i])
    elif 714<=time2[i]<715:
        theta2_3.append(theta2[0][i])
    elif 715<=time2[i]<716:
        theta2_4.append(theta2[0][i])
    elif 716<=time2[i]<717:
        theta2_5.append(theta2[0][i])
    elif 717<=time2[i]<718:
        theta2_6.append(theta2[0][i])


THETA2 = [np.mean(theta2_1), np.mean(theta2_2), np.mean(theta2_3), np.mean(theta2_4), np.mean(theta2_5), np.mean(theta2_6)]


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



# variable parameters
file = np.loadtxt('data_var.txt', delimiter=' ')
#time = file[:,0]-54710+0.5
p_var = [file[:,1], file[:,2]]
theta_var = [file[:,3], file[:,4]]
I_var = [file[:,5], file[:,6]]

param_var = [p_var, theta_var, I_var]

time = [1,2,3,4,5,6]

print(I2_total)
print(I_var)

# contant parameters
p_cons     = 4.0#[4.0, 1.0]
theta_cons = 130#[130, 10]
I_cons = [] #I_cons     = 20

for i in range(0,len(I_var[0])):
    I_cons.append(I2_total[i] - I_var[0][i])

param_cons = [p_cons, theta_cons]

#print(I_cons)

# plot
#p = f.p(param_cons, param_var)
theta = f.theta(param_cons, param_var)


# article results for "p" and "theta"
reference_time = [1.6013986013986015, 2.6013986013986017, 3.6013986013986017, 4.601398601398602, 5.601398601398602, 6.601398601398602]
reference_p = [9.267015706806282, 4.083769633507853, 6.910994764397904, 9.541884816753925, 8.992146596858637, 6.086387434554972]
reference_theta = [83.54978354978354, 93.93939393939394, 104.32900432900433, 116.7965367965368, 118.52813852813853, 128.57142857142856]


##################################### fit - p_var ###################################
#fit = str(input('p or theta? '))
constr = 'y' # str(input('constraints? [y|n] '))


cons = ({'type': 'ineq', 'fun': lambda x:  5.0 - x[0]},
        {'type': 'ineq', 'fun': lambda x:  x[0] - 1.0},
        {'type': 'ineq', 'fun': lambda x:  140 - x[1]},
        {'type': 'ineq', 'fun': lambda x:  x[1] - 110})


x0 = [p_cons, theta_cons]

for i in range(0,len(p_var[0])):
    def func_interp(x1):
        x_model = np.copy(param_cons)
        x_model[0] = x1[0]
        x_model[1] = x1[1]
        #x_model[2] = x1[2]
    
        y_model1 = f.theta(x_model, param_var)[i]
        y_model2 = f.p(x_model, param_var)[i]

        #error1 = []
        #for i in range(0,len(y_model)):
        error1 = (abs((y_model1-THETA2[i])/THETA2[i]))
        error2 = (abs((y_model2-P2[i])/P2[i]))
        #error = np.sum(error1)*100
        #print(error)

        #print('x1[0]',x1[0],'x1[1]',x1[1],'error',error)
    
        return error1 + error2

    if constr == 'N' or constr == 'n':
        res = minimize(func_interp, x0, method='SLSQP')
        print ('Normal optimization',res.x)
        x_res = np.copy(param_cons)
        x_res[0] = res.x[0]
        x_res[1] = res.x[1]
        #x_res[2] = res.x[2]

        y_res = f.theta(x_res, param_var)


    if constr == 'Y' or constr == 'y':
        res = minimize(func_interp, x0, method='SLSQP', constraints = cons)
        print ('Normal optimization',res.x)
        x_res = np.copy(param_cons)
        x_res[0] = res.x[0]
        x_res[1] = res.x[1]
        #x_res[2] = res.x[2]


        y_res = f.theta(x_res, param_var)

