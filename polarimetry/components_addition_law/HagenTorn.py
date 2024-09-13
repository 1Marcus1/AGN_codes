import numpy as np
from scipy.odr import ODR, Model, RealData
import functions_addition_law as add
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class HagenTorn:
    def __init__(self, Q, U, I, Q_err, U_err, I_err, is_SP=True):
        
        if is_SP == True:
            self.Q = np.array(Q) ; self.U = np.array(U) ; self.I = np.array(I)
            self.Q_err = np.array(Q_err) ; self.U_err = np.array(U_err) ; self.I_err = np.array(I_err)
            self.p, self.p_err = self.polarization_degree(self.Q, self.U, self.I, self.Q_err, self.U_err, self.I_err) ; self.theta, self.theta_err = self.polarization_angle(self.Q, self.U, self.Q_err, self.U_err)
        

        elif is_SP == False:
            Q_obs     = []
            Q_obs_err = []
            U_obs     = []
            U_obs_err = []
            for i in range(0,len(Q)):
                Q_obs.append( (Q[i] * I[i]) * np.cos( np.deg2rad(2*U[i]) )/100 )
                Q_obs_err.append( np.sqrt( ((Q_err[i] * I[i]) * np.cos( np.deg2rad(2*U[i]) )/100)**2 + ((Q[i] * I_err[i]) * np.cos( np.deg2rad(2*U[i]) )/100)**2 + ((2*U_err[i]*Q[i] * I[i]) * np.sin( np.deg2rad(2*U[i]) )/100)**2 ) )
                U_obs.append( (Q[i] * I[i]) * np.sin( np.deg2rad(2*U[i]) )/100 )
                U_obs_err.append( np.sqrt( ((Q_err[i] * I[i]) * np.sin( np.deg2rad(2*U[i]) )/100)**2 + ((Q[i] * I_err[i]) * np.sin( np.deg2rad(2*U[i]) )/100)**2 + ((2*U_err[i]*Q[i] * I[i]) * np.cos( np.deg2rad(2*U[i]) )/100)**2 ) )
            
            self.Q = np.array(Q_obs) ; self.U = np.array(U_obs) ; self.I = np.array(I)
            self.Q_err = np.array(Q_obs_err) ; self.U_err = np.array(U_obs_err) ; self.I_err = np.array(I_err)
            self.p, self.p_err = np.array(Q), np.array(Q_err) ; self.theta, self.theta_err = np.array(U), np.array(U_err)
        

        self.p_var, self.theta_var, self.I1_var, self.p_var_err, self.theta_var_err, self.I1_var_err = self.variable(self.Q, self.U, self.I, self.Q_err, self.U_err, self.I_err)
        self.I_var = np.average(self.I1_var, weights=self.I1_var_err)
        self.I_var_err = self.I1_var_err.mean()
        
        self.I_cons = np.average(self.I, weights=self.I_err) - np.average(self.I1_var, weights=self.I1_var_err) ; self.I_cons_err = self.I_var_err

    
    def polarization_degree(self, Q, U, I, Q_err, U_err, I_err):
        p = np.sqrt((Q**2 + U**2)/(I**2))
        perr = (1/p) * np.sqrt( (Q*Q_err)**2 + (U*U_err)**2 )
        return p*100, perr*100
    
    def polarization_angle(self, Q, U, Q_err, U_err):
        theta = np.rad2deg(np.arctan(U/Q))# + 90.0
        theta_err = np.rad2deg( 0.5/( 1 + (U/Q)**2) * np.sqrt( (U_err/Q)**2 + (U*Q_err/(Q**2))**2 ) )
        return theta, theta_err
    
    def standartize(self, a):
        return (a - np.mean(a))/np.std(a)
    
    def line(self, B, x):
        return B[0]*x + B[1]
    '''
    def variable(self, Q, U, I, Q_err, U_err, I_err):
        qi = RealData(I, Q, sx=I_err, sy=Q_err)
        ui = RealData(I, U, sx=I_err, sy=U_err)
        model = Model(self.line)
        #cov_qi = np.cov(I, Q, bias=True)[0][1]
        #cov_ui = np.cov(I, U, bias=True)[0][1]
        #var = np.var(I, ddof=0)
        #slope_qi = cov_qi/var ; slope_ui = cov_ui/var
        #intercept_qi = Q[0]-(I[0]*slope_qi) ; intercept_ui = U[0]-(I[0]*slope_ui)
        odr_qi = ODR(qi, model, beta0=[0.1, np.mean(Q)]).run()
        odr_ui = ODR(ui, model, beta0=[0.1, np.mean(U)]).run()
        
        # p_var
        p_var_x = odr_qi.beta[0] ; p_var_x_err = odr_qi.sd_beta[0]
        p_var_y = odr_ui.beta[0] ; p_var_y_err = odr_ui.sd_beta[0]
        print(p_var_x_err, p_var_y_err)

        p_var = np.sqrt( p_var_x**2 + p_var_y**2 )
        theta_var = np.rad2deg( np.arctan( p_var_y/p_var_x )/2.0 ) + 90
        I_var = I*p_var

        p_var_err = (1/p_var) * np.sqrt( (p_var_x*p_var_x_err)**2 + (p_var_y*p_var_y_err)**2 )
        theta_var_err = np.rad2deg( 0.5/( 1 + (p_var_y/p_var_x)**2) * np.sqrt( (p_var_y_err/p_var_x)**2 + (p_var_y*p_var_x_err/(p_var_x**2))**2 ) )
        I_var_err = I*p_var_err
        
        return p_var*100, theta_var, I_var, p_var_err*100, theta_var_err, I_var_err
    
    def interp_qi(self, x1):
        x_model = [x1[0], x1[1]]
        y_model1 = self.line(x1, self.I)
        error1 = np.sum(((y_model1 - self.Q)**2) * (1 / self.Q_err**2) )
        return error1
    
    def interp_ui(self, x1):
        x_model = [x1[0], x1[1]]
        y_model1 = self.line(x1, self.I)
        error1 = np.sum(((y_model1 - self.U)**2) * (1 / self.U_err**2) )
        return error1
    
    def variable(self, Q, U, I, Q_err, U_err, I_err):
        p_var_x = [] ; p_var_y = []
        for i in range(0,10):
            x0_x = [np.random.normal( (Q[-1]-Q[0])/(I[-1]-I[0]) , 0.01), np.random.normal( Q[0] - ((Q[-1]-Q[0])*I[0]/(I[-1]-I[0])) ,0.01)]
            res_x = minimize(self.interp_qi, x0_x, method='SLSQP')
            p_var_x.append(res_x.x[0])
            
            x0_y = [np.random.normal( (U[-1]-U[0])/(I[-1]-I[0]) , 0.01), np.random.normal( U[0] - ((U[-1]-U[0])*I[0]/(I[-1]-I[0])) ,0.01)]
            res_y = minimize(self.interp_ui, x0_y, method='SLSQP')
            p_var_y.append(res_y.x[0])
            
        slope_qi = np.mean(p_var_x) ; slope_ui = np.mean(p_var_y)
        slope_qi_err = np.std(p_var_x) ; slope_ui_err = np.std(p_var_y)
        
        p_var = np.sqrt(slope_qi**2 + slope_ui**2)
        theta_var = np.rad2deg(np.arctan(slope_ui / slope_qi) / 2.0)# + 90
        I_var = I * p_var

        p_var_err = (1/p_var) * np.sqrt((slope_qi * slope_qi_err)**2 + (slope_ui * slope_ui_err)**2)
        theta_var_err = np.rad2deg(0.5 / (1 + (slope_ui / slope_qi)**2) * np.sqrt((slope_ui_err / slope_qi)**2 + (slope_ui * slope_qi_err / (slope_qi**2))**2))
        I_var_err = I * p_var_err

        return p_var*100, theta_var, I_var, p_var_err*100, theta_var_err, I_var_err
    '''

    def variable(self, Q, U, I, Q_err, U_err, I_err):
        (slope_qi, _), cov_matrix_qi = np.polyfit(I, Q, 1, cov=True, w=1/Q_err)
        (slope_ui, _), cov_matrix_ui = np.polyfit(I, U, 1, cov=True, w=1/U_err)

        slope_qi_err = np.sqrt(cov_matrix_qi[0][0])
        slope_ui_err = np.sqrt(cov_matrix_ui[0][0])
        
        self.slope_qi = slope_qi ; self.slope_ui = slope_ui

        p_var = np.sqrt(slope_qi**2 + slope_ui**2)
        theta_var = np.rad2deg(np.arctan(slope_ui / slope_qi) / 2.0)# + 90
        I_var = I * p_var

        p_var_err = (1/p_var) * np.sqrt((slope_qi * slope_qi_err)**2 + (slope_ui * slope_ui_err)**2)
        theta_var_err = np.rad2deg(0.5 / (1 + (slope_ui / slope_qi)**2) * np.sqrt((slope_ui_err / slope_qi)**2 + (slope_ui * slope_qi_err / (slope_qi**2))**2))
        I_var_err = I * p_var_err
        '''
        fig, ax = plt.subplots(2, sharex=True)
        fig.subplots_adjust(hspace=0)
        ax[1].plot(I, Q, 'o', color='blue')
        ax[1].plot(I, self.line([ slope_qi, (np.mean(Q)-slope_qi*np.mean(I)) ], I), '-', color='black' )
        ax[1].set_ylabel('Q') ; plt.xlabel('I')
        #plt.show()

        #fig, ax = plt.subplots()
        ax[0].plot(I, U, 'o', color='blue')
        ax[0].plot(I, self.line([ slope_ui, (np.mean(U)-slope_ui*np.mean(I)) ], I), '-', color='black' )
        ax[0].set_ylabel('U')# ; plt.xlabel('I')
        plt.show()
        '''

        return p_var*100, theta_var, I_var, p_var_err*100, theta_var_err, I_var_err
    


        
