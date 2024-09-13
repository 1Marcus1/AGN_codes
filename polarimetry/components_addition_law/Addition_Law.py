import numpy as np
import os

class Additionlaw:
    def __init__(self, cons, var, is_SP=False, norm=False):
        
        self.cons = cons
        self.p_cons = self.cons[0] ; self.theta_cons = self.cons[1] ; self.I_cons = self.cons[2]
        self.var = var
        self.p_var  = self.var[0]  ; self.theta_var  = self.var[1]  ; self.I_var  = self.var[2]
        
        #self.cons_err = cons_err
        #self.p_cons_err = self.cons_err[0] ; self.theta_cons_err = self.cons_err[1] ; self.I_cons_err = self.cons_err[2]
        #self.var_err = var_err
        #self.p_var_err  = self.var_err[0]  ; self.theta_var_err  = self.var_err[1]  ; self.I_var_err  = self.var_err[2]
        
        
        if is_SP == False:
            self.p_cons     = np.array(self.p_cons)     ; self.p_var     = np.array(self.p_var)
            self.theta_cons = np.array(self.theta_cons) ; self.theta_var = np.array(self.theta_var)
            self.I_cons     = np.array(self.I_cons)     ; self.I_var     = np.array(self.I_var)
        
        elif is_SP == True:
            p_obs_cons     = [] ; p_obs_var     = []
            theta_obs_cons = [] ; theta_obs_var = []
            for i in range(0,len(self.I_cons)):
                p_obs_cons.append( np.sqrt(self.p_cons**2 + self.theta_cons**2) * 100 / self.I_cons ) ; p_obs_var.append( np.sqrt(self.p_var**2 + self.theta_var**2) * 100 / self.I_var )
                theta_obs_cons.append( np.rad2deg(np.arctan(self.theta_cons/self.p_cons))/2.0 )       ; theta_obs_var.append( np.rad2deg(np.arctan(self.theta_var/self.p_var))/2.0 )
            
            self.p_cons     = np.array(p_obs_cons)     ; self.p_var     = np.array(p_obs_var)
            self.theta_cons = np.array(theta_obs_cons) ; self.theta_var = np.array(theta_obs_var)
            self.I_cons     = np.array(self.I_cons)    ; self.I_var     = np.array(self.I_var)
        
        if norm == True:
            self.p_cons     = self.standartize(self.p_cons)     ; self.p_var     = self.standartize(self.p_var)
            self.theta_cons = self.standartize(self.theta_cons) ; self.theta_var = self.standartize(self.theta_var)
            self.I_cons     = self.standartize(self.I_cons)     ; self.I_var     = self.standartize(self.I_var)

        self.p_tot     = self.p_total(self.p_cons, self.p_var, self.theta_cons, self.theta_var, self.I_cons, self.I_var)
        self.theta_tot = self.theta_total(self.p_cons, self.p_var, self.theta_cons, self.theta_var, self.I_cons, self.I_var)
        
        #self.p_tot_err     = self.p_total_err(self.p_cons, self.p_var, self.theta_cons, self.theta_var, self.I_cons, self.I_var, self.p_cons_err, self.p_var_err, self.theta_cons_err, self.theta_var_err, self.I_cons_err, self.I_var_err)
        #self.theta_tot_err = self.theta_total_err(self.p_cons, self.p_var, self.theta_cons, self.theta_var, self.I_cons, self.I_var, self.p_cons_err, self.p_var_err, self.theta_cons_err, self.theta_var_err, self.I_cons_err, self.I_var_err)
    
    
    def p_total(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var):
        return np.sqrt( ( (p_cons*I_cons)**2 + (p_var*I_var)**2 + 2*p_cons*I_cons*p_var*I_var*np.cos(2*np.deg2rad(theta_cons-theta_var)) ) / ((I_cons + I_var)**2) )
    
    def theta_total(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var):
        return np.rad2deg(np.arctan( ( p_cons*I_cons*np.cos(2*np.deg2rad(theta_cons)) + p_var*I_var*np.cos(2*np.deg2rad(theta_var)) ) / ( p_cons*I_cons*np.sin(2*np.deg2rad(theta_cons)) + p_var*I_var*np.sin(2*np.deg2rad(theta_var)) ) )) / 2. +90

    def p_total_err(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var, sigma_p_cons, sigma_p_var, sigma_theta_cons, sigma_theta_var, sigma_I_cons, sigma_I_var):
        theta_cons_rad = np.deg2rad(theta_cons)
        theta_var_rad = np.deg2rad(theta_var)
        sigma_theta_cons = np.deg2rad(sigma_theta_cons)
        sigma_theta_var = np.deg2rad(sigma_theta_var)

        f = ( (p_cons * I_cons)**2 + (p_var * I_var)**2 + 2 * p_cons * I_cons * p_var * I_var * np.cos(2 * (theta_cons_rad - theta_var_rad)) ) / ( (I_cons + I_var)**2 )
    
        partial_p_cons = (2 * p_cons * I_cons**2 + 2 * I_cons * p_var * I_var * np.cos(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var)**2)
        partial_p_var = (2 * p_var * I_var**2 + 2 * I_var * p_cons * I_cons * np.cos(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var)**2)
        partial_theta_cons = (-4 * p_cons * I_cons * p_var * I_var * np.sin(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var)**2)
        partial_theta_var = (4 * p_cons * I_cons * p_var * I_var * np.sin(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var)**2)
        partial_I_cons = (2 * p_cons**2 * I_cons + 2 * p_cons * p_var * I_var * np.cos(2 * (theta_cons_rad - theta_var_rad)) - f * 2 * I_cons) / ((I_cons + I_var)**2)
        partial_I_var = (2 * p_var**2 * I_var + 2 * p_cons * p_var * I_cons * np.cos(2 * (theta_cons_rad - theta_var_rad)) - f * 2 * I_var) / ((I_cons + I_var)**2)

        sigma_f = np.sqrt(
            (partial_p_cons * sigma_p_cons)**2 +
            (partial_p_var * sigma_p_var)**2 +
            (partial_theta_cons * sigma_theta_cons)**2 +
            (partial_theta_var * sigma_theta_var)**2 +
            (partial_I_cons * sigma_I_cons)**2 +
            (partial_I_var * sigma_I_var)**2
        )
    
        sigma_p_total = (1 / (2 * np.sqrt(f))) * sigma_f
        return sigma_p_total
    
    def theta_total_err(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var, sigma_p_cons, sigma_p_var, sigma_theta_cons, sigma_theta_var, sigma_I_cons, sigma_I_var):
        theta_cons_rad = np.deg2rad(theta_cons)
        theta_var_rad = np.deg2rad(theta_var)
        sigma_theta_cons = np.deg2rad(sigma_theta_cons)
        sigma_theta_var = np.deg2rad(sigma_theta_var)
    
        numerator = p_cons * I_cons * np.cos(2 * theta_cons_rad) + p_var * I_var * np.cos(2 * theta_var_rad)
        denominator = p_cons * I_cons * np.sin(2 * theta_cons_rad) + p_var * I_var * np.sin(2 * theta_var_rad)
    
        theta_total_rad = 0.5 * np.arctan2(numerator, denominator)
    
        partial_p_cons = (I_cons * np.cos(2 * theta_cons_rad) * denominator - I_cons * np.sin(2 * theta_cons_rad) * numerator) / (denominator**2 + numerator**2)
        partial_p_var = (I_var * np.cos(2 * theta_var_rad) * denominator - I_var * np.sin(2 * theta_var_rad) * numerator) / (denominator**2 + numerator**2)
        partial_theta_cons = (-2 * p_cons * I_cons * np.sin(2 * theta_cons_rad) * denominator + 2 * p_cons * I_cons * np.cos(2 * theta_cons_rad) * numerator) / (denominator**2 + numerator**2)
        partial_theta_var = (-2 * p_var * I_var * np.sin(2 * theta_var_rad) * denominator + 2 * p_var * I_var * np.cos(2 * theta_var_rad) * numerator) / (denominator**2 + numerator**2)
        partial_I_cons = (p_cons * np.cos(2 * theta_cons_rad) * denominator - p_cons * np.sin(2 * theta_cons_rad) * numerator) / (denominator**2 + numerator**2)
        partial_I_var = (p_var * np.cos(2 * theta_var_rad) * denominator - p_var * np.sin(2 * theta_var_rad) * numerator) / (denominator**2 + numerator**2)
    
        sigma_theta_total = 0.5 * np.sqrt(
            (partial_p_cons * sigma_p_cons)**2 +
            (partial_p_var * sigma_p_var)**2 +
            (partial_theta_cons * sigma_theta_cons)**2 +
            (partial_theta_var * sigma_theta_var)**2 +
            (partial_I_cons * sigma_I_cons)**2 +
            (partial_I_var * sigma_I_var)**2
        )
    
        return np.rad2deg(sigma_theta_total)
