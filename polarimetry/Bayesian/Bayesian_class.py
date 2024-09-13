import numpy as np
import matplotlib.pyplot as plt
import os

class Bayesian:
    def __init__(self, Q, U, I, n, num_samples, w=0.5):
        
        self.Q = Q
        self.U = U
        self.I = I

        self.w = w
        self.n = n
        self.num_samples = num_samples

        self.Q_score = self.Z_score(self.Q)
        self.U_score = self.Z_score(self.U)
        self.I_score = self.Z_score(self.I)

        self.x = np.array([self.Q, self.U, self.I])

        sample_in = np.array([ [ np.random.normal( np.mean(self.Q) ) ]*len(self.I) , [ np.random.normal( np.mean(self.U) ) ]*len(self.I) ])

        self.long_term_all = []
        self.long_term = []

        for i in range(1, self.n+1):
            print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {i} AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            aaa = ( self.MCMC(sample_in, self.num_samples, self.x)[0] )
            self.long_term.append( np.array([np.mean(aaa[0]), np.mean(aaa[1]), np.mean(aaa[2]) ]) )
        
        self.long_term_a = np.array( self.long_term )

        self.long_term = [ self.long_term_a[:,0].mean() , self.long_term_a[:,1].mean() , self.long_term_a[:,2].mean() ]
        self.long_term_std = [ self.long_term_a[:,0].std() , self.long_term_a[:,1].std() , self.long_term_a[:,2].std() ]



    def Z_score(self, A):
        return (A - np.mean(A))/np.std(A)
    
    # Likelihood
    def L(self, y, x):
        PF_s = self.Z_score( np.sqrt( np.power(self.x[0]-y[0], 2) + np.power(self.x[1]-y[1], 2) ) )
        sigma_PF_s = np.std(PF_s)

        L1 = np.exp( -np.power(self.I_score - PF_s, 2) / (2 * sigma_PF_s**2) ) / np.sqrt( 2 * np.pi * sigma_PF_s**2 )
        return np.prod(L1)
    
    # Prior
    def PI(self, Q):
        b = np.exp( -np.power(Q[1:] - Q[:-1], 2) / (2 * self.w**2) ) / (2 * np.pi * self.w**2)
        return np.prod(b)
    
    # Posterior Probability
    def P(self, y, x):
        C = 1e-55
        return self.L(y, self.x) * self.PI(y[0]) * self.PI(y[1]) / C
    
    # Markov Chain Monte Carlo (MCMC)
    def MCMC(self, sample_in, num_samples, x):
        acceptance_rate = 0
        sample_out = [sample_in]

        d = []
        
        count = 0
        acceptance_prob = [self.P(sample_in, x)]

        j = 1
        while j<= num_samples:
            sample_out.append(np.array( [ [np.random.normal( np.mean(sample_out[j-1][0]) )]*len(self.I) , [np.random.normal( np.mean(sample_out[j-1][1]) )]*len(self.I) ] ))
            acceptance_prob.append( self.P(sample_out[-1], x) )

            u = np.random.uniform(0.92, 1)

            if acceptance_prob[-1] >= acceptance_prob[-2]:
                count = 0
                acceptance_rate += 1

                if j%300==0 or j==num_samples or j==1:
                    d.append( [j, sample_out[-1][0], sample_out[-1][1]] )
                    print('prob: ', acceptance_prob[j], '	iteration: ', j)
                
                j += 1
            
            elif (acceptance_prob[-1]/acceptance_prob[-2])>u:
                count = 0
                acceptance_rate += 1

                if j%300==0 or j==num_samples or j==1:
                    d.append( [j, sample_out[-1][0], sample_out[-1][1]] )
                    print('prob: ', acceptance_prob[j], '	iteration: ', j)
                
                j += 1

            else:# acceptance_prob[-1] <= acceptance_prob[-2] or (acceptance_prob[-1]/acceptance_prob[-2]) < u:
                del sample_out[-1]
                del acceptance_prob[-1]
                count += 1

                if count%10000 == 0:
                    print('count: ', count, '	w={}'.format(self.w))
        
        sample_out = np.array(sample_out)
        I_l = np.sqrt( sample_out[-1][0]**2 + sample_out[-1][1]**2 )
        final_sample = [ sample_out[-1][0] , sample_out[-1][1], I_l ]
        return final_sample, acceptance_prob, acceptance_rate, d
