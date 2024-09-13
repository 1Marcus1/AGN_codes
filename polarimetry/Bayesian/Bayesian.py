import numpy as np
import matplotlib.pyplot as plt
import os
import Bayesian_class as bc

# data
pf_new = np.loadtxt('data_OJ287.txt', delimiter=';')[:,3]
pd_new = np.loadtxt('data_OJ287.txt', delimiter=';')[:,1]
pa_new = np.loadtxt('data_OJ287.txt', delimiter=';')[:,2]
time   = np.loadtxt('data_OJ287.txt', delimiter=';')[:,0]

pa_new[ pa_new < 90 ] += 90 # pi arctangent ambiguity

I_obs = []
for i in range(0,len(time)):
    I_obs.append( pf_new[i]/pd_new[i] )

Q_obs = []
for i in range(0,len(time)):
    Q_obs.append( pf_new[i]*np.cos( np.deg2rad(2*pa_new[i]) ) )

U_obs = []
for i in range(0,len(time)):
    U_obs.append( pf_new[i]*np.sin(np.deg2rad(2*pa_new[i])) )


Q_obs = np.array(Q_obs)
U_obs = np.array(U_obs)
I_obs = np.array(I_obs)

# Bayesian Components Separation
model = bc.Bayesian(Q_obs, U_obs, I_obs, 10, int(1000))

long_term = np.array(model.long_term)
long_term_std = np.array(model.long_term_std)

print('Long term: \n', long_term)
print('\nLong term std: \n', long_term_std)

total = np.array([Q_obs, U_obs, I_obs])
'''
Here the code will crash at first because of an error in the syntax that I didn't fix yet, but it will print the long term trent at the terminal, so you can
comment the bayesian procedure and add the results in the long_term commented lines below.
'''
#long_term = np.array([np.array([0.35308163]), np.array([-0.14431064]), np.array([0.38168214])])
#long_term_std = np.array([np.array([0.01291075]), np.array([0.01621736]), np.array([0.01551229])])


short_term = np.array([Q_obs, U_obs, I_obs]) - long_term
short_term_std = long_term_std

pf_l     = np.sqrt(long_term[0]**2 + long_term[1]**2)
pf_l_err = np.sqrt( (long_term_std[0]*long_term[0]/pf_l)**2 + (long_term_std[1]*long_term[1]/pf_l)**2 )

pf_s     = np.sqrt(short_term[0]**2 + short_term[1]**2)
pf_s_err = np.sqrt( (short_term_std[0]*short_term[0]/pf_l)**2 + (short_term_std[1]*short_term[1]/pf_s)**2 )

pa_l     = np.rad2deg( np.arctan( long_term[1]/long_term[0] )/2.0 ) + 90 + 90
pa_l_err = np.sqrt( ( (0.5/long_term[0])/( 1 + (0.5*long_term[1]/long_term[0])**2 ) )**2 + (  (0.5*long_term[1]/(long_term[0]**2))/( 1 + (0.5*long_term[1]/long_term[0])**2 )  )**2 )

pa_s     = np.rad2deg( np.arctan( short_term[1]/short_term[0] )/2.0 ) + 90 + 90
pa_s_err = np.sqrt( ( (0.5/short_term[0])/( 1 + (0.5*short_term[1]/short_term[0])**2 ) )**2 + (  (0.5*short_term[1]/(short_term[0]**2))/( 1 + (0.5*short_term[1]/short_term[0])**2 )  )**2 )

print(pf_l, pf_l_err)
print(pa_l, pa_l_err)
#print('Short term: \n', short_term)
#print('\nShort term std: \n', short_term_std)


fig, ax = plt.subplots(2, sharex=True)

ax[0].plot( time , Q_obs , 'o' , color='blue' , label='data' )
ax[0].plot( [time[0], time[-1]] , [long_term[0], long_term[0]] , '--' , color='black' , label='constant' )
ax[0].fill_between( [time[0], time[0]] , long_term[0]-long_term_std[0] , long_term[0]+long_term_std[0] , alpha=0.2 )
ax[0].plot( time , short_term[0] , '-' , color='black' , label='Variable' )
ax[0].fill_between( time , short_term[0]-short_term_std[0] , short_term[0]+short_term_std[0] , alpha=0.2 )

ax[0].set_ylabel('Q')
ax[0].grid(linestyle='--', linewidth=0.6)
ax[0].legend(loc='best', ncols=2)
ax[0].set_title('Bayesian Modeling - OJ287')

ax[1].plot( time , U_obs , 'o' , color='blue' , label='data' )
ax[1].plot( [time[0], time[-1]] , [long_term[1], long_term[1]] , '--' , color='black' , label='constant' )
ax[1].fill_between( [time[0], time[0]] , long_term[1]-long_term_std[1] , long_term[1]+long_term_std[1] , alpha=0.2 )
ax[1].plot( time , short_term[1] , '-' , color='black' , label='Variable' )
ax[1].fill_between( time , short_term[1]-short_term_std[1] , short_term[1]+short_term_std[1] , alpha=0.2 )

ax[1].set_ylabel('U')
ax[1].grid(linestyle='--', linewidth=0.6)
plt.xlabel('Time')

plt.show()


fig, ax = plt.subplots(2, sharex=True)

ax[0].plot( time , pf_new , 'o' , color='blue' , label='data' )
ax[0].plot( [time[0], time[-1]] , [pf_l[0], pf_l[0]] , '--' , color='black' , label='long term' )
ax[0].fill_between( [time[0], time[0]] , pf_l[0]-pf_l_err[0] , pf_l[0]+pf_l_err[0] , alpha=0.2 )
ax[0].plot( time , pf_s , '-' , color='black' , label='short term' )
ax[0].fill_between( time , pf_s-pf_s_err , pf_s+pf_s_err , alpha=0.2 )

ax[0].set_ylabel('PF')
ax[0].grid(linestyle='--', linewidth=0.6)
ax[0].legend(loc='best')
#ax[0].set_title('Bayesian Modeling - OJ287')

ax[1].plot( time , pa_new , 'o' , color='blue' , label='data' )
ax[1].plot( [time[0], time[-1]] , [pa_l[0], pa_l[0]] , '--' , color='black' , label='long term' )
ax[1].fill_between( [time[0], time[0]] , pa_l[0]-pa_l_err[0] , pa_l[0]+pa_l_err[0] , alpha=0.2 )
ax[1].plot( time , pa_s , '-' , color='black' , label='short term' )
ax[1].fill_between( time , pa_s-pa_s_err , pa_s+pa_s_err , alpha=0.2 )

ax[1].set_ylabel('PA (º)')
ax[1].grid(linestyle='--', linewidth=0.6)
plt.xlabel('Time')

plt.savefig('oj287_bayesian_separation.png')
plt.show()
