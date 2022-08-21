from fermipy.gtanalysis import GTAnalysis

import numpy as np
from matplotlib import pyplot as plt #matplotlib.pyplot is the package we will use to plot our light-curve.
from astropy.table import Table #astropy.table allows us to read fits tables, which is the structure of our ROI file. 
import os #os lets us easily perform file manipulation within Python. 

fname = "config.yaml"

file = open(fname,'r')
lista = file.readlines()
t_min = int(lista[12].split()[1])
t_max = int(lista[13].split()[1])
file.close()

mes = 'abril'

t_start = t_min                     #Our start time
t_end = t_max                       #Our end time
n = (t_end-t_start)/(86400*1)       #number of bins
t_interval = (t_end - t_start)/n    #Work out the length of a time bin in seconds

bins = []                           #define an empty list
directory = []                      #and another one for the directory names
roi = []                            #and one for our fit name
i = 0                               #Populate both lists, we can easily do this with a while loop
while i < n:
    x = t_start + (t_interval*i)
    bins.append(x)
    directory.append("J1653.8+3945_" + str(x))
    roi.append("J1653.8+3945_fit_" + str(x))
    i = i + 1

bins.append(t_end)


j = 0

while j < n:
    k = j + 1
    gta = GTAnalysis('config.yaml', selection={'tmin' : bins[j], 'tmax' : bins[k]},
                     logging={'verbosity': 3}, fileio={'outdir': directory[j]}) #define our gta object, but with the times from our list
    gta.setup() #photon selection, good time intervals, livetime cube, binning etc
    gta.optimize() #initial optimise
    gta.free_sources(distance=10.0,pars='norm') #frees the point sources
    gta.free_source('galdiff', pars='norm') #frees the galactic diffuse
    gta.free_source('isodiff', pars='norm') #frees the isotropic diffuse
    gta.fit() #full likelihood fit
    gta.sed('4FGL J1653.8+3945') #do an SED, we'll explain why shortly
    gta.write_roi(roi[j]) #save our ROI
    j = j + 1


home = os.getcwd() # get our current working directory and call it home
print(home)
eflux = [] # Make lists for all the quantities we need to plot our light-curve.
eflux_err = []


l = 0
while l < n:  #write a loop which pulls out our flux values from the various ROI files and adds them to lists
    os.chdir(directory[l])
    
    results = Table.read(roi[l] + ".fits")
    eflux.append(results['eflux'][0])
    eflux_err.append(results['eflux_err'][0])
    os.chdir(home)
    l = l + 1

t_centroid = [] #Calculate the central position of each bin. 
for m in bins:
    if m < t_end:
        t_centroid.append(m + (0.5 * t_interval))

print('LightCurve')
print('x: ', t_centroid)
print('x_err: ', t_interval)
print('y: ', eflux)
print('y_err: ', eflux_err)

np.savetxt('lightcurve_{}.txt'.format(mes), np.c_[t_centroid, eflux, eflux_err], delimiter=';', header='time;flux;flux_error')


fig, ax = plt.subplots()
plt.errorbar(x = t_centroid, y = eflux, xerr = t_interval, yerr = eflux_err, fmt='o', ecolor='k')
plt.legend(loc='best',numpoints=1)
plt.xlim([t_start, t_end])
plt.grid(linestyle = '--',linewidth = 0.5)
plt.xlabel('Time (MET)')
plt.ylabel(r'$ Energy Flux $')
ax.set_yscale('log')
#ax.set_xscale('log')
plt.savefig('lightcurve_{}.png'.format(mes))
plt.show()
plt.close()
