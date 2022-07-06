import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from fermipy.gtanalysis import GTAnalysis
import pickle

tmin = 418003203
tmax = 433814403
b = np.arange(tmin, tmax, 86400.*7.)

gta = GTAnalysis('config.yaml',logging={'verbosity': 3})
gta.setup()

# Free Normalization of all Sources within 3 deg of ROI center
gta.free_sources(distance=3.0,pars='norm')

# Free all parameters of isotropic and galactic diffuse components
gta.free_source('galdiff')
gta.free_source('isodiff')

gta.free_source('MKN501')

gta.write_roi('fit0',make_plots=True)

# Get the sed results from the return argument
gta.sed('MKN501', make_plots="True")
#sed = gta.sed('4FGL J1104.4+3812', outfile='sed.fits')
sed = gta.sed('MKN501', outfile='sed.fits')

#Print the SED flux values
print(sed['flux'])
np.savetxt('sed.txt', np.c_[sed['energy'],sed['flux'],sed['flux_err']], delimiter=';', header='flux (ph cm-2 s-1);flux error')

gta.optimize()
gta.write_roi('fit1',make_plots=True)

# Lightcurve
lc = gta.lightcurve('4FGL J1653.8+3945', time_bins=b, make_plots=True, multithread=True)

#Print the Lightcurve values
print(lc['flux'])
print(lc['flux_err'])
np.savetxt('lightcurve.txt', np.c_[lc['flux'],lc['flux_err']], delimiter=';', header='flux (ph cm-2 s-1);flux error')
