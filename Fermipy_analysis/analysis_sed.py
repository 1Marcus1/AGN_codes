from fermipy.gtanalysis import GTAnalysis

import numpy as np
from matplotlib import pyplot as plt #matplotlib.pyplot is the package we will use to plot our light-curve.
from astropy.table import Table #astropy.table allows us to read fits tables, which is the structure of our ROI file. 
#import os #os lets us easily perform file manipulation within Python. 

#import yaml


mes = 'agosto'

gta = GTAnalysis('config.yaml',logging={'verbosity': 3}, fileio={'outdir': 'mkn501_Fit_{}'.format(mes)})
gta.setup()

gta.optimize()

# Free Normalization of all Sources within 3 deg of ROI center
gta.free_sources(distance=10.0,pars='norm')

# Free all parameters of isotropic and galactic diffuse components
gta.free_source('galdiff', pars='norm')
gta.free_source('isodiff', pars='norm')

gta.fit()

gta.residmap(make_plots=True)

gta.tsmap(make_plots=True)

gta.find_sources(sqrt_ts_threshold=5.0, min_separation=0.5)

gta.tsmap(make_plots=True)

gta.delete_sources(minmax_ts=[-np.inf, 9])

gta.free_source('MKN501')

gta.write_roi('fit0',make_plots=True)

# Get the sed results from the return argument
gta.sed('MKN501', make_plots="True")
sed = gta.sed('MKN501', outfile='sed.fits')

print('SED')
print('dnde: ', sed['dnde'])
print('e2dnde: ', sed['e2dnde'])
print('e2dnde_err: ', sed['e2dnde_err'])
print('spectral index')
print('alpha: ', sed['param_values'][1])
print('alpha_err: ', sed['param_errors'][1])
print('beta: ', sed['param_values'][2])
print('beta_err: ', sed['param_errors'][2])

#save the SED values
np.savetxt('sed_{}.txt'.format(mes), np.c_[sed['dnde'],sed['e2dnde'],sed['e2dnde_err']], delimiter=';', header='dnde;e2dnde;e2dnde_err')

#save the spectral index values
np.savetxt('spectral_index_{}.txt'.format(mes), np.c_[sed['param_values'][1],sed['param_errors'][1],sed['param_values'][2], sed['param_errors'][2]], delimiter=';', header='alpha;alpha_err;beta;beta_err')

loc = gta.localize('MKN501')

