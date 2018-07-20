# example script to compute SPTSZ likelihood using cosmoslik
#########

from cosmoslik import *
from SPTSZ_lik import sptsz
import numpy as np

sl = Slik(sptsz())

lkl, e = sl.evaluate(**{'cosmo.As': 2.1E-9,
                       'cosmo.ns': 0.96,
                       'cosmo.ombh2': 0.0221,
                       'cosmo.omch2': 0.12,
                       'cosmo.tau': 0.07,
                       'cosmo.theta': 0.01041,
                       'sptsz.cal': 0.983,
                       'sptsz.egfs.Acl': 20,
                       'sptsz.egfs.Aps': 5,
                       'sptsz.egfs.Asz': 5})

def lnlike(cosmo_params = [3.05, 0.96, 0.0221, 0.12, 0.07, 0.01041],
           sptcalparams = [0.983, 20, 5, 5]):
    """
    return the loglikelihood given six-LCDM parameters
    """

    cosmo_dict = {'cosmo.As': np.exp(cosmo_params[0])/1.E10,
            'cosmo.ns': cosmo_params[1],
            'cosmo.ombh2': cosmo_params[2],
            'cosmo.omch2': cosmo_params[3],
            'cosmo.tau': cosmo_params[4],
            'cosmo.theta': cosmo_params[5]}

    sptsz_dict = {'sptsz.cal': sptcalparams[0],
                  'sptsz.egfs.Acl': sptcalparams[1],
                  'sptsz.egfs.Aps': sptcalparams[2],
                  'sptsz.egfs.Asz': sptcalparams[3]}

    cosmo_dict.update(sptsz_dict)

    lkl, e = sl.evaluate(**cosmo_dict)

    return lkl

def loglike(params):
    """
    for use with scipy.optimize functions
    """
    print (params)
    ln10As, ns, ombh2, omch2, tau, theta, cal, Acl, Aps, Asz = params

    return lnlike(cosmo_params=[ln10As, ns, ombh2, omch2, tau, theta],
                  sptcalparams=[cal, Acl, Aps, Asz])

def Prior(cube, ndim=10):
    """
    uniform prior (parameter limits) for the parameters;
    perhaps do this more manually!!
    """
    prs = likelihoods.priors(sl.params)
    prs.uniform_priors[0] = ('cosmo.ln10As', 2.7, 3.4)
    for i in range(ndim):
        a, b = prs.uniform_priors[i][1:3]
        # if either one is inf or -inf then look at the Gaussian priors and set
        # the limits to be the value + 10 sd
        if (b==np.inf):
            b = 50
        cube[i] = a + cube[i]*(b-a)

    return cube
