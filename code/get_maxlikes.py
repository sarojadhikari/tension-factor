import numpy as np
from likelihoods import Planck_plik_lite_likelihood
plTT = Planck_plik_lite_likelihood(which="TT", tausigma=np.inf)
plEE = Planck_plik_lite_likelihood(which="EE", tausigma=np.inf)
plTTEE = Planck_plik_lite_likelihood(which="TTEE", tausigma=np.inf)

plTT.tausigma = np.inf # in this version we put no prior on tau 

def neglnLikeT(params):
    return (-1.)*plTT.logLike(params)

def neglnLikeE(params):
    return (-1.)*plEE.logLike(params)

def neglnLikeTTEE(params):
    return (-1.)*plTTEE.logLike(params)

from scipy.optimize import fmin, differential_evolution as dev
bounds = plTT.bounds

resTT = dev(neglnLikeT, bounds, disp=True, tol=0.005)
resTT2 = fmin(neglnLikeT, x0=resTT.x)
np.save("resTTnotauprior.npy", resTT2)

resEE = dev(neglnLikeE, bounds, disp=True, tol=0.005)
resEE2 = fmin(neglnLikeE, x0=resEE.x)
np.save("resEEnotauprior.npy", resEE2)

resTTEE = dev(neglnLikeTTEE, bounds, disp=True, tol=0.005)
resTTEE2 = fmin(neglnLikeTTEE, x0=resTTEE.x)
np.save("resTTEEnotauprior.npy", resTTEE2)