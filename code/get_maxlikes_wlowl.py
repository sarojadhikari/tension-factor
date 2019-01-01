import numpy as np
from likelihoods import Planck_plik_lite_likelihood
from split_likelihoods import splitlikelihood

plTT = Planck_plik_lite_likelihood(which="TT", tausigma=np.inf, lowlTT=True)

plbTT = split_likelihood(which="bTTsplit", lsplit=1000, lowlTT=True)

plTT.tausigma = np.inf # in this version we put no prior on tau 

def neglnLikeT(params):
    return (-1.)*plTT.logLike(params)

def neglnLikebT(params):
    return (-1.)*plbTT.logLike(params)

from scipy.optimize import fmin, differential_evolution as dev
bounds = plTT.bounds

resTT = dev(neglnLikeT, bounds, disp=True, tol=0.005)
resTT2 = fmin(neglnLikeT, x0=resTT.x)
np.save("resTT_wlowl.npy", resTT2)

resbTT = dev(neglnLikebT, bounds, disp=True, tol=0.005)
resbTT2 = fmin(neglnLikebT, x0=resbTT.x)
np.save("resbTT_wlowl.npy", resbTT2)

