import numpy as np
from split_likelihoods import split_likelihood

plaTT = split_likelihood(which="aTTsplit", lsplit=1000)
plbTT = split_likelihood(which="bTTsplit", lsplit=1000)

plaEE = split_likelihood(which="aEEsplit", lsplit=1000)
plbEE = split_likelihood(which="bEEsplit", lsplit=1000)

from scipy.optimize import fmin, differential_evolution as dev
bounds = plaTT.bounds[:-1]

fixtau = 0.07

def nlikeaTT(params):
    return plaTT.neglogLike(np.append(params, fixtau))

def nlikeaEE(params):
    return plaEE.neglogLike(np.append(params, fixtau))

def nlikebTT(params):
    return plbTT.neglogLike(np.append(params, fixtau))

def nlikebEE(params):
    return plbEE.neglogLike(np.append(params, fixtau))


res_aTT = dev(nlikeaTT, bounds, disp=True, tol=0.005)
res_aTT2 = fmin(nlikeaTT, x0=res_aTT.x)
np.save("maxlikefits/res_splitaTT_fixtau.npy", res_aTT2)

res_bTT = dev(nlikebTT, bounds, disp=True, tol=0.005)
res_bTT2 = fmin(nlikebTT, x0=res_bTT.x)
np.save("maxlikefits/res_splitbTT_fixtau.npy", res_bTT2)

res_aEE = dev(nlikeaEE, bounds, disp=True, tol=0.005)
res_aEE2 = fmin(nlikeaEE, x0=res_aEE.x)
np.save("maxlikefits/res_splitaEE_fixtau.npy", res_aEE2)

res_bEE = dev(nlikebEE, bounds, disp=True, tol=0.005)
res_bEE2 = fmin(nlikebEE, x0=res_bEE.x)
np.save("maxlikefits/res_splitbEE_fixtau.npy", res_bEE2)

from likelihoods import Planck_plik_lite_likelihood

plTT = Planck_plik_lite_likelihood(which="TT", tausigma=np.inf)
plEE = Planck_plik_lite_likelihood(which="EE", tausigma=np.inf)

def nlikeTT(params):
    return plTT.neglogLike(np.append(params, fixtau))

def nlikeEE(params):
    return plEE.neglogLike(np.append(params, fixtau))

res_TT = dev(nlikeTT, bounds, disp=True, tol=0.005)
res_TT2 = fmin(nlikeTT, x0=res_TT.x)
np.save("maxlikefits/res_TT_fixtau.npy", res_TT2)

res_EE = dev(nlikeEE, bounds, disp=True, tol=0.005)
res_EE2 = fmin(nlikeEE, x0=res_EE.x)
np.save("maxlikefits/res_EE_fixtau.npy", res_EE2)

