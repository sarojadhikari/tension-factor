import numpy as np
import pymultinest

from likelihoods import Planck_plik_lite_likelihood

xTT = np.load("maxlikefits/resTT_notauprior.npy")
xEE = np.load("maxlikefits/resEE_notauprior.npy")
xTTEE = np.load("maxlikefits/resTTEE_notauprior.npy")

plTTEE = Planck_plik_lite_likelihood(which="TTEE", tausigma=np.inf)

plTTEE.get_camb_Cls(xTTEE)
clbfTT = plTTEE.bmTT@(plTTEE.mufac*(plTTEE.cmb.cambTCls[30:2509]))
clbfEE = plTTEE.bmEE@(plTTEE.mufac*(plTTEE.cmb.cambECls[30:1997]))

plTTEE.cldata = np.append(clbfTT, clbfEE)

def Prior(cube, ndim, nparams):
    for i in range(ndim):
        a, b = plTTEE.bounds[i][0], plTTEE.bounds[i][1]
        cube[i] = a + cube[i]*(b-a)
        
def Loglike(cube, ndim, nparams, lnew):
    params = [cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]]
    return plTTEE.logLike(params)

Ev_com = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verbose=True,
                         resume=False, n_live_points=400, sampling_efficiency="model",
                         outputfiles_basename=u'chains/TTEE_combined_')
