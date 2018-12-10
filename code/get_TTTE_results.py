# get the required TTTE evidences starting from best-fit TE and TTTE
# cosmological parameters -- the TT best-fit is already done

import numpy as np
import pymultinest

from likelihoods import Planck_plik_lite_likelihood

suf = "notauprior"
xTT = np.load("maxlikefits/res_TT_"+suf+".npy")

# now get TE and TTTE best-fits
plTTTE = Planck_plik_lite_likelihood(which="TTTE", tausigma=np.inf)

from scipy.optimize import fmin, differential_evolution as dev
bounds = plTTTE.bounds

xTTTE = dev(plTTTE.neglogLike, bounds, disp=True, tol=0.005)
xTTTE = fmin(plTTTE.neglogLike, x0=xTTTE.x)
np.save("maxlikefits/res_TTTE_"+suf+".npy", xTTTE)

plTTTE.get_camb_Cls(xTTTE)
clbfTT = plTTTE.bmTT@(plTTTE.mufac*(plTTTE.cmb.cambTCls[30:2509]))
clbfTE = plTTTE.bmTE@(plTTTE.mufac*(plTTTE.cmb.cambTECls[30:1997]))

plTTTE.cldata = np.append(clbfTT, clbfTE)

def Prior(cube, ndim, nparams):
    for i in range(ndim):
        a, b = plTTTE.bounds[i][0], plTTTE.bounds[i][1]
        cube[i] = a + cube[i]*(b-a)
        
def Loglike(cube, ndim, nparams, lnew):
    params = [cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]]
    return plTTTE.logLike(params)

Ev_com = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verbose=True,
                         resume=False, n_live_points=400, sampling_efficiency="model",
                         outputfiles_basename=u'chains/TTTE_combined_')

plTT = Planck_plik_lite_likelihood(which="TT", tausigma=np.inf)
plTT.get_camb_Cls(xTT)
clbfTT = plTT.bmTT@(plTT.mufac*(plTT.cmb.cambTCls[30:2509]))

plTE = Planck_plik_lite_likelihood(which="TE", tausigma=np.inf)
xTE = dev(plTE.neglogLike, bounds, disp=True, tol=0.005)
xTE = fmin(plTE.neglogLike, x0=xTE.x)
np.save("maxlikefits/res_TE_"+suf+".npy", xTE)

plTE.get_camb_Cls(xTE)
clbfTE = plTE.bmTE@(plTE.mufac*(plTE.cmb.cambTECls[30:1997]))

plTTTE.cldata = np.append(clbfTT, clbfTE)

Ev_sep = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verbose=True,
                         resume=False, n_live_points=400, sampling_efficiency="model",
                         outputfiles_basename=u'chains/TTTE_separate_')

