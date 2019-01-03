import numpy as np
import pymultinest

from likelihoods import Planck_plik_lite_likelihood

xTT = np.load("maxlikefits/res_TT_notauprior.npy")
xEE = np.load("maxlikefits/res_EE_wtauprior.npy")
xTTEE = np.load("maxlikefits/res_TTEE_wtauprior.npy")

plTTEE = Planck_plik_lite_likelihood(which="TTEE", taumean=xTTEE[5], tausigma=0.02)

# combined
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
                         outputfiles_basename=u'chains2019/TTEE_combined_')

# now separate
plTTEE.get_camb_Cls(xTT)
clbfTT = plTTEE.bmTT@(plTTEE.mufac*(plTTEE.cmb.cambTCls[30:2509]))
plTTEE.get_camb_Cls(xEE)
clbfEE = plTTEE.bmEE@(plTTEE.mufac*(plTTEE.cmb.cambECls[30:1997]))

plTTEE.cldata = np.append(clbfTT, clbfEE)
plTTEE.taumean = xEE[5]

Ev_sep = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verbose=True,
                         resume=False, n_live_points=400, sampling_efficiency="model",
                         outputfiles_basename=u'chains2019/TTEE_separate_')
