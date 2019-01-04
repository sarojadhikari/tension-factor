import numpy as np
import pymultinest

from likelihoods import Planck_plik_lite_likelihood
from split_likelihoods import split_likelihood


xTT = np.load("maxlikefits/res_TT_wtauprior_wlowl.npy")
xTa = np.load("maxlikefits/res_splitaTT_wtauprior.npy")
xTb = np.load("maxlikefits/res_splitbTT_notauprior_wlowl.npy")

#xEE = np.load("maxlikefits/res_EE_notauprior.npy")
#xEa = np.load("maxlikefits/res_splitaEE_notauprior.npy")
#xEb = np.load("maxlikefits/res_splitbEE_notauprior.npy")

#xTTEE = np.load("maxlikefits/resTTEE_notauprior.npy")

plTT = Planck_plik_lite_likelihood(which="TT", taumean=xTa[5], tausigma=0.02, lowlTT=True)
plTTa = split_likelihood(which="aTTsplit")
plTTb = split_likelihood(which="bTTsplit")

plTTa.get_camb_Cls(xTa)
plTTb.get_camb_Cls(xTb)

#clbfTT = plTT.bmTT@(plTT.mufac*(plTT.cmb.cambTCls[30:2509]))

clbfTTa = plTTa.bmaTT@(plTTa.mufac*(plTTa.cmb.cambTCls[plTTa.lsplit:2509]))
clbfTTb = plTTb.bmbTT@(plTTb.mufac*(plTTb.cmb.cambTCls[30:plTTb.lsplit]))

plTT.cldata = np.append(clbfTTb, clbfTTa)
plTT.cls_meas_low = plTT.mufac*(plTTb.cmb.cambTCls[2:30])

def Prior(cube, ndim, nparams):
    for i in range(ndim):
        a, b = plTT.bounds[i][0], plTT.bounds[i][1]
        cube[i] = a + cube[i]*(b-a)
        
def Loglike(cube, ndim, nparams, lnew):
    params = [cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]]
    return plTT.logLike(params)

Ev_sep = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verbose=True,
                         resume=True, evidence_tolerance=0.001,  n_live_points=500, sampling_efficiency="model",
                         outputfiles_basename=u'chains2019/TTsplitwlowl_separate_')

plTT.get_camb_Cls(xTT)
clbfTT = plTT.bmTT@(plTT.mufac*(plTT.cmb.cambTCls[30:2509]))

plTT.taumean = xTT[5]

plTT.cldata = clbfTT
plTT.cls_meas_low = plTT.mufac*(plTT.cmb.cambTCls[2:30])

Ev_com = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verbose=True,
			 resume=True, evidence_tolerance=0.001, n_live_points=500, sampling_efficiency="model",
			 outputfiles_basename=u'chains2019/TTsplitwlowl_combined_')
