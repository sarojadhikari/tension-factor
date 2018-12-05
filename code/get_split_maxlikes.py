import numpy as np
from split_likelihoods import split_likelihood

plaTT = split_likelihood(which="aTTsplit", lsplit=1000)
plbTT = split_likelihood(which="bTTsplit", lsplit=1000)

plaEE = split_likelihood(which="aEEsplit", lsplit=1000)
plbEE = split_likelihood(which="bEEsplit", lsplit=1000)

from scipy.optimize import fmin, differential_evolution as dev
bounds = plaTT.bounds

res_aTT = dev(plaTT.neglogLike, bounds, disp=True, tol=0.005)
res_aTT2 = fmin(plaTT.neglogLike, x0=res_aTT.x)
np.save("maxlikefits/res_splitaTT_notauprior.npy", res_aTT2)

res_bTT = dev(plbTT.neglogLike, bounds, disp=True, tol=0.005)
res_bTT2 = fmin(plbTT.neglogLike, x0=res_bTT.x)
np.save("maxlikefits/res_splitbTT_notauprior.npy", res_bTT2)

res_aEE = dev(plaEE.neglogLike, bounds, disp=True, tol=0.005)
res_aEE2 = fmin(plaEE.neglogLike, x0=res_aEE.x)
np.save("maxlikefits/res_splitaEE_notauprior.npy", res_aEE2)

res_bEE = dev(plbEE.neglogLike, bounds, disp=True, tol=0.005)
res_bEE2 = fmin(plbEE.neglogLike, x0=res_bEE.x)
np.save("maxlikefits/res_splitbEE_notauprior.npy", res_bEE2)

