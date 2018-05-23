**Sample script to get the best-fit cosmological parameters using scipy.optimize**
```python
from likelihoods import Planck_plik_lite_likelihood
plTT = Planck_plik_lite_likelihood(which="TT")
    
def neglnLikeT(params):
    return (-1.)*plTT.logLike(params)
        
from scipy.optimize import differential_evolution as dev
boundsTT = plTT.bounds
resultTT = dev(neglnLikeT, bounds, disp=True, tol=0.005)
print (resultTT.x, resultTT.fun)
```
For the dataset-evidence ratio, we compute evidences using two "best-fit data realizations". For example, to compare Planck TT and EE power spectrum data, we will need to similarly also find resultEE and resultTTEE, and then compute two evidences using multinest.
```python
import pymultinest

plTTEE.get_camb_Cls(resultTTEE.x)
clbfTT = plTTEE.bmTT@(plTTEE.mufac*(plTTEE.cmb.cambTCls[30:2509]))
clbfEE = plTTEE.bmEE@(plTTEE.mufac*(plTTEE.cmb.cambECls[30:1997]))

# set the data vector to the best-fit model; this is how we have alterled the 
# definition of evidence so that we can apply the evidence ratio to compare datasets
plTTEE.cldata = np.append(clbfTT, clbfEE)

def Prior(cube, ndim, nparams):
    for i in range(ndim):
        a, b = plTTEE.bounds[i][0], plTTEE.bounds[i][1]
        cube[i] = a + cube[i]*(b-a)

def Loglike(cube, ndim, nparams, lnew):
    return plTTEE.logLike(cube)

Ev_com = pymultinest.run(LogLikelihood=Loglike, Prior=Prior, n_dims=6, verboser=True,
                         resume=False, n_live_points=400, sampling_efficiency="model",
                         outputfiles_basename=u'chains/TTEE_combined_')
```
Now, for the other evidence where we use best-fit values from TT and EE spectra fit separately, we need to get the best-fit data realizations using the corresponding best-fit values.
```python
plTTEE.get_camb_Cls(resultTT.x)
clbfTT = plTTEE.bmTT@(plTTEE.mufac*(plTTEE.cmb.cambTCls[30:2509]))

plTTEE.get_camb_Cls(resultEE.x)
clbfEE = plTTEE.bmEE@(plTTEE.mufac*(plTTEE.cmb.cambECls[30:1997]))

plTTEE.cldata = np.append(clbfTT, clbfEE)
```
