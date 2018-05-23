**Sample code to get the best-fit cosmological parameters using scipy.optimize**
```
from likelihoods import Planck_plik_lite_likelihood
plikliteTT = Planck_plik_lite_likelihood(which="TT")
    
def neglnLikeT(params):
    return (-1.)*plikliteTT.logLike(params)
        
from scipy.optimize import differential_evolution as dev
bounds = plikliteTT.bounds
res = dev(neglnLikeT, bounds, disp = True, tol=0.005)
```
