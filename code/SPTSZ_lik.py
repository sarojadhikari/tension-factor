from cosmoslik import *


param = param_shortcut('start','scale')

class sptsz(SlikPlugin):

    def __init__(self):
        super().__init__()

    
        self.sptsz = likelihoods.SPTSZ_lowl_2017.SPTSZ_lowl(
                        ab_on=True,
                        cal = param(0.983,0.0015,min=0,gaussian_prior=(0.983,0.0015))
                        )

        # set up cosmological params and solver
        self.cosmo = models.cosmology("lcdm")
        self.cmb = models.camb(lmax=4000)

        # parameters, range and priors setup
        ####################################
        self.cosmo.As = param(2.1E-9, 0.1E-9, min=1E-9, max=3E-9)
        self.cosmo.tau = param(0.07,0.02, min=0.005, max=0.2, gaussian_prior=(0.07,0.02))
        self.cosmo.ns = param(0.96, 0.01, min=0.8, max=1.2)
        self.cosmo.ombh2 = param(0.0221, 0.005, min=0.02, max=0.025)
        self.cosmo.omch2 = param(0.12, 0.01, min=0.09, max=0.15)
        self.cosmo.theta = param(0.01041, 0.0001, min=0.0103, max=0.0105)

        self.priors = likelihoods.priors(self)

        self.sampler = samplers.metropolis_hastings(self,
             num_samples=20000,
        #     output_file='/home/kmaylor/planck_and_spt_combo/Saved_chains/SPTSZ_lcdm.chain',
        #     cov_est='/nfs/home/kmaylor/Python_Projects/cosmoslik_proposal_covmats/SPT_chainz_150x150.covmat',
	         print_level=2,
             output_extra_params=['cosmo.H0'])

    def __call__(self):
        # set the calibration parameters the same

        # compute likelihood
        cmb = self.cmb(**self.cosmo)
        return self.sptsz(cmb) + self.priors(self)
