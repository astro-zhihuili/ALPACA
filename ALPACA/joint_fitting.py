import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from astropy.convolution import Gaussian1DKernel, convolve
import os

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import pickle

from constants import kms, lambda13_cm, vth, delta_nu_doppler, a31, pc, G, Msun, clight, nu13, sigma13_factor
from get_abs_profile import get_abs_profile
from get_EW_at_b import get_EW_at_b
import radial_velocity_profile_params

varr_axis = np.linspace(-1500*kms, 1500*kms, 301)    # axis for the observed velocity

Nx = 200

delta_x = 1.0

x_array_1d = np.linspace(-100, 100, Nx)  # array of observed unitless frequencies

b_array = np.array([33, 66, 99]) * 1000 * pc  # three impact parameters where EW is evaluated

Ns_array = np.array([5000, 3000, 1000])   # number of line segments used for three impact parameters

# observed EW at three impact parameters with one sigma uncertainties

EW_CII1334_array = np.array([0.90, 0.67, 0.12]) * 1.57 / 1.72

EW_CII1334_err_array = np.array([0.08, 0.12, 0.12]) * 1.57 / 1.72

invvar_b = 1. / (EW_CII1334_err_array**2)   

g = Gaussian1DKernel(stddev=10)   # smoothing kernel to mimic the instrumental resolution (LSF)

def get_absorption_line_profile(name, wavelength):
    
    '''This function extracts the observed DTB absorption profile for fitting.'''
    
    velocity = np.loadtxt('LBG_absorption_lines/%s_%s.txt'%(name, wavelength))[:,0]
    normflux = np.loadtxt('LBG_absorption_lines/%s_%s.txt'%(name, wavelength))[:,1]
    normfluxerr = np.loadtxt('LBG_absorption_lines/%s_%s.txt'%(name, wavelength))[:,2]
    
    return velocity, normflux, normfluxerr

def run_nested(left_bound, right_bound, name, line):
    
    '''This function uses nested sampling to jointly fit the DTB absorption line profile and the EW v.s. b profile.'''
    
    velocity, normflux, normfluxerr = get_absorption_line_profile(name, line)
    
    def prior_transform(theta):   

        return np.array([0.99 * theta[0] + 0.01, 200.0 * theta[1] + 50.0, 3.5 * theta[2] - 4.0, 2000.0 * theta[3] + 300.0, 0.95 * theta[4] + 1.05, 2.0 * theta[5]], dtype=np.float64)
    
    def loglikelihood(x):
        
        '''This function calculates the likelihood of each model.'''
        
        ISM_A, ISM_sigma, logFv, vinf, alpha, gamma = x
        
        rand_v = 120.0   # assuming fixed sigma_cl = 120 km/s
        
        logNcl_0 = 15.0  # assuming fixed ion column density 
        
        Fv = 10**logFv
        
        ISM_component = - ISM_A * np.exp(-velocity**2 / (2 * ISM_sigma**2))  # the ISM absorption component
        
        normflux_subtracted = normflux - ISM_component
        
        model_v, model_I_unconvolved_2d = get_abs_profile(alpha, Fv, vinf * kms, gamma, logNcl_0, rand_v, varr_axis)
        
        model_I_unconvolved = np.flip(np.product(model_I_unconvolved_2d, axis=1)) # doing product along the clump outflow velocity axis to derive I_obs v.s. v_obs profile
        
        model_I = convolve(model_I_unconvolved, g, preserve_nan=True) # convolve with instrumental resolution
        
        max_model_I = np.max(model_I)
        
        model_I /= max_model_I   # making sure the model spectrum is normalized
        
        fltr = (velocity <= right_bound) & (velocity >= left_bound)
        
        f_model_interp = np.interp(velocity[fltr], model_v, model_I)

        invvar = 1. / (normfluxerr[fltr]**2)    
        
        lnp_spec = -0.5 * np.sum((normflux_subtracted[fltr] - f_model_interp)**2 * invvar - np.log(invvar))  # calculating the likelihood for the DTB spectrum
        
        if np.isnan(lnp_spec) == True:
            lnp_spec = -np.inf
            
        # calculating the EW v.s. b profile predicted by the model
        
        EW_b_array = np.zeros(3)
        
        for i in range(len(EW_b_array)):
            factor_array_2d = get_EW_at_b(b_array[i], Ns_array[i], Nx, x_array_1d, alpha, Fv, vinf, gamma, logNcl_0, rand_v)
            
            EW_b_array[i] = lambda13_cm * vth / clight * np.sum((1 - np.prod(factor_array_2d, axis = 0)) * delta_x) / (1e-8)
                
        lnp_EW = -0.5 * np.sum((EW_CII1334_array - EW_b_array)**2 * invvar_b - np.log(invvar_b)) # calculating the likelihood for the EW v.s. b profile
        
        if np.isnan(lnp_EW) == True:
            lnp_EW = -np.inf
        
        return lnp_spec + lnp_EW
    
    if not os.path.exists("LBG_absorption_lines/%s_%s_table"%(name, line)):
        os.makedirs("LBG_absorption_lines/%s_%s_table"%(name, line))
    
    p = np.random.randint(1,999)
    sampler = NestedSampler(loglikelihood, prior_transform, ndim=6, nlive=600) 

    sampler.run_nested(print_progress=True)
    results = sampler.results
    
    samples = results.samples  # samples
    weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

    tfig, taxes = dyplot.traceplot(results)
    
    tfig.savefig("LBG_absorption_lines/%s_%s_table/trace_%s.png"%(name, line, p), dpi=100)
    
    max_logl = np.max(results.logl)
    max_params = samples[np.argmax(results.logl)]

    # Plot the 2-D marginalized posteriors. 
    
    cfig, caxes = dyplot.cornerplot(results, quantiles=[0.16, 0.5, 0.84], show_titles=True, span=6*[0.99], truths = max_params, truth_color='red', truth_kwargs={"linestyle": "dashed"}, labels = [r'$A_{\rm ISM}$', r'$\sigma_{\rm ISM}$', r'${\rm log}\,F_{\rm V}$', r'$\mathcal{V}\,(\rm km\,s^{-1})$', r'$\alpha$', r'$\gamma$'], title_kwargs={"fontsize": 12}, title_fmt = '.2f', label_kwargs=dict(fontsize=15))
    cfig.savefig("LBG_absorption_lines/%s_%s_table/posterior_%s.png"%(name, line, p), dpi=100)

    # Compute 16%-84% quantiles. 
    
    quantiles = [dyfunc.quantile(samps, [0.16, 0.84], weights=weights)
                 for samps in samples.T]

    # Compute weighted mean and covariance.
    
    mean, cov = dyfunc.mean_and_cov(samples, weights)

    text_file = open("LBG_absorption_lines/%s_%s_table/Results_%s.txt"%(name, line, p), "w")
    text_file.write("Max LogLikelihood: %s \n" % max_logl)
    text_file.write("Best-fit Parameters: %s \n" % max_params)
    text_file.write("Quantiles: %s \n" % quantiles)
    text_file.write("Mean: %s \n" % mean)
    text_file.write("Cov: %s \n" % cov)
    text_file.close()

    with open("LBG_absorption_lines/%s_%s_table/results_%s.pkl"%(name, line, p), "wb") as f:
        pickle.dump(results, f)

    np.save("LBG_absorption_lines/%s_%s_table/samples_%s.npy"%(name, line, p), samples)
    np.save("LBG_absorption_lines/%s_%s_table/weights_%s.npy"%(name, line, p), weights)

    return


if __name__ == "__main__":   

    name = 'LBG_zneb'
    line = '1334.5'

    left_bound = -850.0   # velocity ranges where absorption is seen
    right_bound = 250.0
    
    run_nested(left_bound, right_bound, name, line)
