import numpy as np
from constants import kms, lambda13_cm, vth, delta_nu_doppler, a31, pc, G, Msun, clight, nu13, sigma13_factor
from voigt_function import voigt_function_COLT_approx
from get_EW_at_b import get_EW_at_b, Nx
from radial_velocity_profile_params import M200, c200, rs

varr_axis = np.linspace(-1500*kms, 1500*kms, 301)    # axis for the observed velocity

b_array = np.array([33, 66, 99]) * 1000 * pc  # three impact parameters where EW is evaluated

Ns_array = np.array([5000, 3000, 1000])   # number of line segments used for three impact parameters

def get_abs_profile_and_EW(alpha, Fv, vinf, gamma, logNcl_0, rand_v, varr_axis, 
                           nbins = 9900, rcl = 100 * pc, rmin = 1000 * pc, rmax = 100000 * pc):

    '''This function takes in a set of physical parameters of the clumps and returns four 2D arrays for the DTB absorption spectrum and EWs at three b > 0 impact parameters.
    
    Definition of the parameters:
    
    alpha: power-law index in the clump acceleration force
    Fv: clump volume filling factor
    vinf: asymptotic clump outflow velocity
    gamma: power-law index in the clump number density profile
    logNcl_0: clump ion column density at r = rmin
    rand_v: clump velocity dispersion sigma_cl
    varr_axis: array of the observed velocity axis 
    nbins: number of shells
    rcl: clump radius
    rmin: clump launch radius, or the inner boundary of the CGM halo
    rmax: outer boundary of the CGM halo     
    '''
    
    rarr = np.linspace(rmin, rmax, nbins)
    varr = np.zeros(len(rarr))
    
    varr_sqr = 2 * G * M200 * Msun / (np.log(1 + c200) - c200 / (1 + c200)) * (np.log(1 + rarr / rs) / rarr - np.log(1 + rmin / rs) / rmin) + vinf**2 * (1 - (rarr / rmin)**(1 - alpha))
    
    fltr_v = (varr_sqr > 0)
    varr[fltr_v] = np.sqrt(varr_sqr[fltr_v]) # clump radial outflow velocity
    
    ncarr = np.ones_like(varr) 
    
    rclarr = rcl * (rarr / rmin)**(0.0)  # clump radius (assuming constant)
    
    ncarr = ncarr * (rmin / rarr)**(gamma)  
    
    vol = 4/3. * np.pi * (rmax**3 - rmin**3)
    
    vol_cl_arr = 4/3. * np.pi * rclarr**3
    
    dr = rarr[1] - rarr[0]

    Vcl_total = np.sum(ncarr * 4 * np.pi * rarr**2 * dr * vol_cl_arr)

    nc_norm = vol * Fv / Vcl_total
    
    ncarr = ncarr * nc_norm   # clump number density profile
    
    d_shell = (rmax - rmin) / nbins
    
    fc_r = np.pi * ncarr * (rclarr**2 * d_shell - d_shell**3 / 12.) # clump covering fraction profile
        
    Ncl_array = (10**logNcl_0) * (rmin / rarr)**(0.0)   # clump ion column density (assuming constant)
    
    length_varr = len(varr)
    length_varr_axis = len(varr_axis)   
        
    tau_v = np.zeros((length_varr_axis, length_varr))
    prob_arr_2d = np.zeros((length_varr_axis, length_varr))
                
    np.random.seed(8)
    sigma_rand_array = np.random.normal(0, rand_v, length_varr) # clump radial random motion array
    
    delta_v_array = np.ones(length_varr)
    nu_array = np.ones(length_varr)
    x_array = np.ones(length_varr)
    h_array = np.ones(length_varr)
    sigma_v_j_array = np.ones(length_varr)
    
    for j in range(length_varr_axis):
        delta_v_array = varr_axis[j] - varr + sigma_rand_array * kms
        nu_array = clight / (lambda13_cm * (1 + delta_v_array / (3 * 1e5 * kms)))
        x_array = (nu_array - nu13) / delta_nu_doppler
        
        h_array = voigt_function_COLT_approx(x_array, a31) 
        sigma_v_j_array = sigma13_factor / delta_nu_doppler * h_array
        tau_v[j] = Ncl_array * sigma_v_j_array
        prob_arr_2d[j] = 1 - fc_r + fc_r * np.exp(-tau_v[j]) # 2D array that records the "response" of the clumps moving at varr to the photons observed at a particular velocity on varr_axis
        
    # Getting three more 2D arrays for three impact parameters at b > 0
    
    factor_array_2d_b1 = get_EW_at_b(b_array[0], Ns_array[0], Nx, rmin, rmax, vinf, ncarr, alpha, gamma, rclarr, Ncl_array, rand_v)
    
    factor_array_2d_b2 = get_EW_at_b(b_array[1], Ns_array[1], Nx, rmin, rmax, vinf, ncarr, alpha, gamma, rclarr, Ncl_array, rand_v)
    
    factor_array_2d_b3 = get_EW_at_b(b_array[2], Ns_array[2], Nx, rmin, rmax, vinf, ncarr, alpha, gamma, rclarr, Ncl_array, rand_v)
    
    return np.flip(-varr_axis / kms), prob_arr_2d, factor_array_2d_b1, factor_array_2d_b2, factor_array_2d_b3