import numpy as np
from constants import kms, lambda13_cm, vth, delta_nu_doppler, a31, pc, G, Msun, clight, nu13, sigma13_factor
from voigt_function import voigt_function_COLT_approx
from radial_velocity_profile_params import M200, c200, rs

def get_EW_at_b(b, Ns, Nx, x_array_1d, alpha, Fv, vinf, gamma, logNcl_0, rand_v, 
                nbins = 9900, rcl = 100 * pc, rmin = 1000 * pc, rmax = 100000 * pc):

    '''This function takes in a set of physical parameters of the clumps and returns a 2D array for the EWs at a particular impact parameter.
    
    Definition of the parameters:
    
    b: impact parameter
    Ns: number of line segments used at b
    Nx: length of the observed unitless frequency array
    x_array_1d: array of observed unitless frequencies
    alpha: power-law index in the clump acceleration force
    Fv: clump volume filling factor
    vinf: asymptotic clump outflow velocity
    gamma: power-law index in the clump number density profile
    logNcl_0: clump ion column density at r = rmin
    rand_v: clump velocity dispersion sigma_cl
    nbins: number of shells
    rcl: clump radius
    rmin: clump launch radius, or the inner boundary of the CGM halo
    rmax: outer boundary of the CGM halo
     
    '''
    
    lmin = -np.sqrt(rmax**2 - b**2)

    delta_l = 2 * np.abs(lmin) / Ns  # length of each line segment at b

    i_array = np.linspace(1, Ns, Ns)

    s_array_lower_limit = lmin + (i_array - 1) * delta_l

    s_array_upper_limit = lmin + i_array * delta_l

    s_array_average = (s_array_lower_limit + s_array_upper_limit) / 2  # coordinates along the sightline at b

    r_array = np.sqrt(b**2 + s_array_average**2)
    
    ncarr = np.ones(nbins)
    
    rarr = np.linspace(rmin, rmax, nbins)
    
    ncarr = ncarr * (rmin / rarr)**(gamma)  
    
    vol = 4/3. * np.pi * (rmax**3 - rmin**3)
    
    vol_cl = 4/3. * np.pi * rcl**3
    
    dr = rarr[1] - rarr[0]

    Vcl_total = np.sum(ncarr * 4 * np.pi * rarr**2 * dr * vol_cl)

    nc_norm = vol * Fv / Vcl_total
    
    ncarr = ncarr * nc_norm # clump number density profile

    nc_array = ncarr[0] * (rmin / r_array)**(gamma)

    Ncl_array = (10**logNcl_0) * (rmin / r_array)**(0.0)  # assuming constant clump ion column density
    
    fc_array = nc_array * np.pi * rcl**2  # evaluate clump covering factors along the sightline at b

    h_array_b_2d = np.zeros((Ns, Nx))
    sigma_x_array_2d = np.zeros((Ns, Nx))
    factor_array_2d = np.zeros((Ns, Nx))

    cos_theta = s_array_average / r_array

    vcl_r_array = np.zeros(len(r_array))
    vcl_r_array_sqr = 2 * G * M200 * Msun / (np.log(1 + c200) - c200 / (1 + c200)) * (np.log(1 + r_array / rs) / r_array - np.log(1 + rmin / rs) / rmin) + vinf**2 * (1 - (r_array / rmin)**(1 - alpha))

    fltr_v_b = (vcl_r_array_sqr > 0)
    vcl_r_array[fltr_v_b] = np.sqrt(vcl_r_array_sqr[fltr_v_b])

    np.random.seed(8)
    sigma_rand_array = np.random.normal(0, rand_v, len(r_array)) * kms

    x_array_2d = np.zeros((Ns, Nx))

    for k in range(Ns):
        x_array_2d[k] = x_array_1d - ((vcl_r_array[k] + sigma_rand_array[k]) / vth * cos_theta[k])
        h_array_b_2d[k] = voigt_function_COLT_approx(x_array_2d[k], a31) 
        sigma_x_array_2d[k] = sigma13_factor / delta_nu_doppler * h_array_b_2d[k]
        factor_array_2d[k] = delta_l * fc_array[k] * np.exp(-Ncl_array[k] * sigma_x_array_2d[k]) + 1 - delta_l * fc_array[k] # 2D array that records the "response" of the clumps moving at varr_parallel to the photons observed at a particular velocity on varr_axis

    return factor_array_2d