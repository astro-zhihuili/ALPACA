import numpy as np
from constants import kms, lambda13_cm, vth, delta_nu_doppler, a31, pc, G, Msun, clight, nu13, sigma13_factor
from voigt_function import voigt_function_COLT_approx
from radial_velocity_profile_params import M200, c200, rs

Nx = 200

delta_x = 1.0

x_array_1d = np.linspace(-100, 100, Nx)  # array of observed unitless frequencies

def get_EW_at_b(b, Ns, Nx, rmin, rmax, vinf, ncarr, alpha, gamma, rcl_array, Ncl_array, rand_v):

    '''This function takes in a set of physical parameters of the clumps and returns a 2D array for the EWs at a particular impact parameter.
    
    Definition of the parameters:
    
    b: impact parameter
    Ns: number of line segments used at b
    Nx: length of the observed unitless frequency array
    rmin: clump launch radius, or the inner boundary of the CGM halo
    rmax: outer boundary of the CGM halo
    vinf: asymptotic clump outflow velocity
    ncarr: normalized clump number density array
    alpha: power-law index in the clump acceleration force
    gamma: power-law index in the clump number density profile
    rcl_array: clump radius profile
    Ncl_array: clump column density profile
    rand_v: clump velocity dispersion sigma_cl
     
    '''
    
    smin = -np.sqrt(rmax**2 - b**2)

    delta_si = 2 * np.abs(smin) / Ns  # length of each line segment at b

    i_array = np.linspace(1, Ns, Ns)

    s_array_lower_limit = smin + (i_array - 1) * delta_si

    s_array_upper_limit = smin + i_array * delta_si

    s_array_average = (s_array_lower_limit + s_array_upper_limit) / 2  # coordinates along the sightline at b

    r_array = np.sqrt(b**2 + s_array_average**2)

    nc_array = ncarr[0] * (rmin / r_array)**(gamma)

    Ncl_array = Ncl_array[0] * (rmin / r_array)**(0.0)  # assuming constant clump ion column density

    rcl = rcl_array[0]  # assuming constant clump radius
    
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
        factor_array_2d[k] = delta_si * fc_array[k] * np.exp(-Ncl_array[k] * sigma_x_array_2d[k]) + 1 - delta_si * fc_array[k] # 2D array that records the "response" of the clumps moving at varr_parallel to the photons observed at a particular velocity on varr_axis

    return factor_array_2d