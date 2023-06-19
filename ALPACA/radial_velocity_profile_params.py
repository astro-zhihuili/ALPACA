import numpy as np
from constants import kpc, G, Msun

# All quantities used in ALPACA are in cgs units

H0 = 70 * 1e5 / (1e3 * kpc)

z = 3.0

H = H0 * (0.3 * (1 + z)**3 + 0.7)**0.5

M200 = 10**12

c200 = 10**(0.905 - 0.101 * np.log10(M200 / (10**12 / 0.7)))

rvir = (G * M200 * Msun/ 100.0 / H**2) **(1./3.) 

rs = rvir / c200
