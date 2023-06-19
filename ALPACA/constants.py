import numpy as np

# All quantities used in ALPACA are in cgs units

e_ch = 4.80320451 * 1e-10        # electron charge
me = 9.10938356 * 1e-28          # electron mass
clight = 2.99792458 * 1e10
G = 6.67408 * 1e-8
Msun = 1.98848 * 1e33
pc = 3.08567758 * 1e18  
kpc = 3.08567758 * 1e21
kms = 1e5 

# CII 1334.5 transition, data from NIST

lambda13 = 1334.53              # transition wavelength [A]
lambda13_cm = lambda13 / 1e8
A31 = 2.41 * 1e8                  # spontaneous decay [/s]
nu13 = clight / lambda13_cm
f13 = 0.129                      # oscillator strength

sigma13_factor = np.sqrt(np.pi) * e_ch**2 * f13 / me / clight

vth = 15. * kms    # Clump Doppler parameter, b_D
delta_nu_doppler = vth / lambda13_cm
a31 = A31 / (4 * np.pi * delta_nu_doppler)
