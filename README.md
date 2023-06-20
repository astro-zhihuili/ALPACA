# ALPACA: Absorption Line Profiles Arising from Clumpy Absorbers
<img src="Alpaca.jpg" height="315">

ALPACA is a fast and easy to use semi-analytic forward model for metal absorption lines emerging from a clumpy galactic environment.

Usage
-------

Here is a brief introduction to the usage of the key functions in ALPACA:

get_EW_at_b(): This function takes in a set of physical parameters of the clumps and returns a 2D array that records the "response" of the moving clumps along a sightline at impact parameter b to the photons observed at a particular velocity. The 2D array can be further converted into an EW of absorption observed at impact parameter b.

get_abs_profile(): This function takes in a set of physical parameters of the clumps and returns a 2D array that records the "response" of the moving clumps along the "down-the-barrel" sightline to the photons observed at a particular velocity. The 2D array can be further converted into a normalized absorption line profile. 

get_absorption_line_profile(): This function extracts an observed absorption line profile for fitting. The format for the data file should be: [velocity (km/s), normalized flux density, normalized flux density uncertainty]. 

run_nested(): This function performs a joint fitting of down-the-barrel absorption line profile and a EW v.s. b profile. A fitting run can be started via: python3 joint_fitting.py

Citations:
-------

If you find ALPACA useful in your research, please cite Li et al. (2023).

License 
-------
ALPACA is publicly available under the MIT license.

Copyright 2023 Zhihui Li and contributors.
