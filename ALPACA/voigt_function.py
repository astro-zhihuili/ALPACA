import numpy as np

# Coefficients used in the Voigt function approximation

A0 = 15.75328153963877
A1 = 286.9341762324778
A2 = 19.05706700907019
A3 = 28.22644017233441
A4 = 9.526399802414186
A5 = 35.29217026286130
A6 = 0.8681020834678775
B0 = 0.0003300469163682737
B1 = 0.5403095364583999
B2 = 2.676724102580895
B3 = 12.82026082606220
B4 = 3.21166435627278
B5 = 32.032981933420
B6 = 9.0328158696
B7 = 23.7489999060
B8 = 1.82106170570


def voigt_function_COLT_approx(xarray, a):
        
        '''This function takes in an array of unitless frequencies x and the normalized natural line width a and returns an array of Voigt function values. The approximation for the Voigt approximation comes from Smith et al. (2015).'''
        
        zarray = xarray**2

        output_array = np.zeros(xarray.shape)

        fltr1 = (zarray <= 3.0)
        fltr2 = ((zarray > 3.0) & (zarray < 25.0))
        fltr3 = (zarray >= 25.0)

        output_array[fltr1] = np.exp(-zarray[fltr1]) * (1.0 - a * (A0 + A1/(zarray[fltr1] - A2 + A3/(zarray[fltr1] - A4 + A5/(zarray[fltr1] - A6)))))
        output_array[fltr2] = np.exp(-zarray[fltr2]) + a * (B0 + B1/(zarray[fltr2] - B2 + B3/(zarray[fltr2] + B4 + B5/(zarray[fltr2] - B6 + B7/(zarray[fltr2] - B8)))))
        output_array[fltr3] = a / np.sqrt(np.pi) / (zarray[fltr3] - 1.5 - 1.5 / (zarray[fltr3] - 3.5 - 5.0 / (zarray[fltr3] - 5.5)))

        return output_array