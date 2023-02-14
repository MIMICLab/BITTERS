import pywt
import numpy as np

def extract_coeff(img):
    """
    Returns RGB dwt applied coefficients tuple
    Parameters
    ----------
    img: PIL Image
    Returns
    -------
    (coeffs_r, coeffs_g, coeffs_b):
        RGB coefficients with Discrete Wavelet Transform Applied
    """
    img = np.asarray(img)
    mat_r, mat_g, mat_b = img[:,:,0], img[:,:,1], img[:,:,2]

    coeffs_r = pywt.dwt2(mat_r, 'db4')
   
    coeffs_g = pywt.dwt2(mat_g, 'db4')
    
    coeffs_b = pywt.dwt2(mat_b, 'db4')

    dwt_r = coeffs_r[0]
    dwt_g = coeffs_g[0]
    dwt_b = coeffs_b[0]

    return dwt_r, dwt_g, dwt_b
