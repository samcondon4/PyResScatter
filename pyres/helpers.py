""" helpers.py

Individual helper functions useful for resonator data analysis.
"""
import numpy as np
from scipy import stats
import scipy.optimize as spopt


def connect_bluefors_logs_to_store(log_path, store):
    """ Connect data from a set of bluefors log files into an HDFStore object in the 'meta' group.

    :param log_path: Path to a directory with all of the bluefors log files to link to the HDFStore object.
    :param store: HDFStore object with resonator data to be linked to bluefors logs.
    """ 
    pass


def csv_to_hdf(csv_path, hdf_path):
    """ Convert a csv file with resonator data into an HDF5 file.
:param csv_path: String pointing to the csv file to convert.
    :param hdf_path: String file path with the HDF file name to convert into.
    """
    pass


def circle_fit(sdata):
    """ Fit a circle to complex scattering data. Adapted from `Probst et. al. resonator_tools <https://github.com/sebastianprobst/resonator_tools>`_. 
    See `Probst et. al. 2015 <https://pubs.aip.org/aip/rsi/article/86/2/024706/360955>`_ for a detailed description of the algebraic circle fit technique
    implemented below.

    :param sdata: Array of complex numbers containing the scattering data to fit. 
    """
    xi = sdata.real
    xi_sqr = xi*xi
    yi = sdata.imag
    yi_sqr = yi*yi
    zi = xi_sqr+yi_sqr
    Nd = float(len(xi))
    xi_sum = xi.sum()
    yi_sum = yi.sum()
    zi_sum = zi.sum()
    xiyi_sum = (xi*yi).sum()
    xizi_sum = (xi*zi).sum()
    yizi_sum = (yi*zi).sum()
    
    M = np.array([ [(zi*zi).sum(), xizi_sum, yizi_sum, zi_sum],  \
    [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum], \
    [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum], \
    [zi_sum, xi_sum, yi_sum, Nd] ])
        
    a0 = ((M[2][0]*M[3][2]-M[2][2]*M[3][0])*M[1][1]-M[1][2]*M[2][0]*M[3][1]-M[1][0]*M[2][1]*M[3][2]+M[1][0]*M[2][2]*M[3][1]+M[1][2]*M[2][1]*M[3][0])*M[0][3]+(M[0][2]*M[2][3]*M[3][0]-M[0][2]*M[2][0]*M[3][3]+M[0][0]*M[2][2]*M[3][3]-M[0][0]*M[2][3]*M[3][2])*M[1][1]+(M[0][1]*M[1][3]*M[3][0]-M[0][1]*M[1][0]*M[3][3]-M[0][0]*M[1][3]*M[3][1])*M[2][2]+(-M[0][1]*M[1][2]*M[2][3]-M[0][2]*M[1][3]*M[2][1])*M[3][0]+((M[2][3]*M[3][1]-M[2][1]*M[3][3])*M[1][2]+M[2][1]*M[3][2]*M[1][3])*M[0][0]+(M[1][0]*M[2][3]*M[3][2]+M[2][0]*(M[1][2]*M[3][3]-M[1][3]*M[3][2]))*M[0][1]+((M[2][1]*M[3][3]-M[2][3]*M[3][1])*M[1][0]+M[1][3]*M[2][0]*M[3][1])*M[0][2]
    a1 = (((M[3][0]-2.*M[2][2])*M[1][1]-M[1][0]*M[3][1]+M[2][2]*M[3][0]+2.*M[1][2]*M[2][1]-M[2][0]*M[3][2])*M[0][3]+(2.*M[2][0]*M[3][2]-M[0][0]*M[3][3]-2.*M[2][2]*M[3][0]+2.*M[0][2]*M[2][3])*M[1][1]+(-M[0][0]*M[3][3]+2.*M[0][1]*M[1][3]+2.*M[1][0]*M[3][1])*M[2][2]+(-M[0][1]*M[1][3]+2.*M[1][2]*M[2][1]-M[0][2]*M[2][3])*M[3][0]+(M[1][3]*M[3][1]+M[2][3]*M[3][2])*M[0][0]+(M[1][0]*M[3][3]-2.*M[1][2]*M[2][3])*M[0][1]+(M[2][0]*M[3][3]-2.*M[1][3]*M[2][1])*M[0][2]-2.*M[1][2]*M[2][0]*M[3][1]-2.*M[1][0]*M[2][1]*M[3][2])
    a2 = ((2.*M[1][1]-M[3][0]+2.*M[2][2])*M[0][3]+(2.*M[3][0]-4.*M[2][2])*M[1][1]-2.*M[2][0]*M[3][2]+2.*M[2][2]*M[3][0]+M[0][0]*M[3][3]+4.*M[1][2]*M[2][1]-2.*M[0][1]*M[1][3]-2.*M[1][0]*M[3][1]-2.*M[0][2]*M[2][3])
    a3 = (-2.*M[3][0]+4.*M[1][1]+4.*M[2][2]-2.*M[0][3])
    a4 = -4.

    func = lambda x: a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x 
    d_func = lambda x: a1+2*a2*x+3*a3*x*x+4*a4*x*x*x  
    x0 = spopt.fsolve(func, 0., fprime=d_func)
            
    M[3][0] = M[3][0]+2*x0[0]
    M[0][3] = M[0][3]+2*x0[0]
    M[1][1] = M[1][1]-x0[0]
    M[2][2] = M[2][2]-x0[0]

    U, s, Vt = np.linalg.svd(M)
    
    A_vec = Vt[np.argmin(s),:]

    xc = -A_vec[1]/(2.*A_vec[0])
    yc = -A_vec[2]/(2.*A_vec[0])
    r0 = 1./(2.*np.absolute(A_vec[0]))*np.sqrt(A_vec[1]*A_vec[1]+A_vec[2]*A_vec[2]-4.*A_vec[0]*A_vec[3])

    return xc, yc, r0
