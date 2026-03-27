""" helpers.py

Individual helper functions useful for resonator data analysis.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as spopt
from datetime import datetime


def connect_bluefors_logs_to_store(log_path, store):
    """ Connect data from a set of bluefors log files into an HDFStore object in the 'meta' group.

    :param log_path: Path to a directory with all of the bluefors log files to link to the HDFStore object.
    :param store: HDFStore object with resonator data to be linked to bluefors logs.
    """ 
    # - vectorized datetime conversion functions - # 
    to_datetime_meta = np.vectorize(lambda x: datetime.strptime(x, '%Y%m%d_%H%M%S')) 
    to_datetime_temp = np.vectorize(lambda x: datetime.strptime(x, '%d-%m-%y %H:%M:%S'))

    # - make datetime array from meta group, write back to the dataframe - #
    meta = store.meta
    meta['datetime'] = to_datetime_meta(meta.timestamp.values)

    # - construct temperature and datetime dataframe - # 
    files = [f for f in os.listdir(log_path) if f.endswith('log')]
    files.sort()
    temp_df = pd.concat(
        [pd.read_csv(
            os.path.sep.join([log_path, f]),
            names=['date', 'time', 'temperature'] 
        ) for f in files]
    )
    dt_array = np.array([temp_df.date.values, temp_df.time.values]).T
    dt_array = dt_array[:, 0] + " " + dt_array[:, 1]
    dt_array = np.sort(to_datetime_temp(dt_array))
    temp_df = pd.DataFrame({
        'datetime': dt_array,
        'temperature': temp_df.temperature.values,
    })

    # - merge the dataframes, write back to the original store - #
    merged_df = pd.merge_asof(meta, temp_df)
    merged_df.index = meta.index
    store.put('meta', merged_df, format='table')


def csv_to_hdf(csv_path, hdf_path, meta_parameter=None):
    """ Convert a csv file with resonator data into an HDF5 file.

    :param csv_path: String pointing to the csv file to convert.
    :param hdf_path: String file path with the HDF file name to convert into.
    :param meta_parameter: Parameter to include in the meta group and to base the index sweep off of. 
    """
    # - open csv data file - # 
    csv_df = pd.read_csv(csv_path)
    if meta_parameter is not None:
        mp = csv_df[meta_parameter].values
    else:
        mp = None

    # - write to HDF store - # 
    with pd.HDFStore(hdf_path) as store:
        for i, m in enumerate(np.unique(mp)):
            dfm = csv_df.query(f'{meta_parameter} == {m}')
            N = dfm.shape[0]
            data_index = pd.MultiIndex.from_product(
                [['000000'], ['%06i' % i], ['%06i' % j for j in np.arange(N)]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow']
            )
            df = pd.DataFrame({
                'frequency': dfm.frequency.values * 1e9,
                'I': dfm.I.values,
                'Q': dfm.Q.values,
            }, index=data_index)
            meta_index = pd.MultiIndex.from_product(
                [['000000'], ['%06i' % i]],
                names=['RecordGroup', 'RecordGroupInd'],
            )
            meta_df = pd.DataFrame({
                meta_parameter: m
            }, index=meta_index)
            store.append('data', df)
            store.append('meta', meta_df)


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
