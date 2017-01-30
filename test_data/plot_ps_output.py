#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NeuroTools.parameters import ParameterSet, ParameterSpace
import parameterset
import nest_preprocessing as npp
import os
import h5py
import scipy.sparse as sp
from matplotlib.animation import FuncAnimation
from copy import copy


def get_network_analysis_object(parameter_set_file, ps_id,
                                TRANSIENT=0.,
                                BINSIZE_TIME=1.0,
                                BINSIZE_AREA=0.1):
    '''
    Return the meso_analysis.NetworkAnalysis object corresponding to a
    parameterset hash id
    
    Implemented as a mean to avoid defining parameters such as TRANSIENT,
    BINSIZE_TIME, BINSIZE_AREA in more than one place.
    
    Parameters
    ----------
    parameter_set_file : str
        path to parameterset sli file
    ps_id : str
        unique id of parameterset
    
    Returns
    -------
    object : meso_analysis.analysis.NetworkAnalysis
    '''
    #set up the analysis class instance, it is now a daughter class of
    #NetworkAnalysisParams, and will pass kwargs to parent class
    analysis = npp.ViolaPreprocessing(
                 input_path=os.path.join('out_raw', ps_id),
                 output_path=os.path.join('out_proc', ps_id),
                 TRANSIENT=TRANSIENT,
                 BINSIZE_TIME=BINSIZE_TIME,
                 BINSIZE_AREA=BINSIZE_AREA,
    )
    
    return analysis


def load_analysis_dict(subParamSpace):
    '''
    We need to look at a bundle of analysis objects at the time

    kwargs:
    ::
        base_parameters : dict, default parameterset
        subspace : dict, subspace of full parameterspace
    '''
    #iterate over parameter subspace
    ANALYSIS = {}
    for paramset in subParamSpace.iter_inner():
        ps_id = parameterset.get_unique_id(paramset)

        print('loading dataset %s' % ps_id)

        ANALYSIS[ps_id] = get_network_analysis_object(os.path.join('parameters', ps_id + '.sli'),
                                                      ps_id,
                                                      BINSIZE_AREA=0.1)
        ANALYSIS[ps_id].GIDs_corrected = ANALYSIS[ps_id].get_GIDs()
        ANALYSIS[ps_id].positions_corrected = ANALYSIS[ps_id].get_positions()
 
    return ANALYSIS


def print_subspace_ID_overview(ANALYSIS, subParamSpace, subspace,
                               dim0str, dim0, dim1, dim2,
                               keys=['PSP_e', 'g'], titlekey='conn_profile',
                               fname='ps_overview.txt'):
    '''
    Print text overview of parameter combination IDs to
    file and screen
    '''
    f = file(fname, 'w')

    #print header
    f.write('Parameter ID overview: {0} {1}, {2} vs. {3}:\n'.format(
                            dim0str, dim0, dim1, dim2))

    #iterate over each variable to get nice ordered layout
    for i, value_i in enumerate(subParamSpace[keys[0]]):
        for j, value_j in enumerate(subParamSpace[keys[1]]):
            # no need to copy from base_parameters,
            # ParameterRange s will be overwritten
            paramset = subParamSpace.copy()
            paramset[keys[0]] = value_i
            paramset[keys[1]] = value_j
            paramset[titlekey] = subspace[titlekey]
            ps_id = parameterset.get_unique_id(paramset)
            
            f.write(ps_id[:6] + '\t'),
        f.write('\n')

    f.close()

    print('\n\nFile {0}\n'.format(fname))
    f = file(fname, 'r')
    for l in f.readlines():
        print(l)
    f.close()

def load_h5_to_sparse(X, f):
    '''load sparse matrix stored on COOrdinate format from HDF5.
    
    Arguments
    ---------        
    X : str
        group name, group must contain datasets:
            'data', 'row', 'col' vectors of equal length
            'shape' : shape of array tuple
    f : file
        <HDF5 file "filename.h5" (mode r+)>
    
    Returns
    -------
    data : scipy.sparse.csr.csr_matrix
        
    '''
    data = sp.coo_matrix((f[X]['data_row_col'].value[:, 0],
                          (f[X]['data_row_col'].value[:, 1],
                           f[X]['data_row_col'].value[:, 2])),
                         shape=f[X]['shape'].value) 

    return data.tocsr()



if __name__ == '__main__':
    #base_parameters from file
    PSET_DEFAULTS = parameterset.PSET_DEFAULTS

    plt.close('all')

    #time window time series
    T = (200, 400)

    variables = [['sigma_ex', 'g', 'eta']] # dim 1 and 2 iterable,
                                           # dim 3 is fixed or iterable

    for dim1, dim2, dim3 in variables:
        # top level iterable is dim3
        for value_h in getattr(parameterset.PS, dim3): 
             # define subparameterspace as PS can be N-dimensional
             subspace = {
                 dim1 : getattr(parameterset.PS, dim1),
                 dim2 : getattr(parameterset.PS, dim2),
                 dim3 : value_h
             }
             subParamSpace = ParameterSpace(PSET_DEFAULTS.copy())
             subParamSpace.update(subspace)
             
             
             #load analysis objects
             ANALYSIS = load_analysis_dict(subParamSpace)
     
             #print out parameter space ids
             fname = 'ps_output_{}_vs_{}_vs_{}_{}.txt'.format(
                          dim1, dim2, dim3, value_h)
             print_subspace_ID_overview(ANALYSIS, subParamSpace, subspace,
                                        dim3, subParamSpace.epsilon,
                                        dim1, dim2,
                                        keys = [dim1, dim2],
                                        titlekey=dim3,
                                        fname=fname,
                                     )
     
     
             nrows = len(getattr(subParamSpace, dim1))
             ncols = len(getattr(subParamSpace, dim2))
     
             fig, axes = plt.subplots(nrows, ncols, figsize=(nrows*3, ncols*3))
             fig.subplots_adjust(hspace=0.01, wspace=0.01)
             
                 
             TSTEP = 0
             datas = []
             images = []
             
             for i, value_i in enumerate(subParamSpace[dim1]):
                 for j, value_j in enumerate(subParamSpace[dim2]):
     
                     paramset = PSET_DEFAULTS.copy()
                     paramset[dim1] = value_i
                     paramset[dim2] = value_j
                     paramset[dim3] = subspace[dim3]
                     ps_id = parameterset.get_unique_id(paramset)
                     print(ps_id)
     
                     A = ANALYSIS[ps_id]
                     
                     X = 'EX'
                     f = h5py.File(os.path.join(A.output_path, 'all_binned_sprates_rs.h5'))
                     data = load_h5_to_sparse(X, f).todense()
                     data = np.array(data).reshape((A.pos_bins.size-1, -1, data.shape[1]))
                     datas.append(data)
                     f.close()
                     
                     im = axes[i, j].pcolormesh(A.pos_bins, A.pos_bins,
                                        data[:, :, TSTEP],
                                        vmin=0., vmax=500., cmap='gray')
                     
                     axes[i, j].set_xticklabels([])
                     axes[i, j].set_yticklabels([])
                     axes[i, j].axis(axes[i, j].axis('tight'))
                     # axes[i, j].set_title('{}={}, {}={}'.format(dim1, value_i, dim2, value_j))
                     if j == 0:
                         axes[i, j].set_ylabel('{}={}'.format(dim1, value_i))
                     if i == 0:
                         axes[i, j].set_title('{}={}'.format(dim2, value_j))
     
                     
                     images.append(im)
     
             dt = A.BINSIZE_TIME
             tbins = np.arange(T[0]/dt, (T[1]+dt)/dt).astype(int)
             
     
             def init():
                 return
             
             def update(frame_number):
                 '''update function for animation'''
                 TSTEP = frame_number % (tbins.size-1) + int(T[0] / dt)
                 fig.suptitle('{}={}, '.format(dim3, value_h) + 't=%.3i ms' % int(tbins[frame_number % (tbins.size-1)]*dt))
                 for i, (data, im) in enumerate(zip(datas, images)):
                     im.set_array(data[:, :, TSTEP].flatten())
         
             ani = FuncAnimation(fig=fig, func=update, init_func=init,
                                 frames=tbins.size, interval=50)
             ani.save('ps_output_{}_vs_{}_vs_{}_{}.mp4'.format(
                          dim1, dim2, dim3, value_h),
                      fps=10, writer='ffmpeg',
                      extra_args=['-b:v', '20000k', '-r', '10', '-vcodec', 'mpeg4'],)
     
             plt.close(fig)
        # plt.show()