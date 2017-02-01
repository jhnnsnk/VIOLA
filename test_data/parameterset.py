#!/usr/bin/env python
'''parameters for network simulation'''
import os
from NeuroTools import parameters as ps
import numpy as np
from NeuroTools.parameters import ParameterSet
import hashlib
import pickle
import operator

PSET_DEFAULTS = ps.ParameterSet(dict(
    dt        = 0.1,    # Simulation time resolution in ms
    simtime   = 1000.,  # Simulation time in ms
    transient = 200., # Simulation transient, discarding spikes at times < transient 
    
    g       = 4.5,  # ratio inhibitory weight/excitatory weight (before: 5.0)
    eta     = 2.0,  # external rate relative to threshold rate
    epsilon = 0.1,  # connection probability (before: 0.1)
    
    order = 10000,    # network scaling factor 
    
    tauSyn = 0.5,    # synaptic time constant in ms
    tauMem = 20.0,   # time constant of membrane potential in ms
    CMem   = 250.0,  # capacitance of membrane in in pF
    theta  = 20.0,   # membrane threshold potential in mV
    J      = 0.5,    # postsyaptic amplitude in mV (before: 0.1)
    
    extent_length = 4.,   # in mm (layer size = extent_length x extent_length)
    sigma_ex = 0.3,       # width of Gaussian profile of excitatory connections
    sigma_in = 0.3,       # sigma in mm

    c_EX = .3,      # constant term for linear distance-dependent delay,
    a_EX = 0.7,      # propagation velocity for linear delay param, p(d)=c+a*d

))

# define parameterspace to iterate over
PS = ps.ParameterSpace(PSET_DEFAULTS)
PS.update(dict(
    g = ps.ParameterRange(np.linspace(4., 5., 5)),
    sigma_ex = ps.ParameterRange((np.linspace(0.2, 0.4, 5))),
    # eta = ps.ParameterRange(np.linspace(0., 2., 5)),
    J = ps.ParameterRange(np.linspace(0.1, 1.1, 5)),    
))


def sort_deep_dict(d):
    '''
    sort arbitrarily deep dictionaries into tuples
    
    Arguments
    ---------
    d : dict
    
    Returns:
    x : list of tuples of tuples of tuples ...
    '''
    x = sorted(iter(d.items()), key=operator.itemgetter(0))
    for i, (key, value) in enumerate(x):
        if type(value) == dict or type(value) == ps.ParameterSet:
            y = sorted(iter(value.items()), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if type(v) == dict or type(v) == ps.ParameterSet:
                    y[j] = (k, sort_deep_dict(v))
    return x

def get_unique_id(paramset):
    '''
    create a unique hash key for input dictionary
    
    Arguments
    ---------
    paramset : dict
        parameter dictionary
    
    Returns
    -------
    key : str
        hash key
    
    '''
    sorted_params = sort_deep_dict(paramset)
    string = pickle.dumps(sorted_params)
    key = hashlib.md5(string).hexdigest()
    return key




if __name__ == '__main__':
    # Parameterset container
    PSETdict = {}
    
    # destination of input and output files
    parameterset_dest = 'parameters'
    jobscript_dest = 'jobs'
    log_dir = 'logs'
    output_raw_prefix = 'out_raw'
    output_proc_prefix = 'out_proc'
    
    for paramset in PS.iter_inner():
        # unique id for each parameter set, constructed from the parameset dict
        # converted to a sorted list of tuples
        ps_id = get_unique_id(paramset)
        
        if ps_id in list(PSETdict.keys()):
            print('skipping {0}, already in job list'.format(ps_id))
            pass
        else:
            print(ps_id)
            
            # unique output path for each parameter set
            spike_output_path = os.path.join(output_raw_prefix, ps_id)
            output_path_proc = os.path.join(output_proc_prefix, ps_id)
            # set up file destinations
            if not os.path.isdir(parameterset_dest):
                os.mkdir(parameterset_dest)
            if not os.path.isdir(jobscript_dest):
                os.mkdir(jobscript_dest)
            if not os.path.isdir(output_raw_prefix):
                os.mkdir(output_raw_prefix)
            if not os.path.isdir(spike_output_path):
                os.mkdir(spike_output_path)
            if not os.path.isdir(output_proc_prefix):
                os.mkdir(output_proc_prefix)
            if not os.path.isdir(output_path_proc):
                os.mkdir(output_path_proc)
            
                    
            # put output_path into dictionary, as we now have a unique ID of
            # though this will not affect the parameter space object PS
            paramset = paramset.copy()
            paramset.update({
                'ps_id' : ps_id,
                'spike_output_path' : spike_output_path,
                })

            #put parameter set into dict using ps_id as key
            PSETdict[ps_id] = paramset

            #write using ps.ParemeterSet native format
            parameter_set_file = os.path.join(parameterset_dest, '{}.pset'.format(ps_id))
            ps.ParameterSet(PSETdict[ps_id]).save(url=parameter_set_file)
            
            #specify where to save output and errors
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)
            output_network = os.path.join(os.getcwd(), log_dir, ps_id+'.txt')



            jobscript_skeleton = '''#!/bin/bash
#SBATCH --job-name %s
#SBATCH --time %s
#SBATCH -e %s
#SBATCH -o %s
#SBATCH -N %i
#SBATCH --cpus-per-task %i
#SBATCH --exclusive
python topo_brunel_alpha_nest.py %s
python nest_preprocessing.py %s %s
python fake_LFP_signal.py %s %s
'''

            # set up jobscript
            wt = '0:15:00'
            lnodes = 1
            ppn = 48
            jobscript = jobscript_skeleton % (ps_id,
                                            wt,
                                            output_network,
                                            output_network,
                                            lnodes,
                                            ppn,
                                            parameter_set_file,
                                            spike_output_path,
                                            output_path_proc,
                                            spike_output_path, output_path_proc)

            # write jobscript
            f = open(os.path.join(jobscript_dest, '%s.job' % ps_id), 'w')
            f.write(jobscript)
            f.close()
            
            # submit jobscript
            os.system('sbatch {}'.format(os.path.join(jobscript_dest, '%s.job' % ps_id)))
            
            # os.system(jobscript)
            

