#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:20:43 2017

@author: shai.goldman
"""
print 'i exist'
from regression import get_vars_1subj, get_fit_wvars
import misc_funcs
import numpy as np
import time

start_time = time.time()
num_pirts = 4
sample_size = 100
permuting=True
#num_pirts = 3

parameters = ['OP', 'R', 'LAG', 'SIM', 'SESS', 
              'SIMxLAG']
#parameters = ['OP']

for i in range(num_pirts):
    parameters.append('PREVIRT%d' % (i+1))

pvals = [[] for i in range(len(parameters))]
fstats = [[] for i in range(len(parameters))]

permed_fstats = [[[] for i in range(len(parameters))] for i in range(sample_size)]
permed_pvals = [[[] for i in range(len(parameters))] for i in range(sample_size)]
files = misc_funcs.get_ltpFR2_filenames()
for fname in files:
    print fname, files.index(fname)
    try:
        vs = get_vars_1subj(fname, num_pirts=num_pirts)
    except IndexError:
        continue
    if len(vs['irts']) < 100:
        continue
    
    #get actual fstat
    mainfit = get_fit_wvars(vs, exclude=[])
    fits = [get_fit_wvars(vs, exclude=[i]) for i in parameters]
    for i in range(len(pvals)):
        fstat, pval, idk = mainfit.compare_f_test(fits[i])
        pvals[i].append(pval)
        fstats[i].append(fstat)
    print 'nonperm done\nstarting perms:'
    #get permeated fstats
    for perm in range(sample_size):
        
        print perm,
        pvs = vs.copy()
        
        if permuting:
            pvs['irts'] = np.random.permutation(pvs['irts'])
        else:
            args = np.random.choice(a=pvs['irts'].size, size=pvs['irts'].size, replace=True)
            for i in pvs:
                if i == 'previrts':
                    for j in pvs[i]:
                        pvs[i][j] = pvs[i][j][args]
                elif i == 'lambda':
                    continue
                elif pvs[i].shape == args.shape:
                    pvs[i] = pvs[i][args]
        mainfit = get_fit_wvars(pvs, exclude=[])
        fits = [get_fit_wvars(pvs, exclude=[i]) for i in parameters]
        for i in range(len(pvals)):
            fstat, pval, idk = mainfit.compare_f_test(fits[i])
            permed_pvals[perm][i].append(pval)
            permed_fstats[perm][i].append(fstat)
    print ''
    print("Subj run time: " + str(time.time() - start_time))
fstats = np.asarray(fstats)
pvals = np.asarray(pvals)

permed_fstats = np.asarray(permed_fstats)
permed_pvals = np.asarray(permed_pvals)
seed = np.random.randint(1000)
pref = 'test2'
misc_funcs.writeArrayFile('%s_permed_fstats%d.txt' % (pref, seed), permed_fstats)
misc_funcs.writeArrayFile('%s_permed_pvals%d.txt' % (pref, seed), permed_pvals)
np.savetxt('%s_pvals%d.txt' % (pref, seed), pvals)
np.savetxt('%s_fstats%d.txt' % (pref, seed), fstats)

print("Full run time: " + str(time.time() - start_time))