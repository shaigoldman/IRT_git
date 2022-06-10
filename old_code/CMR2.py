#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:48:45 2019

@author: lumdusislife
"""

import CMR2_pack_cyth as cmr
import numpy as np
import os
import scipy.io as sio

cmrpath = '/Users/lumdusislife/Desktop/IRT/pyCMR2/CMR2_Optimized/'
datapath = '/Users/lumdusislife/Desktop/IRT/Scripts/Resources/'
lynnpath = datapath+'/LohnEtal14.model/K02_param.mat'

os.chdir(datapath+"..")
import data_collection as dc

params = {'alpha': 0.679,
         'beta_enc': 0.520,
         'beta_rec': 0.628,
         'beta_rec_post': 0.803,
         'c_thresh': 0.074,
         'dt': 10,
         'eta': 0.393,
         'gamma_cf': 0.895,
         'gamma_fc': 0.425,
         'kappa': 0.313,
         'lamb': 0.130,
         'learn_while_retrieving': False,
         'max_recalls': 50,
         'nitems_in_accumulator': 50,
         'omega': 11.9,
         'phi_d': 0.990,
         'phi_s': 1.41,
         'rec_time_limit': 60000.0,
         's_cf': 1.29, #not sure whats the deal with "cf" and "fc"
         's_fc': 0
         #,'beta_distract': 0
         }

data = np.load(datapath+'data.npy', allow_pickle=True).tolist()
#data = dc.truck_alldata(data, lists=2000)

w2v = np.loadtxt(cmrpath+'wordpools/PEERS_w2v.txt').tolist()

mode = 'IFR' #check this if we want DFR instead maybe
identifiers = data['session']#data['subject']


mdata = data.copy()
mdata['pres_itemnos'] = data['pres_itemnos']

mdata['rec_itemnos'], mdata['rectimes'] = cmr.run_cmr2_multi_sess(params=params, 
     pres_mat=data['pres_itemnos'], 
     identifiers=identifiers, sem_mat=w2v, 
     source_mat=None, mode=mode) 

mdata['rectimes'] /= 1000
mdata['recalls']=dc.get_recalls(mdata['pres_itemnos'], mdata['rec_itemnos']).astype(int)

mdata['irts']=dc.get_irts(mdata['rectimes'], mdata['recalls'], max_output=24)
mdata['irts'][np.where(mdata['irts']==0)]=np.nan
mdata['model']=True

from crls import *

lag_crl(mdata)
sem_crl(mdata)
opR_crl(mdata)
prevIrt_crl(mdata)

#clean_dat = exclude_false_recalls(recalls,rectimes, rec_itemnos,True, rec_itemnos)




