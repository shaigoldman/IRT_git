import os
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm
from glob import glob

# CONSTANTS
LL = 24 # listlength
NSESS = 24 # number of sessions
NLIST = 24*NSESS # number of lists total accross all sessions
OUTPT_PSTNS = np.arange(LL) # the possible output possitions
LIST_NUMS = np.arange(NLIST) # the possibile list numbers
DATA_KEYS = ['subject', 'session', 'good_trial', 'pres_words',
             'pres_nos', 'rec_words', 'rec_nos', 'recalled',
             'times', 'intrusions', 'recalls', 'lag', 'irt',
             'total_recalls', 'sem'] # the keys our data should have
# PATHS
BEH_PATH = '/data/eeg/scalp/ltp/ltpFR2/behavioral/data/'
W2V_PATH = '/home1/shai.goldman/IRT_git/Scripts/Resources/w2v.mat'
# DEFAULT VALUES
def_irt_lags = 2 # number of irt_lags to count
def_min_irt = def_irt_lags + 1 # min irt that isn't NaN'd out. 
# ^ we NaN all irts without a corresponding irt-lag at the max lag num
# LOAD W2V
try:
    w2v = sio.loadmat(W2V_PATH)['w2v']
except FileNotFoundError:
    print('No w2v was able to load')

def get_subjs():
    """Returns a list of LTP subj names"""
    if os.path.exists(BEH_PATH):
        # on rhino
        files = [f for f in glob(BEH_PATH+'beh_data_LTP*.json') if 'incomplete' not in f]
        subjs = ['LTP'+file.split('LTP')[1].replace('.json', '') for file in files]
    else:
        # use preloaded package data
        path = './data/*'
        subjs = [f.split('/')[-1] for f in glob(path)]
    return subjs
    

def load_data(filename=None, subject=None):
    """ Loads data from BEH_PATH, either using a direct path
        with param 'filename' or building the path using subject
        code, in format of either LTP### or ###.
    """
    
    if filename is None and subject is None:
        raise ValueError('filename and subject cannot both be None')
    elif subject is not None:
        if not 'LTP' in subject:
            subject = f'LTP{subject}'
        filename = f'{BEH_PATH}beh_data_{subject}.json'

    x = pd.read_json(filename)

    # convert data stats to pandas dfs with columns for
    # output position and rows for list number
    data = {}
    for key in x.keys():
        data[key] = np.array([i for i in x[key].values])
        if len(data[key].shape) > 1:
            data[key] = data[key][:, :LL]
            data[key] = np.pad(data[key],
                               [(0,0), (0,LL-data[key].shape[1])],
                               mode='edge')
            data[key] = pd.DataFrame(data[key], columns=OUTPT_PSTNS, index=LIST_NUMS)
        else:
            data[key] = pd.Series(data[key], index=LIST_NUMS)
    # I prefer the old naming convention for 'recalls' matrix
    data['recalls'] = data.pop('serialpos')
    
    return data


def get_lags(data):
    """Calculates serial lags from base data dict."""
    prev_rec = data['recalls'].loc[:, :LL-2]
    prev_rec.columns = range(1,LL)
    return data['recalls'] - prev_rec


def strech_intrus(data):
    """ Stretches intrusions so that every recall after an
        intrusion is also counted as an intrusion
    """
    data = data.copy()
    # remove all recalls after the first intrusion in a list
    first_intrus = pd.Series([list(i).index(-1) if -1 in i
                                 else len(i) for i in data['recalls'].values],
                                index = LIST_NUMS
                               )
    for ls in data['recalls'].index:
        fi = first_intrus.loc[ls]
        data['recalls'].loc[ls, fi:] = 0
        data['times'].loc[ls, fi:] = 0
    return data


def get_irts(data):
    """Calcualte IRTs from data dict."""
    prev_times = data['times'].loc[:,:LL-2].astype(float)
    prev_times.columns = range(1, LL)

    irts = data['times'] - prev_times
    irts[irts<=0] = np.nan
    return irts/1000 # divide by 1000 for time in seconds


def add_prev_irts(data, lags=def_irt_lags):
    """Adds prev irt (irt-lags) into the data df."""
    
    for lag in range(1,lags+1):
        # shift all the irts by the lag
        prev_irts = data['irt'].loc[:, :LL-lag-1].copy()
        prev_irts[0] = 0 # change from NAN to zero
        prev_irts.columns = (OUTPT_PSTNS+lag)[:-lag]
        # insert [ZEROS] (nans) for OUTPT_PSTNS before the first lag
        for output in range(lag):
            prev_irts[output] = 0 #np.nan
        # resort columns since we added the first OUTPT_PSTNS to the end
        prev_irts = prev_irts[sorted(prev_irts.columns)]
        # input to data array
        data[f'irt-{lag}'] = prev_irts
        
        
def sem_sim(a, b):
    """Helper func for finding semantic sims."""
    if a <= 0 or b <= 0:
        return np.nan
    # the -1 is very important because of 0 indexing in python vs matlab
    # they originally started the rec_nos from index 1 when the lab was
    # matlab and to have a w2v matrix its going to start from index 0 in python
    return w2v[a-1, b-1]


def get_sems(data):
    """calc semantic similarities"""
    sims = [[sem_sim(row.loc[i-1], no) 
              if i>0 else np.nan
              for i, no in row.iteritems()]
             for r, row in data['rec_nos'].iterrows()]
    return pd.DataFrame(sims, index=LIST_NUMS, columns=OUTPT_PSTNS)


def ddata_path(subject, create=False):
    """Gives a path for local detailed data for saving/loading."""
    saves_dir = 'data/'
    if create and not os.path.exists(saves_dir):
        os.mkdir(saves_dir)
    saves_dir += subject +'/'
    if create and not os.path.exists(saves_dir):
        os.mkdir(saves_dir)
    return saves_dir


def save_data(data):
    """ Saves all the data to data/SUBJECT/ path"""
    subject = str(data['subject'].unique().squeeze())
    saves_dir = ddata_path(subject, create=True)
    for key in data:
        data[key].to_pickle(saves_dir+key+'.pkl')
        
        
def cut_irts(data, min_irt=def_min_irt, irt_lags=def_irt_lags, **kwargs):
    """ cuts off (i.e., turns to NaN) irts before min irt.
        Note: kwargs is not used here, it is only to allow
        for using the same, larger kwargs in multiple functions.
    """
    if min_irt is None:
        return
    data['irt'].loc[:, :min_irt-1] = np.nan
    for lag in range(1, irt_lags+1):
        data[f'irt-{lag}'].loc[:, :min_irt-1] = np.nan
        
        
def detailed_data(data, irt_lags=def_irt_lags, min_irt=def_min_irt, save=True, model_num=None):
    """ Adds details extrapolated from the basic data,
        and edits some of the data for our analyses.
        Additions:
            'lag': serial lag between items
            'sem': semantic lags between items
            'total-recalls': number of items recalled from a list
            'irt': inter-response times
            'irt-X': irts of lag X, will include 'irt_lags' num of
                these items
        Edits:
            Converts repeat recalls to -1, i.e., intrusions
            Considers all recalls following an intrusion as intrustions
            Sets IRTs before min_irt to NaN
        
        Params:
            data (df): base data from 'load_data'
            irt_lags (int): how many irt_lags to count
            min_irt (int): the minimum irt that isnt NaN'd. This is
                relevant bc if we are counting irt_lag 2, we want
                to discount all irts that have no 2nd lag, so we 
                should set min_irt to 3.
            save (bool): if True, saves the result in  'data/'
            model_num (None): this param is not used, but necessary
                for expanding **kwargs for usage here and in
                Regressions.py
        
        Returns:
            pd.DataFrame containing detailed data
    """
    
    data = data.copy()
    
    data['lag'] = get_lags(data)
    
    # set all repeats as intrusions
    data['recalls'][data['lag']==0] = -1
    
    # remove all recs after an intrus
    data = strech_intrus(data)
    
    data['irt'] = get_irts(data)
    add_prev_irts(data, lags=irt_lags)
    
    # calc total recalls per list
    data['total_recalls'] = pd.Series(
        [r[r>0].size for i, r in data['recalls'].iterrows()],
        index=LIST_NUMS
    )
    
    data['sem'] = get_sems(data)
    
    if save:
        save_data(data)

    # nan out irts under the minimum
    cut_irts(data, min_irt=min_irt, irt_lags=irt_lags)    
    
    return data


def load_detailed_data(subject=None, path=None,
                       irt_lags=def_irt_lags,
                       reload=False,
                       **kwargs):
    """ Loads detailed data if it exists, otherwise creates it."""
    # get data path
    if subject is None and path is None:
        raise ValueError('subject and path cannot both be None.')
    if subject is None:
        if '.json' in path:
            subject = path.split('beh_data_')[1].replace('.json', '')
        elif '.pkl' in path:
            path = '/'.join(path.split('/')[:-1])
    if subject is not None:
        path = ddata_path(subject, create=False)
    
    # load the data
    if reload:
        print(f'Reloading data for {subject}...')
        data = detailed_data(load_data(subject=subject),
                                 irt_lags=irt_lags, **kwargs)
    else:
        data = {}
        for filename in glob(path+'/*'):
            key = filename.split('/')[-1].replace('.pkl', '')
            data[key] = pd.read_pickle(filename)
        for key in DATA_KEYS:
            if not key in data:
                print('Not all data found. Loading data...')
                data = detailed_data(load_data(subject=subject),
                                     irt_lags=irt_lags, **kwargs)
                break
    # add missing lags
    for lag in range(1, irt_lags+1):
        if not f'irt-{lag}' in data:
            add_prev_irts(data, lags=irt_lags)
            break
    
    # nan out irts under the minimum
    cut_irts(data, irt_lags=irt_lags, **kwargs)
            
    # remove extra lags
    lag_keys = [int(i.replace('irt-', '')) for i in data if 'irt-' in i]
    for lag in lag_keys:
        if lag > irt_lags:
            data.pop(f'irt-{lag}')
    
    return data