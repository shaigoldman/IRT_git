import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as ss
from scipy.stats.mstats import zscore
from tqdm.notebook import tqdm

import DataBuild as db


def prep_data_for_ols(data, normalize=True, model_num=None, **kwargs):
    """ Prepare data for OLS modeling by flattening it,
        running the equations on it (e.g., OP=1/(LL-OP))
        and normalizing it if desired.
        
        'model_num' can be 1,2, or 3, depending on which model
        from the paper we are working with.
        
        Note: kwargs is unused here, it is only to allow
        for using the same kwargs in this func and in DataBuild
        functions.
    """
    
    
    
    #-----flatten data----#
    flat_data = {}
    for key in data:
        if len(data[key].shape) > 1:
            flat_data[key] = data[key].values.flatten()
        else:
            flat_data[key] = np.repeat(data[key], db.LL).values.flatten()

    flat_data = pd.DataFrame(flat_data)
    
    # include output pos as a variable
    flat_data['output_pos'] = np.repeat([db.OUTPT_PSTNS], db.NLIST, axis=0).flatten()

    #-----filter data----#
    # remove keys that wont go into the model
    for key in ['pres_words', 'pres_nos', 'rec_words',
                'rec_nos', 'recalled', 'times', 'intrusions',
                'subject', 'good_trial', 'recalls'
               ]:
        flat_data.pop(key)
    # remove keys for basic models 1&2
    if model_num is not None:
        if model_num < 3:
            # models 1&2 do not include practice effects
            flat_data.pop('session')
        if model_num < 2:
            # model 1 does not include irt-lags
            keys_to_pop = [key for key in flat_data.keys() if 'irt-' in key]
            for key in keys_to_pop:
                flat_data.pop(key)

    # remove all nans in prep for modeling
    for key in flat_data:
        flat_data = flat_data[~np.isnan(flat_data[key])]
        
    #----adjust some variables for the model----#
    # output position is inverted
    flat_data['output_pos'] = flat_data['output_pos'].astype(float)
    flat_data['output_pos'] = (db.LL-flat_data['output_pos']) ** -1
    # total recalls is normalized
    flat_data['total_recalls'] /= db.LL
    # lag is taken as ln(|lag|)
    flat_data['lag'] = np.log(np.abs(flat_data['lag']))
    
    # include lag sim interaction
    flat_data['lag_sem'] = flat_data['lag'] * flat_data['sem']
    
    
    bcx, lmbda = ss.boxcox(flat_data['irt'])
    flat_data['irt'] = bcx
    
    # take bcx of irts
    for key in flat_data.keys():
        if not 'irt-' in key:
            continue
        bcx = ss.boxcox(flat_data[key], lmbda=lmbda)
        flat_data[key] = bcx
        
    # normalize
    if normalize:
        for key in flat_data.keys():
            flat_data[key] = zscore(flat_data[key])
    
    return flat_data


def fit_model(flat_data):
    """Fits an OLS model to flattened data."""
    flat_data = flat_data.copy()
    X = sm.add_constant(flat_data)
    y = X.pop('irt')

    model = sm.OLS(y, X)
    return model.fit()


def get_model(filename=None, subject=None, **kwargs):
    """ Loads data, flattens it, and fits a model to it.
        Can be used either by giving the file path, or
        subject ID.
    """
    data = db.load_detailed_data(subject=subject, path=filename, **kwargs)
    flat_data = prep_data_for_ols(data, **kwargs)
    return fit_model(flat_data)


def data_mult_subj(files, irt_lags, min_irt):
    """Concatenates flattened data accross many subjects."""
    
    filename = f'alldata_il{irt_lags}mi{min_irt}.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    
    all_data = []
    for file in tqdm(files):
        subj = file.split('LTP')[1].replace('.json', '')
        data = db.load_detailed_data(path=file, irt_lags=irt_lags, min_irt=min_irt)
        flat_data = prep_data_for_ols(data)
        flat_data['subject'] = int(subj)
        all_data.append(flat_data)

    all_data = pd.concat(all_data)
    all_data.index = range(len(all_data))
    
    all_data.to_pickle(filename)
    
    return all_data