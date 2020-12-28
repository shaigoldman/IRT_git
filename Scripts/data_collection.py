import scipy.io as sio
import scipy.stats
import numpy as np
import misc_funcs
import os

"""
Created on Mon May 15 15:09:05 2017

@author: shait
"""

def fillArray(size, filler=0, nparray=np.array([])):
    # fits all our data to consistant sizes. Input a size, what to 
    # fill blank spaces with, and an array, and it extends the array
    # to the maximum size, filling the extentions with the filler.
    # If no array is inputed, it functions sort of like np.zeros, but
    # with whatever you want.
    nparray = np.asarray(nparray)
    for i in range(0, size - nparray.size):  # append NaNs if its too small
        nparray = np.append(nparray, filler)
    return nparray


def extend_2darray(ndarray, max_output, filler=0):
    # Same as fillArray but for two dimensions
    new_arr = []
    for trial in ndarray:
        new_arr.append(fillArray(max_output, filler, trial))
    return np.asarray(new_arr)


def clean_rectimes(rectimes, recalls):  # Rectimes is from the events folder, 
    # so naturally it has a bunch NaN values for no apparent reason 
    maxResp = recalls.shape[1]  # we wnat it to fit the size of the recalls matrix
    clean_rectimes = []
    for i in rectimes:
        if i == i: clean_rectimes.append(i[0, 0])  # exclude ALL the NaNs
    # Now we need to even it out, because right now all the rows are different legnths
    # Even bigger a problem is the way the events folder is built there are no rows,
    # its a 1d list.  So we'll have to figure out where the rows should be cut off based
    # on the recalls data.
    rectimes = np.array([[]])
    trial = np.array([])
    trialnum = 0
    for i in range(0, len(clean_rectimes)):
        if not i == 0 and (trial.size == maxResp or recalls[trialnum, trial.size] == 0):
            # i.e. the subject stopped recalling, new FR period, the trial is done
            trial = fillArray(maxResp, np.nan, trial)  # fill up our trial to the size of a recalls row
            try:
                rectimes = np.vstack((rectimes, trial))  # stack the trial up on our 2d array
            except ValueError:  # this will happen the first time since you can't stack something on nothing    
                rectimes = trial  # instead of stacking just create our base layer row
            trial = np.array([])  # reset the trial
            trialnum += 1  # just tracks what trail we are on to compare to recalls matrix
        trial = np.append(trial, clean_rectimes[i])
        if i == len(clean_rectimes) - 1:  # we've reached the end of the road
            rectimes = np.vstack((rectimes, fillArray(maxResp, np.nan, trial)))  # stack the last one
    return rectimes


def exclude_false_recalls(recalls, rectimes, rec_items, lsaing=False, rec_itemnos=None):
    # gets rid of intrusions, repetitions, and the like from our data.
    # In terms of the irts, they will go as if the intrusions never
    # happened (i.e for your first recall, at 1000 you recall item 11, then have an intrusion
    # at 1300, and then recall 7 at 2500. The irt matrix would look like
    # this: [1000,1500]))
    for trial in range(0, recalls.shape[0]):
        recalled = []
        
        for resp in range(recalls.shape[1]):
            if recalls[trial, resp] < 0 or recalls[trial, resp] in recalled:
                recalls[trial, resp] = -1
                rec_items[trial, resp] = ''
                rectimes[trial, resp] = np.nan
                if lsaing:
                    rec_itemnos[trial, resp] = -1
            if recalls[trial, resp] > 0:
                recalled.append(recalls[trial, resp])
                
    return {'recalls': recalls, 'rectimes': rectimes, 'rec_items': rec_items, 'rec_itemnos': rec_itemnos}


def get_irts(rectimes, recalls, max_output):  # Makes an array of calculated interresponse times
    # corresponding to the recalls matrix, where irts[0] would be up until the first item, and irts[1]
    # would be rectimes[1]-rectimes[0]. As noted previously, it ignores false recalls as if they 
    # never occured
    irts = np.zeros([recalls.shape[0], recalls.shape[1]])
    for trial in range(0, recalls.shape[0]):
        irts[trial, 0] = rectimes[trial, 0]
        for pos in range(0, max_output - 1):
            irts[trial, pos+1] = rectimes[trial, pos + 1] - rectimes[trial, pos]
            if irts[trial, pos+1] <= 0:
                irts[trial, pos+1] = np.nan
            
    for i in range(0, irts.shape[0]):
        if True in [np.isnan(j) for j in irts[i]]:
            irts[i][[np.isnan(j) for j 
                in irts[i]].index(True):] = np.nan
    #irts[irts>3.2] = np.nan
    return irts


def get_numsessions(fname):
    data = sio.loadmat(misc_funcs.get_datadir()+'/stat_data' + fname + '.mat')['data']
    try:
        session = np.asarray(
                [i[0] for i in data['session'][0, 0]])
    except IndexError:
        return 0
    return session.max()


def get_rt_from_irt(irttrial):
    rectimes = [0]
    for i in range(1, len(irttrial)):
        rectimes.append(rectimes[i-1]+irttrial[i])
    return np.asarray(rectimes)

def load_scipy(fname):
    data = sio.loadmat(
            misc_funcs.get_datadir()+'/stat_data' + fname + '.mat'
            )['data']

    mydata = {}
    
    mydata['rec_itemnos'] = data['rec_itemnos'][0, 0]
    mydata['pres_itemnos'] = data['pres_itemnos'][0, 0]

    mydata['recalls'] = data['recalls'][0, 0]  # the way scipy formatted it we have the array within two empty arrays
    
    mydata['rec_items'] = data['rec_items'][0, 0]
    
    mydata['trialnums'] = np.asarray(
            [[i] for i in range(1, len(mydata['recalls'])+1)])
    
    shapes = np.asarray(
            [[np.asarray(j.shape).tolist() for j in i] 
            for i in mydata['rec_items']])
    
    for i in range(shapes.shape[0]):
        for j in range(shapes.shape[1]):
            if shapes[i,j] == [0]:
                mydata['rec_items'][i, j] = np.array([]).reshape(1,0)
    
    mydata['rec_items'] = np.asarray(
            [[i[0] for i in j] for j in mydata['rec_items']])
    mydata['rec_items'][mydata['rec_items'] == u'[]'] = u''
    
    for i in range(mydata['rec_items'].shape[0]):
        for j in range(mydata['rec_items'].shape[1]):
            mydata['rec_items'][i,j] = str(mydata['rec_items'][i,j])
    
    
    mydata['rectimes'] = data['times'][0][0].astype('float64')/1000.
    mydata['rectimes'][mydata['rectimes'] == 0] = np.nan
    
    mydata['session'] = np.asarray(
        [i[0] for i in data['session'][0, 0]])
    
    subjnum = data['subject'][0][0][0]
    mydata['subject'] = []
    for i in range(0, mydata['recalls'].shape[0]):
        mydata['subject'].append(subjnum)
    mydata['subject'] = np.asarray(mydata['subject'])
    

def append_sessdata(mydata):
    sortedses = [[] for i in np.arange(mydata['session'].max())]
    for i in mydata['session']:
        if sortedses[i-1]:
            sortedses[i-1].append(sortedses[i-1][-1]+1)
        else:
            sortedses[i-1].append(0)
    mydata['sestrialnums'] = []   
    for i in sortedses:
        mydata['sestrialnums'].extend(i)
    mydata['sestrialnums'] = np.asarray(mydata['sestrialnums'])


def remove_empty_trials(mydata, max_output=24):
    index = mydata['session'].tolist().index(24)        
    
    for i in mydata:
        mydata[i] = mydata[i][:index]
    
    clean_recalls = []  # use this to clear trials where the participant said nothing at all
    for i in range(0, mydata['recalls'].shape[0]):
        if len(mydata['recalls'][i][mydata['recalls'][i]>0])>2:
            if (-1 in mydata['recalls'][i] and 
                mydata['recalls'][i].tolist().index(-1) <= 1):
                continue
            clean_recalls.append(i)
    
    for i in mydata:
        mydata[i] = mydata[i][clean_recalls]
        if len(mydata[i].shape)>1:
            mydata[i] = extend_2darray(mydata[i], max_output)
    

def get_data(fname, max_output=24, 
             model=False, excludeFromMod=[]):
    # returns the recalls matrix along with the irts and rectimes arrays that it 
    # finds from the given file. max_output is for formatting, it makes the row
    # legnth of each equal to it

    mydata = load_scipy(fname)
    append_sessdata(mydata)
    remove_empty_trials(mydata, max_output)

        
    clean_dat = exclude_false_recalls(mydata['recalls'], 
                                      mydata['rectimes'], 
                                      mydata['rec_items'],
                                      True,
                                      mydata['rec_itemnos'])
    
    for i in clean_dat:
        mydata[i] = clean_dat[i]
        
    mydata['irts'] = get_irts(mydata['rectimes'], 
          mydata['recalls'], max_output)
    
    #responses past 24 aren't quality data
    for i in mydata:
        if len(mydata[i].shape)>1:
            mydata[i] = mydata[i][:, 0:max_output]      
    
    #Kill sessions with negative irts
    nirts = mydata['irts'].copy()
    nirts[np.isnan(nirts)] = 1
    for i in mydata:
        mydata[i] = np.delete(
                mydata[i], np.where(nirts < 0), axis = 0)
        
    for i in range(0, mydata['rec_itemnos'].shape[0]):
        for j in range(1, mydata['rec_itemnos'].shape[1]):
            if (np.isnan(misc_funcs.get_semrelat(
                    mydata['rec_itemnos'][i, j],
                     mydata['rec_itemnos'][i, j-1])) 
                and not (mydata['recalls'][i, j] <= 0 or 
                     mydata['recalls'][i, j-1] <= 0)):
                mydata['irts'][i,:] = np.nan
        
    
    i = 0; m = mydata['irts'].shape[0]
    while i < m:
        if len(mydata['irts'][i][~np.isnan(mydata['irts'][i])]) < 3:
            for key in mydata:
                mydata[key] = np.delete(mydata[key], i, axis = 0)
            i -= 1; m-= 1
        i += 1
        
    if model:
        tobcx = mydata['irts'].copy()
        tobcx[:, 0] = np.nan
    
        trialnums = []
        for trial in range(tobcx.shape[0]):
            for op in range(tobcx.shape[1]):
                if not np.isnan(tobcx[trial, op]):
                    trialnums.append(trial)
        
        tobcx = tobcx[~np.isnan(tobcx)]
        bcx, mydata['lambda'] = scipy.stats.boxcox(tobcx)
        mydata['bcx'] = misc_funcs.twodirts(bcx, trialnums)
#        path = misc_funcs.get_scriptdir()+'/simulations'
#        data = sio.loadmat(path+'/stat_data' + fname + '.mat')['data']['net'][0][0]
        os.chdir(misc_funcs.get_scriptdir())
        from regression import get_pred_1subj
        mydata['logged_irts'] = np.log(mydata['irts'])
        mydata['irts'] = get_pred_1subj(fname, mydata, excludeFromMod)
        
        for i in range(mydata['irts'].shape[0]):
            mydata['rectimes'][i] = get_rt_from_irt(mydata['irts'][i])
            
    mydata['trialnums'] = mydata['trialnums'][:, 0]
    
    mydata['logged_irts'] = np.log(mydata['irts'])
    tobcx = mydata['irts'].copy()
    tobcx[:, 0] = np.nan

    trialnums = []
    for trial in range(tobcx.shape[0]):
        for op in range(tobcx.shape[1]):
            if not np.isnan(tobcx[trial, op]):
                trialnums.append(trial)
    
    tobcx = tobcx[~np.isnan(tobcx)]
    bcx, mydata['lambda'] = scipy.stats.boxcox(tobcx)
    mydata['bcx'] = misc_funcs.twodirts(bcx, trialnums)
    
    
    mydata['subject'] = mydata['subject'].reshape(mydata['subject'].size)
    
    
#    while np.inf in mydata['irts']:
#        for i in range(0, mydata['irts'].shape[0]):
#            if np.inf in mydata['irts'][i]:
#                for key in mydata:
#                    mydata[key] = np.delete(mydata[key], i , axis=0)
#                break 
    mydata['model'] = model
    
    Rs = np.asarray([len(i[i>0]) for i in mydata['recalls']])
    mydata['Rs'] = Rs
        
    return mydata


def get_recalls(pres_itemnos, rec_itemnos):
    
    recalls = np.zeros(pres_itemnos.shape)
    
    for i in range(pres_itemnos.shape[0]):
        for j in range(pres_itemnos.shape[1]):
            if rec_itemnos[i,j] in pres_itemnos[i]:
                recalls[i,j] = np.where(pres_itemnos[i]==rec_itemnos[i,j])[0][0]+1
            elif rec_itemnos[i,j] == 0:
                pass
            else:
                recalls[i, j] = -1;
                
    return recalls


def stack_alldata(model=False, save=True):
    def vstack(x, y):
        try:
            if len(np.asarray(x).shape) == 1 and len(np.asarray(y).shape) == 1:
                
                x = np.append(x, y)
                return x
        except AttributeError:
            pass
        if x is None:
            return y
        try:
            return np.vstack((x,y))
        except ValueError:
            return x
        
    files = misc_funcs.get_ltpFR2_filenames()
    keys = get_data('_LTP228').keys()
#    ['rec_itemnos', 'pres_itemnos', 
#            'recalls', 'irts', 'rectimes', 'subject',
#            'Rs', 'rec_items', 'session', 'bcx', 'lambda']
    all_data = {}
    for i in keys:
        all_data[i] = None
    
    rstore = []
    
    for f in range(0, len(files)):
        print (files[f], ':', f, '/', len(files)-1)
        fname = files[f]
        try:
            data = get_data(fname, model=model)
            #print data['irts']
        except IndexError:
            print ('ERROR')
            continue
#        for i in range(0, data['recalls'].shape[0]):
#            all_data['subject'].append(f)
        for key in keys:
            all_data[key] = vstack(all_data[key], data[key])

        rstore.append(all_data['Rs'].copy())
        #print all_data['session']
        
    all_data['model'] = model
    
    all_data['irts'][:, 0] = np.nan
    
    if save:
        np.save(misc_funcs.get_resourcedir()
            +'/'+('mod'*model)+'data.npy', all_data)
    
    return all_data

def truck_alldata(alldata, lists=1000):
    for key in alldata:
        if key == 'model':
            continue
        alldata[key]=alldata[key][:lists]
    return alldata
    
    
def get_alldata():
    adata = np.load('/Users/lumdusislife/Desktop/IRT/Scripts/Resources/data.npy', allow_pickle=True, encoding='latin1')[()]
    return adata
    
    
    