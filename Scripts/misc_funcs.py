
#FOR SHAIS LAPTOP - data=np.load('Resources/data.npy').tolist()

def get_listlength():
    return 24

def setpath():
    import os
    # checks if we are in rhino, or a mounted system through our personal CPU
    try:
        os.chdir('/data10')
        return '/home1/shai.goldman/IRT2017/'
    except OSError:
        try:
            os.chdir('/Users/lumdusislife/rhino_mount/home1/shai.goldman/IRT2017')
            return '/Users/lumdusislife/rhino_mount/home1/shai.goldman/IRT2017/'
        except OSError:
            return '/Users/lumdusislife/Desktop/IRT/'


# Folder with all our python scripts
def get_scriptdir():
    return setpath() + 'Scripts'


# Folder with all our text files and such
def get_resourcedir():
    return setpath() + 'Scripts/Resources'


def get_datadir():
    return setpath() + '../../../data10/eeg/scalp/ltp/ltpFR2/behavioral/data'


# where we keep our graphs
def get_graphdir():
    return setpath() + 'TempGraphs'
    

def get_LSA():
    import os
    os.chdir(get_resourcedir())
    import scipy.io as sio
    LSA = sio.loadmat('LSA_withnans.mat')['LSA']
    return LSA


def get_w2v():
    import scipy.io as sio, os
    try:
        return sio.loadmat(
                get_resourcedir()+'/w2v.mat')['w2v']
    except IOError:
        return sio.loadmat(
                '/Users/lumdusislife/Desktop/IRT/Scripts/Resources/w2v.mat')['w2v']
    
    
def get_WAS():
    import numpy as np
    WAS = np.loadtxt(
            '/Users/shait/Desktop'+
            '/word2vec/wasnorm_was.txt')
    
    return WAS


def get_wfreq():
    import scipy.io as sio
    import numpy as np
    return np.asarray([i[0] for i in sio.loadmat(
            get_resourcedir()+'/Frequency_norms.mat')['F']])
  
def subscript(i, neg=True, pos=False):
    num = unichr(8320+i)
    sign = unichr(8330 + neg)
    return sign+num
    
def readTxtLst(fname):
    # just returns a list of items from a text file, used to interpret the
    # all_ltpFR2_completed.txt file
    import re
    lines = []
    f = open(fname + '.txt')
    for line in f:
        lines.append(re.sub('\n', '', re.sub('\r\n', '', line)))
    return lines


def get_ltpFR2_filenames(onlyComp=True, reeval=False, act=None):
    # Returns a list of all the ltpFR2 participants in the following format: _LTP###
    # If onlyComp is true, it only returns those who completed the paradigm
    import os, re, numpy as np
    
    if act:
        lis = np.unique(act['subject']).astype('S')
        for i, item in enumerate(lis):
            print (item, '_LTP' + item)
            lis[i] = '_LTP' + item
        return lis
    
    files = os.listdir(get_datadir())

    clean_files = []
    for file in range(0, len(files)-1):
        if '.mat' in files[file] and not '214' in files[file]:  # ltp214 is empty for some reason
            fname = re.sub('stat_data', '', files[file])  # isolate it from the prefix so it can be used
            # for events files too
            fname = re.sub('.mat', '', fname)  # get rid of the .mat tag for simplicity
            clean_files.append(fname)
    files = clean_files
    
    if onlyComp:
        if not reeval:
            try:
                os.chdir(get_resourcedir())
                files = [re.sub('\n', '', i) 
                         for i in readTxtLst('comp_fnames')]
            except IOError:
                reeval=True
            
        if reeval:
            os.chdir(get_scriptdir())
            from data_collection import get_numsessions
            newfiles = []
            for i in files:
                if get_numsessions(i) >= 23:
                    newfiles.append(i)
            files = newfiles
            os.chdir(get_resourcedir())
            writeArrayFile('comp_fnames', files)
        
    
    return files


def readArrayFile(file):
    # reads a text file (only numericals, spaces and lines) as a numpyarray of float64s.
    # Each row is a line, each element is seperated by a space
    if not '.txt' in file:
        file += '.txt'
    import numpy as np
    f = open(file, 'r')
    text = f.read()
    f.close()
    grand_arr = []
    array = []
    row = np.array([])
    num = ''
    for i in range(0, len(text)):
        char = text[i]
        num += char
        if char == ' ':
            try:
                row = np.append(row, float(num))
            except ValueError:
                continue
            num = ''
        elif char == '\n':
            if text[i - 1] == '\n':  # double newline means new 3d space
                grand_arr.append(np.asarray(array))
                array = []
            else:
                array.append(row)
                row = []
    try:
        row = np.append(row, float(num))
    except ValueError:
        pass
    if row:
        array.append(row)
    try:
        array[1]  # this is to stop it from embedding a 1d array in 2 dimensions like so: [[array]]
    except IndexError:
        try:
            array = array[0]
        except:
            pass
    if grand_arr:
        if array: grand_arr.append(array)
        return grand_arr
    return np.asarray(array)


def writeArrayFile(filename, nparray, d='default', string=False):
    # Writes a numpy array, python tuple, or list into a text file with the specified name
    # Each row is a line, each element is seperated by a space
    # Used in conjecture with either readArrayFile or readTupleFile
    import numpy as np
    if not '.txt' in filename:
        filename += '.txt'
    f = open(filename, 'w')
    nparray = np.asarray(nparray)  # just in case it was submitted as a list
    copy = nparray.copy()
    
    if d == 'default':
        d = 0
        for i in range(0, 5):
            try:
                copy = copy[0]
                
                if type(copy) == str:
                    string=True
                    raise TypeError

            except (TypeError, IndexError):
                d = i
                break
            
    btwn = ' '
    if string:
        btwn = '\n'        
    
    if d == 1:
        for i in nparray:
            f.write(str(i) + btwn)
    elif d == 2:
        for i in nparray:
            for j in i:
                f.write(str(j) + ' ')
            f.write('\n')
    elif d == 3:
        for i in nparray:
            for j in i:
                for k in j:
                    f.write(str(k) + ' ')
                f.write('\n')
            f.write('\n')
    f.close()


def readTupleFile(file):
    # reads a text file of floats as a tuple. Helpful when the dimensions don't match.
    # specifically used to read histograms.
    # Each row is a line, each element is seperated by a space
    file += '.txt'
    f = open(file, 'r')
    text = f.read()
    f.close()
    array = []
    row = []
    num = ''
    for char in text:
        num += char
        if char == ' ':
            try:
                row.append(float(num))
            except ValueError:
                continue
            num = ''
        elif char == '\n':
            array.append(row)
            row = []
    try:
        row.append(float(num))
    except ValueError:
        pass
    if row != []: array.append(row)
    return tuple(array)


def get_lag(rec1, rec2):
    import numpy as np
    return np.absolute(rec2-rec1)    


def get_semrelat(itemno1, itemno2, w2v=get_w2v()):
    import numpy as np
    try:
        if itemno1 < 0 or itemno2 < 0:
            return np.nan
    except ValueError:
        #array inputs
        pass
    try: 
        return w2v[itemno1-1, itemno2-1]
    except IndexError:
        print (itemno1, itemno2)
        return np.nan


def plotTable(data, clabels, rlabels, fname='temp'):
    from matplotlib import pyplot as plt
    def plot():
        plt.axis('tight')
        plt.axis('off')
                        
        plt.table(cellText=data, colLabels=clabels,
           rowLabels=rlabels, loc='center', 
           colWidths=[.3]*len(clabels))
    plt_std_graph(plot, fsize=12, fname=fname)


def lmerr(data):
    def loftus_mason(data):
        # Finds loftus mason errors off a 2dimensional data matrix (must be formatted as matlab.double)
        import os, matlab.engine
        # access the lab's matlab code to find it for us
        os.chdir('/home1/shai.goldman/BootCamp')
        eng = matlab.engine.start_matlab()
        err = eng.l_m(data)
        return err
    # formats a numpy array so that loftus mason error can be found on it using the previous function
    # returns a nparray of the loftus_mason func's output
    import numpy as np
    import matlab
    double_matrix = []  # lets convert our np array to a matlab double matrix (technically still
    # an array but I like to refer to them as matrixes to distinguish between python and matlab)
    for i in data:
        dimension = []  # we need to get a regular list of lists instead
        # of a 2d np array because matlab can't read np arrays
        for j in i:
            dimension.append(j)
        double_matrix.append(dimension)
    double_matrix = matlab.double(double_matrix)
    err = loftus_mason(double_matrix)
    err = np.asarray(err)[0]  # it returns it as a 2d array with the data embedded in the 0 of the second d
    return err


def sterr(data, axis=None):
    # just magnified standard deviation up to 95% confidence.
    import numpy as np
    data = np.asarray(data)
    std = np.nanstd(data, axis=axis)
    err = std / np.sqrt(data[~np.isnan(data)].size)  # 63% conf
    err *= 1.96  # 95% conf
    return err
    

def flat_arr_to_bins(arr, bin_num, sort=True):
    import numpy as np
    arr = np.asarray(arr)
    if sort:
        arr = np.sort(arr.flatten())
    reshaped = arr[:arr.size/bin_num*bin_num].reshape([bin_num, arr.size/bin_num]).tolist()
    reshaped[-1]=np.append(reshaped[-1], arr[arr.size/bin_num*bin_num:])
    for i in range(len(reshaped)):
        reshaped[i] = np.asarray(reshaped[i])
    return reshaped
    

def which_bin(binned_arr, value):
    for i, arr in enumerate(binned_arr):
        if arr.min() <= value <= arr.max():
            return i
    print ('Warning: bin not found')
    return float('nan')


def plt_std_graph(plot, fname='temp', dir=get_graphdir(), 
                  show=True, tight=True, fsize=15, width=6.5, height=5.5,
                  letter=None):
    from matplotlib import pyplot as plt
    import os


    fig = plt.gcf()
    fig.set_size_inches(width, height)
    
    plt.rcParams.update({'font.size': fsize})
    plt.rcParams.update({'errorbar.capsize': 3})
    plt.rcParams.update({'xtick.major.top': True})
    plt.rcParams.update({'ytick.major.right': True})
    
    out = plot()
    if letter:
        plt.text(.02, .94, '%s.' % letter, transform=plt.axes().transAxes, size='large', weight='bold')

    if tight:
        plt.tight_layout()

    os.chdir(dir)
    fig.savefig(fname+'.eps', format='eps')
    fig.savefig(fname+'.png', format='png', dpi=500)
    if show:
        plt.show()
    plt.clf()
    return out


def get_sub_data(fulldata, sub):
    import numpy as np
    indecies = np.where(fulldata['subject']==sub)
    subdata = {}
    for key in fulldata:
        if key == 'lambda':
            subdata[key] = fulldata[key][np.where(np.unique(fulldata['subject'])==sub)]
        elif type(fulldata[key]) == type(np.array([])):
            subdata[key] = fulldata[key][indecies]
        else:
            subdata[key] = fulldata[key]
    return subdata

def difference(x, y, list_legnth): #for use in Lag_CRP
    # function that finds the lag between two serial positions
    # if there is no difference, the lag will be listed as NaN. This should never happen though.
    # If there is a negative difference, it places it in a corresponding place between 0
    # and the number of total items
    # If there is a positive difference, it boosts it to be between 1 more than the
    # total number of itmes and twice the number of items *2-1
    # EXAMPLE:
    # if there are 5 items, lags -4 to -1 will go under 0 to 4, 5 will
    #  be NaN, and +1 to +4 will be under 6 to 9
    diff = x - y;
    if diff == 0:
        diff = float('nan')
    elif diff < 0:
        diff = list_legnth - (diff * -1)
    else:
        diff += list_legnth
    return diff


def lag_crp(recalls, list_length=24, plt=True):  # finds the lag-CRP values (numerators and denominators from 0-47, 24 being NaN
    import numpy as np
    numerators = np.zeros([list_length * 2 - 1])  # total times each lag ACTUALLY happened
    denominators = np.zeros([list_length * 2 - 1])  # total times each lag COULD HAVE happened
    for i in range(0, recalls.size / recalls[0].size):  # iterate through trials
        recalled = []  # exclude repeats
        for j in range(0, recalls[i].size):  # iterate through responses
            if recalls[i, j] in recalled or recalls[i, j] < 1:
                continue
            try:
                if recalls[i, j + 1] < 1:
                    continue
            except IndexError:
                continue
            recalled.append(recalls[i, j])
            # DENOMINATORS
            for k in range(1, list_length + 1):
                if k in recalled:
                    continue
                diff = difference(k, recalls[i, j], list_length)
                denominators[diff - 1] += 1
            # NUMERATORS
            if recalls[i, j + 1] in recalled:
                continue
            diff = difference(recalls[i, j + 1], recalls[i, j], list_length)
            numerators[diff - 1] += 1
            
    def plot():
        from matplotlib import pyplot as plt
        plt.errorbar(np.arange(-list_length+1, list_length), 
                     numerators/denominators, 
                     color='k',
                     marker='o',
                     linewidth=3)
        plt.xlabel('Temporal Lag')
        plt.ylabel('Recall Prob')
    
    if plt:
        plt_std_graph(plot, 'lag_crp')        
    
    return [numerators, denominators]


def semantic_crp(data, LSA=get_w2v(), typ='W2V', bin_num=10, savename=''):
    
    rec_itemnos = data['rec_itemnos']
    pres_itemnos = data['pres_itemnos']
    recalls = data['recalls']
    
    import os
    os.chdir(get_scriptdir())
    from semantic_crp import sem_crp
    crp = sem_crp(pres_itemnos, rec_itemnos, 
                  recalls, LSA, bin_num=bin_num)
    
    def plot():
        from matplotlib import pyplot as plt
        plt.errorbar(crp['bin_mean'], 
                     crp['crp'], 
                     yerr=crp['err'],
                     color='k',
                     marker='o',
                     linewidth=3)
        plt.xlabel(typ+' semantic similarity')
        plt.ylabel('Conditional recall probability')
    
    
    plt_std_graph(plot, savename + typ + 'sem_crp')
    
    
def get_flat_irts_lags_and_sims(data):
    import numpy as np
    w2v = get_w2v()
    
    lags = []
    sims = []
    irts = []
    for trial in range(len(data['recalls'])):
        for rec in range(1, len(data['recalls'][trial])):
            if data['recalls'][trial][rec] <= 0 or data['recalls'][trial][rec-1] <= 0:
                continue
            lags.append(data['recalls'][trial][rec] - data['recalls'][trial][rec-1])
            sims.append(get_semrelat(data['rec_itemnos'][trial][rec], data['rec_itemnos'][trial][rec-1], w2v))
            irts.append(data['irts'][trial][rec])
    
    lags = np.asarray(lags)
    sims = np.asarray(sims)
    irts = np.asarray(irts)
    lags=lags[irts<75]
    sims=sims[irts<75]
    irts=irts[irts<75]
    
    return irts, lags, sims

def spc(recalls):
    import numpy as np
    numerators = np.zeros(recalls.shape[1])
    denominator = 0
    for i in recalls:
        recalled = []
        for j in i:
            if j not in recalled and j > 0:
                numerators[j-1] += 1.
                recalled.append(j)
        denominator += 1
    return numerators/denominator

def new_spc(recalls):
    import numpy as np
    from matplotlib import pyplot as plt
    spc = np.asarray([recalls[recalls==i].size 
        for i in range(1,recalls.max()+1)])/float(recalls.shape[0])
    plt.plot(spc)
    
def lag_av_sim(data, minLag=-23, maxLag=23):
    import numpy as np
    import os
    os.chdir(get_scriptdir())
#    vs = get_vars(data, absolute_lags=False)
#    
#    irts = data['irts'][vs['yind'], vs['xind']]
#    lags = vs['lag']
#    sims = vs['semlag']

    irts, lags, sims = get_flat_irts_lags_and_sims(data)
           
    irts_by_lag = []
    sims_by_lag = []
    for i in range(minLag, maxLag+1):
        irts_by_lag.append(irts[lags==i])
        sims_by_lag.append(sims[lags==i])
    
    irts_by_lag = np.asarray(irts_by_lag)
    sims_by_lag = np.asarray(sims_by_lag)
    
    #average sims for lags graph   
    
    def plot():
        from matplotlib import pyplot as plt
        x = np.arange(minLag, maxLag+1, 1)
        y = np.asarray([np.nanmean(i) for i in sims_by_lag])
        err = np.asarray([sterr(i) for i in sims_by_lag])
        plt.errorbar(x, y, yerr=err, linewidth=3, 
                     color='k', marker='o', linestyle='-')
        plt.xlabel('Lag')
        plt.ylabel('Average similarity')
        plt.xlim([minLag-2, maxLag+2])
    plt_std_graph(plot, fname='lag_av_sim')
    
    
def zscore(data):
    import numpy as np
    return ((data - np.nanmean(data)) 
        / np.nanstd(data))
        

def twodify(trialnums, onedarray):
    import numpy as np
    twodarray = []
    row = []
    for i in range(len(trialnums)):
    
        if i == 0 or trialnums[i] == trialnums[i-1]:
            row.append(onedarray[i])
        else:
            while len(row)<24:
                row.append(np.nan)
            twodarray.append(row)
            row = []
    return np.asarray(twodarray)


def zeroone_norm(data):
    return (data-min(data))/(max(data)-min(data))


def add_subplot_axes(ax, x, y, axisbg='w'):
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform([x, y])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= .3
    height *= .3
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= .3**0.5
    y_labelsize *= .3**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
    
    
def irt_dist(data, data2=None, logged=False, bcx=False, cap=8, legend=False,
             negcap=-4, sv=True, colors=False, savedir=None,
             bin_num=200, fitto='norm'):
    import os
    os.chdir(get_scriptdir())
    import numpy as np
    
    irts = data['irts']
    irts = irts [~np.isnan(irts)]
    if data2:
        irts2 = data2['irts']
    
    if logged:
        irts = np.log(irts)
    if bcx:
        irts = data['bcx']
        if data2:
            irts2 = data2['bcx']
        
    if cap and not np.isnan(cap): 
        irts = irts[irts<=cap]
    if negcap and not np.isnan(negcap): 
        irts = irts[irts>=negcap]
        
    #print max(irts)
    #irts = np.log(irts)
    
    hist, bins = np.histogram(irts, bins=bin_num, normed=1)
    #histn, bins = np.histogram(np.random.normal(size=len(irts[~np.isnan(irts)])), bins=bin_num)
    if data2:
        hist2, bins = np.histogram(irts2[~np.isnan(irts2)], bins = bins)
    
    def plot():
        from matplotlib import pyplot as plto
        #plto.clf()
        plto.rcParams.update({'font.size': 16})
        plt = plto.subplot()
        
        
        maxi = max(hist/np.float64(irts.size))
        maxi += maxi
        label = 'Actual Data'
        if data['model']:
            label = 'Model Data'
        color = ('b'*colors) + ('k'*(not colors))
        x = bins[:-1]
        y = hist
        plt.plot(x, y, linewidth=3, color=color, label=label)
        
        
        letter = 'A'
        if bcx:
            letter = 'B'
        plt.text(.02, .93, '%s.' % letter, transform=plt.axes.transAxes, size='large', weight='bold')
        
        import scipy.stats as stats
        
        color = 'grey'
        if fitto == 'gamma':
            a, b, c = stats.gamma.fit(irts)
            plt.plot(x, stats.gamma.pdf(x, a, b, c), linewidth=3, linestyle='--', color=color, label='Gamma Fit')
        elif fitto == 'norm':
            mu, sigma = stats.norm.fit(irts)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=3, linestyle='--', color=color, label='Normal Fit')
        elif fitto == 'weibull_min':
            a, b, c = stats.weibull_min.fit(irts)
            plt.plot(x, stats.weibull_min.pdf(x, a, b, c), linewidth=3, linestyle='--', color=color, label='Weibull Fit')
        
        if fitto:
            x = .72
            y= .51

            if bcx:
                x = .15
            
            subplot = add_subplot_axes(plt, x, y)
            r2 = probplot(irts, dist=fitto, plot=subplot, fsize=8)
                 
            y = plt.get_ylim()[1] * .2
            x = plt.get_xlim()[1] * .6
            
            if bcx:
                x -= 5
            if logged:
                x -= .5
            
            plt.text(x, y, "$R^2=%1.4f$" % r2)
            
        if data2:
            linestyle = '-'+('-'*(not colors))
            color = ('grey'*colors) + ('k'*(not colors))
            plt.plot(x, hist2/np.float64(irts.size), 
                     linewidth=3, color=color, label='model', linestyle = linestyle)
                     
        xlabel = 'Inter-Response Time (s)'
        if bcx:
            xlabel = 'bcx(%s)' % xlabel
        if logged:
            xlabel = 'ln(%s)' % xlabel
            
        plt.set_xlabel(xlabel)
        plt.set_ylabel('Proportion of Recalls')

#        ax = plt.axes
#        box = ax.get_position()
#        ax.set_position([box.x0, box.y0, 
#                            box.width * 0.7, box.height])
        loc = 'upper right'
        axbox = plt.get_position()
        if bcx:
            loc = (axbox.x0 + 0, axbox.y1 - .083)
            
        leg = plt.legend(loc=loc)
        leg.get_frame().set_alpha(1)
            
        
        fig = plt.figure
        os.chdir(get_graphdir())
        
        gfname = 'irt_dist'
        if bcx:
            gfname += '-bcx'
        elif logged:
            gfname += '-log'
        else:
            gfname+='-'+fitto
        if data['model']:
            gfname = 'model_' + gfname

        
        if not sv:
            gfname = 'temp'
        
        fig.tight_layout()
        
        fig.set_size_inches(6.5, 5.5)
        fig.savefig(gfname+'.eps', format='eps')
        fig.savefig(gfname+'.png', format='png', dpi=200)
        fig.show()
        return fig
        fig.clf()        
        
    return plot()
    

def lambda_hist(act):
    from matplotlib import pyplot as plt
    import numpy as np
    def plot():
        plt.hist(act['lambda'], color='k', edgecolor='w', bins=np.arange(-1.5, -.2, .1))
        plt.xlabel('Lambda Bins')
        plt.ylabel('Frequency')
        plt.ylim(plt.ylim()[0], plt.ylim()[1]+1)
    plt_std_graph(plot, 'lambda_hist')
    
        
def similarity_dist(semlags, typ='LSA'):
    import numpy as np
    from matplotlib import pyplot as plt
    
    def plot():
        x, y = np.histogram(semlags, bins = np.arange(-.2, 1, .01))
        plt.plot(y[:-1], x.astype(float)/sum(x), 
                 color='k', linewidth=3)
        plt.xlabel(typ + ' semantic similarity')
        plt.ylabel('Frequency')
        
    plt_std_graph(plot, fname=typ+'_distribution')
    

def twodirts(flatirts, trialnums):
    import os
    os.chdir(get_scriptdir())
    from data_collection import fillArray
    import numpy as np
    irts = []
    row = [np.nan]
    for i in range(0, len(flatirts)):
        #print predicted[i], i,trialnums[i],Rs[i],invops[i]
        row.append(flatirts[i])
        if (i >= len(flatirts)-1 or
            trialnums[i+1] != trialnums[i]
            ):
            #print 'newrow'
            irts.append(
                    fillArray(24, np.nan, np.asarray(row))
                    [0:24])
            row = [np.nan]
    return np.asarray(irts)
        
        
def logbaseN(tolog, N):
    import numpy as np
    return np.log(tolog)/np.log(N)
        

def reverse_boxcox(boxcoxed, gamma):
    import numpy as np
    return ((np.asarray(boxcoxed)*gamma)+1)**(1/gamma)


def boxcox(toboxcox, lmbda):
    return (toboxcox ** lmbda -1)/lmbda
        

def probplot(x, sparams=(), dist='norm', fit=True, plot=None, fsize=15, supressdig2=True):
    # code taken from https://github.com/scipy/scipy/blob/v0.16.1/scipy/stats/morestats.py

    import numpy as np
    import scipy.stats as stats
    from scipy.stats.morestats import (_calc_uniform_order_statistic_medians,
     _parse_dist_kw, isscalar, sort)
    
    """
    Calculate quantiles for a probability plot, and optionally show the plot.
    Generates a probability plot of sample data against the quantiles of a
    specified theoretical distribution (the normal distribution by default).
    `probplot` optionally calculates a best-fit line for the data and plots the
    results using Matplotlib or a given plot function.
    """    
    
    x = np.asarray(x[~np.isnan(x)])
    _perform_fit = fit or (plot is not None)
    if x.size == 0:
        if _perform_fit:
            return (x, x), (np.nan, np.nan, 0.0)
        else:
            return x, x

    osm_uniform = _calc_uniform_order_statistic_medians(x.size)
    dist = _parse_dist_kw(dist, enforce_subclass=False)
    if sparams is None:
        sparams = ()
    if isscalar(sparams):
        sparams = (sparams,)
    if not isinstance(sparams, tuple):
        sparams = tuple(sparams)
        
    if sparams == ():
        sparams = dist.fit(x)

    osm = dist.ppf(osm_uniform, *sparams)
    osr = sort(x)
    if _perform_fit:
        # perform a linear least squares fit.
        slope, intercept, r, prob, sterrest = stats.linregress(osm, osr)
        #plot.plot(osm, slope*osm+intercept)

    if plot is not None:
        plot.plot(osm, osr, marker='', linestyle='-', linewidth=3, color='k', label='values')
        plot.plot(osm, slope*osm + intercept, color='grey', linestyle='--', linewidth=2, label='fit')
        
        try:
            plot.set_xlabel('Quantiles', fontsize=fsize)
            plot.set_ylabel('Ordered Values', fontsize=fsize)
            if max(plot.get_yticks())>1 or min(plot.get_yticks())<-1:
                plot.set_yticks(np.asarray(plot.get_yticks()).astype(int))
            if supressdig2:
                while max(plot.get_yticks()) >= 10:
                    plot.set_yticks(plot.get_yticks()[:-1])
                while min(plot.get_yticks()) <= -10:
                    plot.set_yticks(plot.get_yticks()[1:])
        except AttributeError:
            plot.xlabel('Quantiles', fontsize=fsize)
            plot.ylabel('Ordered Values', fontsize=fsize)
            if max(plot.yticks())>1 or min(plot.yticks())<-1:
                plot.yticks(np.asarray(plot.yticks()).astype(int))
            if supressdig2:
                while max(plot.yticks()) >= 10:
                    plot.yticks(plot.yticks()[:-1])
                while min(plot.yticks()) <= -10:
                    plot.yticks(plot.yticks()[1:])    
                
        
        #plot.legend(loc='best', fontsize=fsize-1)

    if fit:
        return r ** 2
        return (osm, osr), (slope, intercept, r)
    else:
        return osm, osr