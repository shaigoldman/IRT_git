import numpy as np
import collections
import regression as rg
import misc_funcs as mf
import sys
from matplotlib import pyplot as plt 
import data_collection as dc

def getFSmat(fname, irts=None, itemnos=None, 
             recalls=None, exclude_pirt=False,
             only_tenplus=False):
    
#    data = dc.get_data(fname)
#    actual_irts = data['irts']
#    clean_acts = []
#    for i in range(len(actual_irts)):
#        if len(actual_irts[i][~np.isnan(actual_irts[i])]) > 0:
#            clean_acts.append(i)
#    actual_irts = actual_irts[clean_acts]
    catagorized = False
    #exclude_pirt = True
    
    print 'catagorized is', catagorized
    print 'excl pirts is', exclude_pirt
    
    exclude = []
    if exclude_pirt:
        exclude = ['PREVIRT']
    actual_irts, pred_irts= rg.get_pred_1subj(fname, include_act=True, 
                                              exclude=exclude, only_tenplus=only_tenplus, 
                                              catagorized=catagorized)
    
    FSmat = (actual_irts > pred_irts).astype(float) #slows are 1s, fasts are 0
    FSmat[np.isnan(actual_irts)] = np.nan
    FSmat[np.isnan(pred_irts)] = np.nan
    
    return FSmat

    

def get_pdat(fname, buffrlen=3, exclude_pirt=True):
    probs = collections.OrderedDict([])
    irts = getFSmat(fname, exclude_pirt=exclude_pirt)
    for lis in irts:
        buffr = []
        for op in lis[~np.isnan(lis)]:
            buffr.append(int(op))
            if len(buffr) == buffrlen:
                key = ''.join(map(str,buffr))
                if not key in probs:
                    probs[key] = 0
                probs[key] += 1
                buffr = buffr[1:]
                
    probs['-'] = [1-np.nanmean(irts), np.nanmean(irts)]
    return probs


def analy_pdat_sstart(pdat, show=False):
    keys = np.asarray(pdat.keys())
    slowstarts = keys[np.asarray([keys[i][0] for i in range(len(keys))]) == '1']
    spdat = {}
    for i in slowstarts:
        spdat[i] = pdat[i]
    tot = float(sum([spdat[i] for i in spdat]))
    for i in spdat:
        spdat[i] = spdat[i] / tot
    rlabels = [get_rowlabel(spdat.keys()[i][1:]) for i in range(len(spdat.keys()))]
    clabels = ['P after Slow']
    data = np.transpose([[spdat[i] for i in spdat]])
    if show:
        mf.plotTable(data, clabels, rlabels, fname='temp')
    return (data, pdat['-'][0])

def get_rowlabel(numstr):
    lbl = ''
    if numstr == '':
        numstr = '-'
        
    for i in numstr:
        if i == '0':
            lbl += 'Fast'
        elif i == '1':
            lbl +='Slow'
        elif i == '-':
            lbl += 'Overall'
        else:
            lbl += 'err'
    return lbl
    

def analy_pdat(pdat):
    analys = collections.OrderedDict([])
    
    analys['-'] = [pdat['-'][0], '--']
    
    for i in pdat:
        if i == '-':
            continue
        key = i[:-1]
        try:
            analys[key] = [pdat[key+'0'], pdat[key+'1']]
        except KeyError:
            analys[key] = [1, 1]
    for i in analys:
        if i == '-':
            continue
        tot = float(analys[i][0] + analys[i][1])
        analys[i] = [analys[i][0]/tot, analys[i][0]/tot-analys['-'][0]]
    
    return analys


def plotTablewAnalys(analys):
    mf.plotTable([[i[1][0], i[1][1]] for i in analys.items()], ['pfast', 'deltaP'],
               [get_rowlabel(numstr) for numstr in analys.keys()],
               'PSFTable')


def personal_analy_all():
    for fname in mf.get_ltpFR2_filenames():
        print fname
#        analy_pdat_sstart(get_pdat(fname))
        analy = analy_pdat(get_pdat(fname))
        plotTablewAnalys(analy)


def analy_all(exclude_pirt=True, only_top_ten=False):
    all_analy = {}
    files = mf.get_ltpFR2_filenames()
    if only_top_ten:
        files = np.load('/home1/shai.goldman/IRT2017/Scripts/Resources/BestSubjs.npy')
    for fname in files:
        print fname,
        sys.stdout.flush()
        analy = analy_pdat(get_pdat(fname, exclude_pirt=exclude_pirt))
        for i in analy:
            if i in all_analy:
                all_analy[i].append(analy[i])
            else:
                all_analy[i] = [analy[i]]
    avg_analy = {}
    for i in all_analy:
        all_analy[i] = np.asarray(all_analy[i])
        
        if i == '-':
            avg_analy[i] = (int(np.nanmean(
                    all_analy[i][:,0].astype(
                            np.float64))*1000)/1000.,
                 '--',
                 int(mf.sterr(all_analy[i][:,0].astype(
                         np.float64))*1000)/1000.)
            continue
        
        avg_analy[i] = (int(np.nanmean(all_analy[i][:,0])*1000)/1000.,
                 int(np.nanmean(all_analy[i][:,1])*1000)/1000.,
                 int(mf.sterr(all_analy[i][:,0])*1000)/1000.)
        
    data = np.asarray([[i[1][0], i[1][1]] for i in avg_analy.items()])
    error = np.asarray([i[1][2] for i in avg_analy.items()])
    
   
    strdata = []
    for i in range(data.shape[0]):
        row = []
        for j in range(data.shape[1]):

            strng = unicode(data[i, j])
            
            if strng != u'--':
                strng += u' \xb1 '
                strng += unicode(error[i])
                if len(unicode(error[i])) < 5:
                    strng += '0'
                    
            row.append(strng)
        strdata.append(row)
    
    

    mf.plotTable(strdata, ['Pfast (act<pred)', 'deltaP (Pfast-Av)'],
           [get_rowlabel(numstr) for numstr in avg_analy.keys()],
           'PSFTable_Excluded=%s' % (str(exclude_pirt)))

def analy_all_sfirst(exclude_pirt=True, only_top_ten=False):
    analys = []
    overalls = []
    files = mf.get_ltpFR2_filenames()
    if only_top_ten:
        files = np.load('/home1/shai.goldman/IRT2017/Scripts/Resources/BestSubjs.npy')
    for fname in files:
        print fname,
        sys.stdout.flush()
        pdat = get_pdat(fname, exclude_pirt=exclude_pirt)
        analy, overall = analy_pdat_sstart(pdat)
        analys.append(analy)
        overalls.append(overall)
        
        
    analys = np.asarray(analys)
    overalls = np.asarray(overalls)
    
    data = np.nanmean(analys, axis=0) #avg_analys
    data = np.append(data, np.nanmean(overalls))
    data = data.tolist()
    for i in range(len(data)):
        data[i] = [data[i]]
    data = np.asarray(data)
    data = data[:-1] # get rid of the overall statistic
    
    error = mf.sterr(analys, axis = 0)
    
    strdata = []
    for i in range(data.shape[0]):
        row = []
        for j in range(data.shape[1]):

            strng = unicode(data[i, j])[:5]
            
            if strng != u'--':
                strng += u' \xb1 '
                strng += unicode(error[i, j])[:5]
                if len(unicode(error[i, j])) < 5:
                    strng += '0'
                    
            row.append(strng)
        strdata.append(row)
    
    keys = np.asarray(pdat.keys())
    slowstarts = keys[np.asarray([keys[i][0] for i in range(len(keys))]) == '1']
    spdat = {}
    for i in slowstarts:
        spdat[i] = pdat[i]
        
    rlabels = [get_rowlabel(spdat.keys()[i][1:]) 
        for i in range(len(spdat.keys()))]
    clabels = ['P after Slow']

    
    mf.plotTable(strdata, clabels, rlabels,
           'PSFTableSstart_Excluded=%s'%str(exclude_pirt))
    
    return data, error

def resid_color_map(colorful=False):
    act = np.load('/home1/shai.goldman/Chunking/Resources/data.npy').tolist()
    mod = np.load('/home1/shai.goldman/Chunking/Resources/moddata.npy').tolist()
    
    for R in range(24, 25):
        print R
        resids = (act['irts']-mod['irts'])[:, 1:]
        subjs = act['subject']
        
        listsizes = np.asarray([len(i[~np.isnan(i)])+1 for i in resids])
        resids = resids[listsizes==R] # the zed axis
        resids=resids[:, :R-1]
        subjs = subjs[listsizes==R]
    #    resids = resids[36:78]
    #    subjs=subjs[36:78]
        resids = np.vstack((np.nanmean(resids, axis=0), resids))
        subjs = np.append(000, subjs)
        newsubj_indecies = []
        for s, subj in enumerate(subjs):
            if s == 0 or subj != subjs[s-1]:
                newsubj_indecies.append(s)
        additive = 0
        for newsub in newsubj_indecies:
            if newsub == 0:
                continue
            for i in range(2):
                resids = np.insert(resids, newsub+additive, np.full(resids.shape[1], 0), axis=0)    
                additive += 1
        
        ops = np.asarray([np.arange(resids.shape[1]) for i in range(resids.shape[0])]) #the x axis
        trialnums = np.asarray([np.arange(resids.shape[0]) for i in range(resids.shape[1])]) # the y axis
        
    
        resids[resids>2] = .4; resids[resids<-2] = -.4
        if not colorful:
            resids[resids>0] = .4; resids[resids<0] = -.4
            resids[0, 0] = 1
            resids[1, 1] = -1
        def plot():
            myplt = plt
            if not colorful:
                myplt = plt.subplot(111)
                myplt.plot([1,2], color='r', linestyle='-', marker='', linewidth=3, label='fast')
                myplt.plot([1,2], color='b', linestyle='-', marker='', linewidth=3, label='slow')
            myplt.pcolor(ops, np.transpose(trialnums), resids, cmap='seismic_r')
            

            if not colorful:
                box = myplt.get_position()
                myplt.set_position([box.x0, box.y0, 
                             box.width * 0.7, box.height])
                myplt.legend(bbox_to_anchor=(1.05, 1), loc=2, 
                            borderaxespad=0.)
            else:
                myplt.colorbar(label='IRT Resids')
            plt.xlim([0,ops.max()])
            plt.ylim([0,trialnums.max()])
            plt.xlabel('Output Position')
            plt.ylabel('Completed Lists')
            plt.xticks([], [])
            plt.yticks([], [])
            for i in np.unique(np.where(resids==0)[0]):
                myplt.plot(ops[0], trialnums[:, i], color='k')
                myplt.plot(ops[0], trialnums[:, i]+1, color='k')
            myplt.setp(plt.axes().get_yticklabels(), visible=False)
    

        mf.plt_std_graph(plot, 'cmap_slowfast')

if __name__ == '__main__':
    if raw_input('continue (y/n)? ') == 'y':
        'No PIRT'
        analy_all_sfirst(True, False)
        'Now with PIRT'
        analy_all_sfirst(False, False)
        #analy_all(True)
        #analy_all(False)
        #resid_color_map()
        
        
def bargraph():
    
    data1, error1 = analy_all_sfirst(True, False)
    data2, error2 = analy_all_sfirst(False, False)
    
    #it comes out as FF, FS, SS, SF
    # we want FF, SS, FS, SF
    args = [0, 2, 1, 3]
    
    x = np.sort(args)
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    axs[0].bar(x, data1[args].flatten(), yerr=error1[args].flatten(), color='w', edgecolor='k')
    axs[0].set_title('Without PREVIRT')
    axs[1].bar(x, data2[args].flatten(), yerr=error2[args].flatten(), color='w', edgecolor='k')
    axs[1].set_title('With PREVIRT')
    
    for i in range(2):
        axs[i].set_ylim([.19, .31])
        axs[i].set_yticks([.2, .25, .3])
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(['FF', 'SS', 'FS', 'SF'])
        xlim = axs[i].get_xlim()
        axs[i].hlines(.25, xlim[0], xlim[1], linestyle='--', color='grey')
        axs[i].set_xlim(xlim)
    
    #x
    f.text(0.5, 0.04, '', ha='center')
    #y
    f.text(0.01, 0.5, 'Probability Following Slow Response', va='center', rotation='vertical')
    
    for fmt in ['png', 'eps']:
        f.savefig(mf.get_graphdir()+'/SFBar.'+fmt, format=fmt)
    
    f.show()
        

def list_elligible_subjs(reeval=False):
    if not reeval:
        try:
            return np.loadtxt(mf.get_resourcedir()+'/elligibles.txt', dtype='S')
        except IOError:
            list_elligible_subjs(True)
    mylist = []
    for fname in mf.get_ltpFR2_filenames():
        data = dc.get_data(fname)
        if data['Rs'][data['Rs']>=16].size >= 576/2:
            mylist.append(fname)
            print fname, 'elligible'
        else:
            print fname, 'not elligible'
    np.savetxt(mf.get_resourcedir()+'/elligibles.txt', mylist, fmt="%s")
    return mylist


def resid_graph_1subj(fname, plt=plt):
#    actual, predicted = rg.get_pred_1subj(fname, exclude=[], 
#                   include_act=True, only_tenplus=False,
#                   catagorized=True, unbcx=False)
    data = dc.get_data(fname)
    vs = rg.get_vars(data, only_10plus=False)
    fit = rg.get_fit_wvars(vs, exclude=[], catagorized=False)
    r_squared = fit.rsquared
    resid_f = fit.resid
    
    xind, yind = vs['xind'], vs['yind']
    residuals = np.full(data['irts'].shape, np.nan)
    residuals[yind, xind] = resid_f

    #residuals = (actual-predicted)
    #residuals[residuals>1] = 1
    #residuals[residuals<1] = -1
    Rs = np.asarray([len(trial[~np.isnan(trial)]) + 3 for trial in residuals])
    Rs[Rs<=3] = 0
    residuals = residuals[Rs>=16]
    residuals = residuals[:, 3:16]
    resid_mean = np.nanmean(residuals, axis=0)
    resid_err = mf.sterr(residuals, axis=0)
    x = np.arange(3, 15)
    split_args = [[0]]
    for op in range(1, len(resid_mean)):
        if (np.absolute(resid_mean[op]) > resid_err[op]) == (np.absolute(resid_mean[op-1]) > resid_err[op-1]):
            split_args[-1].append(op)
        else:
            split_args.append([op-1, op])
            
    #plt.plot(x, resid_mean, linewidth=2, marker='o', color='grey', linestyle='--')
    split_args.reverse()
    for i in split_args:
        x = np.asarray(i) + 3
        y = resid_mean[i]
        yerr = resid_err[i]
        ispos = bool(np.absolute(y[-1]) > yerr[-1])
        c = ispos*'k' + (not ispos)*'w'
        plt.errorbar(x, y, yerr=yerr, marker='o', linewidth=3, markerfacecolor=c, color='k')
    
    xlim = plt.get_xlim()
    plt.hlines(0, xlim[0], xlim[1], linestyle='--', linewidth=3)
    #plt.set_xlabel('Output Position')
    #plt.set_ylabel('Faster-Resdiual-Slower')
    
    
    plt.set_xlim(xlim)
    plt.set_xticks(np.arange(3, 17, 3))
    plt.set_ylim(-.29, .38)
    plt.set_xticks(np.arange(3, 16, 3))
    plt.text(.8, .88, str(r_squared)[:4], transform=plt.transAxes, ha='center')
    
def resids_all_above16():
#    for fname in list_elligible_subjs():
#        resid_graph_1subj(fname)
    
    subjs = list_elligible_subjs()
    
    numcol = 3
    numrow = 7
    
    f, axs = plt.subplots(numcol, numrow, sharex=True, sharey=True)
    f.set_size_inches(11, 8.5)
    for i in range(numcol):
        for j in range(numrow):
            fname = subjs[(i*numrow)+j]
            print fname, subjs.tolist().index(fname)+1, '/', len(subjs)
            resid_graph_1subj(fname, axs[i, j])
    
    #x
    f.text(0.5, 0.04, 'Output Position', ha='center')
    #y
    f.text(0.04, 0.5, 'Resdiual', va='center', rotation='vertical')
    
    for fmt in ['png', 'eps']:
        f.savefig(mf.get_graphdir()+'/Residual_tests.%s'%fmt, format=fmt)
    
    f.show()
    
    
def resids_all(act):
#    for fname in list_elligible_subjs():
#        resid_graph_1subj(fname)
    
    subjs = mf.get_ltpFR2_filenames
    
    numcol = 3
    numrow = 7
    
    f, axs = plt.subplots(numcol, numrow, sharex=True, sharey=True)
    f.set_size_inches(11, 8.5)
    for i in range(numcol):
        for j in range(numrow):
            fname = subjs[(i*numrow)+j]
            print fname, subjs.tolist().index(fname)+1, '/', len(subjs)
            resid_graph_1subj(fname, axs[i, j])
    
    #x
    f.text(0.5, 0.04, 'Output Position', ha='center')
    #y
    f.text(0.04, 0.5, 'Resdiual', va='center', rotation='vertical')
    
    for fmt in ['png', 'eps']:
        f.savefig(mf.get_graphdir()+'/Residual_tests.%s'%fmt, format=fmt)
    
    f.show()


    
    
    
    
    
    
    
    