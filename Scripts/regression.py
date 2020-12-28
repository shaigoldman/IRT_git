import scipy
from matplotlib import pyplot as plt
import numpy as np
import os 

try:
    import statsmodels.discrete.discrete_model as sd
except:
    pass

import data_collection as dc
import misc_funcs

import sys
import collections
import glob


# Lets find a statistical model for interresponse times.
def get_pirt_string(lag):
    return 'PREVIRT%i'%lag
    #return 'IRT'+misc_funcs.subscript(lag, neg=True)
def get_pirt_ucode(lag):
    #return 'IRT-%i'%lag
    return 'IRT'+misc_funcs.subscript(lag, neg=True)

def param_names():
    return ['OP', 'R', 'LAG', 'SIM', 
            'SESS', 'SIMxLAG', 
            get_pirt_string(1),
            get_pirt_string(2),
            'BLOCK', 'TRIAL'
            ]



def getTrialCorSize(trial):
    trial = trial[trial>0]
    recalled = []
    for i in trial:
        if not i in recalled:
            recalled.append(i)
    return len(recalled)


def get_vars(data, w2v = misc_funcs.get_w2v(), 
             absolute_lags=True, bcx=True, 
             resids=False, resid_exclude=[], resid_cat=True,
             only_10plus=False, num_pirts = 2):
    
    irts = data['irts']
    thresh = np.percentile(irts[~np.isnan(irts)], 73)
    recalls = data['recalls']
    rec_itemnos = data['rec_itemnos']
    #Rs = data['Rs']
    session = data['session']
    tnums = data['sestrialnums']
    
    
    lamda = np.nan
    if bcx:
        irts = data['bcx'] 
        lamda = data['lambda']
        
    if resids:
        those_vs = get_vars(data, w2v=w2v, absolute_lags=absolute_lags, resids=False,
                                     bcx=bcx, only_10plus=only_10plus, num_pirts=num_pirts)
        fit = get_fit_wvars(those_vs,
                            exclude = resid_exclude, catagorized=resid_cat)
        irts = np.full(data['irts'].shape, np.nan)
        
        irts[those_vs['yind'], those_vs['xind']] = np.asarray(list(fit.resid))
    
    vs = {}
    keys = ['OP', 'R', 'R2', 'lag', 'sim', 'sess', 'wordlen',
            'tri', 'block',
              'isshort', 'islong', 'trialnums', 
              'xind', 'yind']
    for i in keys:
        vs[i] = []
    vs['previrts'] = {}
    for i in range(1, num_pirts+1):
        vs['previrts'][i] = []
    
    newirts = []
    findecies = []
    
    counter = -1
    for trial in range(0, irts.shape[0]):
        on=False
        found=False
        for i, item in enumerate(irts[trial]):
            if np.isnan(item) and not on:
                on=True
            elif np.isnan(item) and on and not np.isnan(irts[trial][i-1]):
                r = i
                found=True
        if not found:
            r = len(np.where(~np.isnan(irts[trial]))[0]) + 1
        if not np.isnan(irts[trial][1]):
            r = len(np.where(~np.isnan(irts[trial]))[0]) + 1
        
        for i in range(1, len(irts[trial])):
            
            if np.isnan(irts[trial][i]):
                continue
            
            counter += 1
            
            if only_10plus and irts[trial][~np.isnan(irts[trial])].size +1 <= 16:
                continue
                #4continue
        
            #skip the first since they break the IRT pattern
            
            #IRTS ARE THE TIME BETWEEN CURRENT OP AND PREVIOUS
            if recalls[trial, i] < 0 or i < num_pirts+1 or np.isnan(irts[trial][i-num_pirts]):
                continue
            
            slag = misc_funcs.get_semrelat(
                    rec_itemnos[trial, i-1], 
                    rec_itemnos[trial, i], w2v)
            
            vs['OP'].append(i)
            vs['trialnums'].append(trial)
            

            
            vs['R'].append(r)
            vs['R2'].append(data['Rs'][trial])
            vs['sess'].append(session[trial])
            vs['isshort'].append(int(irts[trial, i]<thresh))
            vs['islong'].append(int(irts[trial, i]>=thresh))
            vs['tri'].append(tnums[trial]%3)
            vs['block'].append(tnums[trial]/8)

            for p in vs['previrts']:
                pos = i-p
                if pos < 1:
                    vs['previrts'][p].append(np.nan)
                else:
                    vs['previrts'][p].append(irts[trial, pos])
                
                
            vs['wordlen'].append(len(data['rec_items']))
            #worldlen.append()
            
            newirts.append(irts[trial, i])

            mylag = recalls[trial, i-1] - recalls[trial, i]
            if absolute_lags:
                mylag = np.absolute(mylag)
            vs['lag'].append(mylag)
            vs['sim'].append(slag)
            
            vs['xind'].append(i)
            vs['yind'].append(trial)
            findecies.append(counter)
    

    #irts = np.log(np.asarray(newirts)*1000)
    irts = irts[:, 1:]
    irts = irts[~np.isnan(irts)]
    irts = irts[findecies]

    for key in keys:
        vs[key] = np.asarray(vs[key])
    for key in vs['previrts']:
        vs['previrts'][key] = np.asarray(vs['previrts'][key])
    
    vs['lambda'] = lamda
    vs['irts'] = irts
    
    return vs

    
def get_vars_1subj(fname, model=False,
                    w2v = misc_funcs.get_w2v(), 
                    absolute_lags=True, bcx=True, 
                    only_10plus=False, num_pirts = 2):
    
    data = dc.get_data(fname, model=model)
    vs = get_vars(data, w2v=w2v,
                       absolute_lags=absolute_lags, bcx=bcx,
                       only_10plus=only_10plus, num_pirts=num_pirts)
    return vs


def extract_vs_to_betas(vs, normalized=False, 
                        exclude=[]):
    if normalized:
        zscore = scipy.stats.mstats.zscore
    else:
        def zscore(data): 
            return data    
    
    
    
    bs = collections.OrderedDict([])
    

#    bs['op1'] = zscore( 
#            (vs['OP'].astype(float)))
#    bs['op2'] = zscore(
#            (vs['OP'].astype(float)**2))
    
    bs['R'] = zscore(vs['R'].astype(float)/24)
    #bs['R2']= zscore(vs['R2'].astype(float)/24)
    
    bs['OP'] = zscore(
            1. / 
            ( 24-#vs['R'] -
            vs['OP'].astype(float) )
            ) 
   
    #bs['op_root'] = zscore(1. / ( vs['R'] -vs['OP'].astype(float)) ** .5) 
   # bs['R2'] = zscore(vs['R2'].astype(float)/24)
    
    bs['LAG'] = zscore(
            np.log(vs['lag']))
        
    
    bs['SIM'] = zscore(vs['sim'])
    bs['SESS'] = zscore(vs['sess'].astype(float)/24)
    #bs['WORDLEN'] = (zscore(vs['wordlen']))
    
    bs['SIMxLAG'] = zscore(vs['sim']*vs['lag'])
    
    bs['BLOCK'] = zscore(vs['block'])
    bs['TRIAL'] = zscore(vs['tri'])
    
    
    for p in vs['previrts']:
        bs[get_pirt_string(p)] = (zscore(vs['previrts'][p]))
    

    for i in bs:
        if ((i in exclude)
            or ('PREVIRT' in i and 'PREVIRT' in exclude) 
            or (i not in param_names() and 'IRT' not in i)):
            del bs[i]
        elif len(bs[i][~np.isnan(bs[i])]) < 10:
            print i, 'excluded for size'
            del bs[i]
        
    bs['irts'] = zscore(vs['irts'])
            
    return bs


def neg_bin_dist(fname):
    vs = get_vars_1subj(fname, logged=False)
    y, irts = extract_vs_to_betas(vs, normalized=False)
    model = sd.NegativeBinomial(irts, y, missing='raise')
    return model.fit_regularized()


def find_params_with_vars(vs, normalized=True, exclude=[], catagorized=True):
    vs = extract_vs_to_betas(vs, 
                normalized=normalized, exclude=exclude)
            
    from pandas import DataFrame
    dvars = DataFrame(data=vs)
    import statsmodels.formula.api as smf
    formulastr = 'irts ~'
    
    if catagorized:
        catagorized = ['R', 'R2', 'OP', 'op_root', 'LAG', 'SESS', 'BLOCK', 'TRIAL']
    else:
        catagorized = []
    
    for i in vs:
        if not i in exclude and i != 'irts':
            lbl = i
            if lbl in catagorized:
                lbl = 'C(%s)' % lbl
            formulastr += ' %s +' % lbl

    formulastr = formulastr[:-2]
    
    results = smf.ols(formula=formulastr, data=dvars).fit()
    
#    model = sm.OLS(irts, y, missing='drop')
#    results = model.fit()  # actually gets the regression variable values
    
    #results.summary()
    #print results.summary()

    return results


def get_fit_1subj(fname, model=False, normalized=True, exclude=[], catagorized=True, bcx=True):
    # finds the neccesary variables to make a regression
    # on a subject level
    nfname = fname + '_params'
    if normalized:
        nfname += '_normalized'
    vs = get_vars_1subj(fname, model=model, bcx=bcx)
    fit = find_params_with_vars(vs, normalized=normalized, 
                                exclude=exclude, catagorized=catagorized)
    return fit


def get_fit_wvars(vs, normalized=True, exclude=[], catagorized=True):
    # finds the neccesary variables to make a regression
    # on a subject level
    
    fit = find_params_with_vars(
            vs, normalized=normalized, exclude=exclude, catagorized=catagorized)    
    return fit


def get_pred_1subj(fname, data=None, exclude=[], 
                   include_act=False, only_tenplus=False,
                   catagorized=True, unbcx=True):
    
    if not data:
        data = dc.get_data(fname)
        
    vs = get_vars(data, only_10plus=only_tenplus)
    fit = get_fit_wvars(vs, normalized=False, exclude=exclude, catagorized=catagorized)
    
    xind, yind = vs['xind'], vs['yind']
    
    predicted_f = fit.predict()
    actual_f = fit.model.endog
    if unbcx:
        predicted_f = misc_funcs.reverse_boxcox(predicted_f, vs['lambda'])
        actual_f = misc_funcs.reverse_boxcox(actual_f, vs['lambda'])
    
    predicted = np.full(data['irts'].shape, np.nan)
    actual = np.full(data['irts'].shape, np.nan)
    
    predicted[yind, xind] = predicted_f
    actual[yind, xind] = actual_f
    if include_act:
        return actual, predicted
    return predicted

    
def scatter_1subj(fname):
   # vs = get_vars_1subj(fname)
#    actual_irts = dc.get_data(fname)['irts']
#    actual_irts[:, 0] = np.nan
##    actual_irts = scipy.stats.boxcox(
##            actual_irts[~np.isnan(actual_irts)])[0]
##    actual_irts = actual_irts[::]
##    actual_irts = data['logged_irts']
#    
#    pred_irts = np.asarray(get_pred_1subj(fname)).flatten()
#    actual_irts = actual_irts.flatten()
#    actual_irts = actual_irts[~np.isnan(pred_irts)]
#    pred_irts = pred_irts[~np.isnan(pred_irts)]
#

    fit = get_fit_1subj(fname)
    actual_irts = fit.model.endog
    pred_irts = fit.predict()

#    try:
#        actual_irts = scipy.stats.boxcox(actual_irts)[0]
#    except IndexError:
#        print 'Model Error'
#        return
#        
#    pred_irts = scipy.stats.boxcox(pred_irts)[0]
    
    
    def plot():
#        for OP in range(0, len(actual_irts)):
#                plt.plot(actual_irts[OP],
#                         pred_irts[OP],
#                         marker='.', linestyle=' ',
#                         color = colors[vs['lag'][OP]])
#                
        plt.plot(actual_irts, pred_irts,
                 marker='.', linestyle=' ', 
                 color='#969696', alpha=.8) 
        lim = plt.ylim()
        plt.xlabel('Actual IRTs')
        plt.ylabel('Predicted IRTs')
        
        plt.plot(actual_irts, actual_irts, marker='', 
                linewidth=3, color='k')
        plt.ylim(lim)
    misc_funcs.plt_std_graph(plot, fname+'_ScatteredAvP', 
                             dir=misc_funcs.get_graphdir()+'/scatterAPs')
    
    
def find_all_params(reevaluate=True, normalized=True, 
                    exclude = [], catagorized=True, 
                    bcx=True):
    data = {}
    al_pars = []
    al_pvals = []
    subjs = []
        
    if reevaluate:

        data['r2'] = [] 
        data['fstat_pvals'] = []
        for fname in misc_funcs.get_ltpFR2_filenames():
            print fname, 
            sys.stdout.flush()
            try:
                vs = get_vars_1subj(fname, bcx=bcx)
                results = get_fit_wvars(vs, 
                                     normalized=normalized, exclude=exclude,
                                     catagorized=catagorized)
                fstat_pvals = []
                
                mainfit = get_fit_wvars(vs, exclude=[],
                                     catagorized=catagorized)
                for param in param_names():                
                    exl_fit = get_fit_wvars(vs, exclude=[param],
                                     catagorized=catagorized)
                    fstat_pvals.append(mainfit.compare_f_test(exl_fit)[1])
            except ValueError:
                print 'ValueError'
                continue
            except IndexError:
                print 'IndexError'
                continue
                #print results.params
            al_pars.append(results.params)
            al_pvals.append(results.pvalues)
            subjs.append(fname)
            data['r2'].append(results.rsquared)
            data['fstat_pvals'].append(fstat_pvals)
            print results.rsquared,
    
    data = {'params': al_pars, 'pvalues': al_pvals, 
            'r^2': data['r2'], 'subj':subjs, 
            'fstat_pvals': data['fstat_pvals']}    
    
    return data

def get_subj_from_fdata(fdata, subj):
    sdata={}
    for i in fdata:
        try:
            if fdata[i].shape[0]== fdata['subject'].size:
                sdata[i] = fdata[i][fdata['subject']==subj]
            elif fdata[i].size > 70: #lambda val
                sdata[i] = fdata[i][
                        np.unique(fdata['subject']).tolist(
                                ).index(subj)]
        except AttributeError:
            sdata[i] = fdata[i] #model param
    return sdata

def find_all_params_wdata(data, normalized=True, exclude = [], 
                          catagorized=False, bcx=True):
    pdata = {}
    al_pars = []
    al_pvals = []
    subjs = []
    
    pdata['r2'] = [] 
    pdata['fstat_pvals'] = []
    allsubjs = np.unique(data['subject']).tolist()
    print str(len(allsubjs))+' Total'
    for fname in allsubjs:
        print 'LTP'+str(fname), allsubjs.index(fname), 
        sys.stdout.flush()
        fdata=get_subj_from_fdata(data, fname)
        
        try:
            vs = get_vars(fdata, bcx=bcx)
            results = get_fit_wvars(vs, 
                                 normalized=normalized, exclude=exclude,
                                 catagorized=catagorized)
            fstat_pvals = []
            
            mainfit = get_fit_wvars(vs, exclude=[],
                                 catagorized=catagorized)
            for param in param_names():                
                exl_fit = get_fit_wvars(vs, exclude=[param],
                                 catagorized=catagorized)
                fstat_pvals.append(mainfit.compare_f_test(exl_fit)[1])
        except ValueError:
            print 'ValueError'
            continue
        except IndexError:
            print 'IndexError'
            continue
            #print results.params
        al_pars.append(results.params)
        al_pvals.append(results.pvalues)
        subjs.append(fname)
        pdata['r2'].append(results.rsquared)
        pdata['fstat_pvals'].append(fstat_pvals)
        print results.rsquared

    pdata = {'params': al_pars, 'pvalues': al_pvals, 
            'r^2': pdata['r2'], 'subj':subjs, 
            'fstat_pvals': pdata['fstat_pvals']}    
    
    return pdata

def concatPs1subj(ps):
    pdict = {}
    for pkey in ps.keys():
        try:
            mykey = pkey[pkey.index("(")+1:pkey.index("]")]
        except ValueError:
            mykey = pkey
        if not mykey in pdict:
            pdict[mykey] = []
        pdict[mykey].append(ps[pkey])
    for i in pdict:
        pdict[i] = np.asarray(pdict[i])
        pdict[i] = np.nanmean(pdict[i])
    return pdict


def concatps(ps):
    pdict = {'pars': [], 'pvalues': []}
    for j in range(len(ps['subj'])):
        pdict['pars'].append([i[1] for i in 
                concatPs1subj(ps['params'][j]).items()])
        pdict['pvalues'].append([i[1] for i in 
            concatPs1subj(ps['pvalues'][j]).items()])
    for i in pdict:
        pdict[i] = np.asarray(pdict[i])
    pdict['subj'] = np.asarray(ps['subj'])
    return pdict
        

def graphParams(data, exclude=[]):
    lis = param_names()
    ind = []
    for i in lis:
        if not i in exclude:
            ind.append(i)
    lis = ind
    def plot():
        for param in range(len(lis)):
            for subj in range(len(data['subj'])):
                color = 'w'
                if data['pvalues'][subj][param] < .05:
                    color = 'k'
                plt.plot(param, data['pars'][subj][param], 
                    marker='o', color=color, markersize=4, markeredgecolor='k')
            mean = np.nanmean(data['pars'][:, param])
            x = np.arange(param-.1, param+.15,.1)
            y = [mean]*len(x)
            plt.plot(x, y, color='k', linewidth=1.4)
            y = max(data['pars'][:, param])+.1
            text = ''

            ttest = scipy.stats.ttest_ind(data['pars'][:, param], 
                [0]*len(data['pars'][:, param]))
            for k in [.001, .01, .05]:
                if ttest.pvalue < k:
                    text += ''#'*'
            #print ttest.pvalue
            fnt = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 12,
                }
            plt.text(param-.1, .2, text, fontdict=fnt)
        plt.plot(np.arange(-2,len(lis)+1), [0]*(len(lis)+3), 
                 linewidth=2, color ='k', linestyle = ':')
        plt.xticks(np.arange(0,len(lis)), lis, rotation='45')
        plt.xlim([-1, len(lis)])
        #plt.ylim(-.6,.3)
    misc_funcs.plt_std_graph(plot, 'model_parameters')

    ##----------R^2
    def plot():
        bins = np.arange(.55, .85, .01)
        hist, bins = np.histogram(data['r2'], bins)
        plt.xlabel('R^2 values')
        plt.ylabel('Frequency')
        plt.bar(bins[:-1], hist, 
                color ='k', edgecolor='w', 
                width = .01, align='edge')
    misc_funcs.plt_std_graph(plot, 'r^2_hist')
    print np.nanmean(data['r2'])
  

def graph_all_pars():
    ps = concatps(find_all_params(True, True))
    graphParams(ps)
    


def model_graphs_per_subj():
    for fname in misc_funcs.get_ltpFR2_filenames():
        data = dc.get_data(fname, model=True)
        savename = 'smg/'
        misc_funcs.op_and_R_vs_irt(data, savename=savename+'opR/'+fname+'_')
        misc_funcs.crl(data, savename=savename+'crl/'+fname+'_')
        misc_funcs.semcrl(data, bin_num=10, graphit=True, 
               savename=savename+'semcrl/'+fname+'_')
        misc_funcs.sess_crl(data, model=True, savename=savename+'sess/'+fname+'_')


def lilbits(act): 
    for i in np.arange(576, len(act['irts']), 5000):
        print i
        data = act.copy()
        for j in data:
            try:
                data[j] = data[j][0:i]
            except TypeError:
                continue
        misc_funcs.opR_crl(data)
#        misc_funcs.crl(data, savename=savename+'crl/'+str(i)+'_')
#        misc_funcs.semcrl(data, bin_num=10, graphit=True, 
#               savename=savename+'semcrl/'+str(i)+'_')
#        misc_funcs.lag_crl(data)
#    


def test1par(par, fname, catagorized=True):
    
    vs = get_vars_1subj(fname)
    exclude = param_names()
    exclude.remove(par)
    results = find_params_with_vars(vs, normalized=False, 
                        exclude=exclude, catagorized=catagorized)
    print results.rsquared
    print results.summary()
    pred_irts = results.predict()
    actual_irts = results.model.endog
    
    
    def plot():
#        for OP in range(0, len(actual_irts)):
#                plt.plot(actual_irts[OP],
#                         pred_irts[OP],
#                         marker='.', linestyle=' ',
#                         color = colors[vs['lag'][OP]])
#                
        plt.plot(actual_irts, pred_irts,
                 marker='.', linestyle=' ', 
                 color='#969696', alpha=.8) 
        
        ylim = plt.ylim()
        plt.xlabel('Actual IRTs')
        plt.ylabel('Predicted IRTs')

        
        plt.plot(actual_irts, actual_irts, marker='', 
                linewidth=3, color='k')
        plt.ylim(ylim)
        
        plt.title('Parameter: '+par)
    misc_funcs.plt_std_graph(plot, fname+par+'_ScatteredAvP', 
                             dir=misc_funcs.get_graphdir()+'/scatterAPs')
    

def test_Each_par_1subj(fname):
    for par in param_names():
        test1par(par, fname)


def scatters_eachsub():
    for fname in misc_funcs.get_ltpFR2_filenames():
        print fname
        scatter_1subj(fname)
        

def irt_dists_subjs():
    for fname in misc_funcs.get_ltpFR2_filenames():
        print fname
        fit = get_fit_1subj(fname)
        actual_irts = fit.model.endog
        pred_irts = fit.predict()
        misc_funcs.irt_dist({'irts':actual_irts, 'model':False})
        misc_funcs.irt_dist({'irts':pred_irts, 'model':True})


def get_bi_cutoff(irts):
    irts = irts[~np.isnan(irts)]
    rsquareds = []
    minirt = 0
    maxirt = irts.size+1000
    for step in [1000, 100, 10, 1]:
        print step
        for irt_thresh in np.arange(minirt, maxirt, step):
            rsquareds.append([irt_thresh, misc_funcs.qqplot(
                irts, cap=irt_thresh, typ = 'norm', 
                savestr = '')[1][2] ** 2])
        hotspot = rsquareds[np.argmax(np.asarray(rsquareds)[:, 1])][0]
        print hotspot
        minirt = hotspot-step
        maxirt = hotspot+step
    print rsquareds[np.argmax(np.asarray(rsquareds)[:, 1])]


def various_stuffs_for_tung_persub():
    for fname in misc_funcs.get_ltpFR2_filenames():
        print fname
        
        fit = get_fit_1subj(fname)
        actual_irts = fit.model.endog
        pred_irts = fit.predict()
        
        savedir = misc_funcs.get_graphdir()+'/fortung/'+fname
        def plot():
    #                
            plt.plot(actual_irts, pred_irts,
                     marker='.', linestyle=' ', 
                     color='#969696', alpha=.8) 
            lim = plt.ylim()
            plt.xlabel('Actual IRTs')
            plt.ylabel('Predicted IRTs')
            
            plt.plot(actual_irts, actual_irts, marker='', 
                    linewidth=3, color='k')
            plt.ylim(lim)
            
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        misc_funcs.plt_std_graph(plot, 'ScatteredAvP', 
                                 dir=savedir)
        misc_funcs.irt_dist({'irts':actual_irts, 'model':False}, savedir=savedir, sv=True)
        misc_funcs.irt_dist({'irts':pred_irts, 'model':True}, savedir=savedir, sv=True)


def table_w_params(data, sn1='', fs=True):    
    all_ps = data['params']
    all_pvals = data['pvalues']
    
    combined = collections.OrderedDict([])  
    meanedpardict = collections.OrderedDict([])  
    
    for subj in range(len(all_ps)):
        ps = all_ps[subj]
        pvals = all_pvals[subj]
    
        ps = ps.to_dict()
        pvals = pvals.to_dict()
        pitems = ps.items()
        pvalitems = pvals.items()
        for i in range(len(pitems)):
            pitems[i] = [pitems[i][0], pitems[i][1]]
            pvalitems[i] = [pvalitems[i][0], pvalitems[i][1]]
        pitems = np.asarray(pitems)
        pvalitems = np.asarray(pvalitems)
        
        rlabels = pitems[:, 0]
        print rlabels
        pars = pitems [:, 1]
        pvals = pvalitems[:, 1]
        lbls = []
        cnums = collections.OrderedDict([])
        pardict = collections.OrderedDict([])
        pvaldict = collections.OrderedDict([])
        
        myorder = ['Intercept']+param_names()
        for i in myorder:
            if not i in cnums:
                cnums[i] = []
                pardict[i] = []
                pvaldict[i] = []
            if not i in meanedpardict:
                meanedpardict[i] = []
        
        for i in range(len(rlabels)):
            if '(' in rlabels[i]:
                lbls.append(rlabels[i][rlabels[i].index('(')+1:rlabels[i].index(')')])
                cnums[lbls[-1]].append(float(rlabels[i][rlabels[i].index('.')+1:rlabels[i].index(']')]))
                pardict[lbls[-1]].append(pars[i])
                pvaldict[lbls[-1]].append(pvals[i])
                
            else:
                lbls.append(rlabels[i])
                pardict[lbls[-1]].append(pars[i])      
                pvaldict[lbls[-1]].append(pvals[i])      
                
        for i in cnums:
            cnums[i] = np.asarray(cnums[i])
            
        cnums['LAG'] = (np.e ** cnums['LAG'])
        cnums['R'] *= 24
        cnums['OP'] = ((24-cnums['OP']) ** -1)
        cnums['SESS'] *= 24    
        
        for i in cnums:
            if not i in combined:
                combined[i] = collections.OrderedDict([])
            for j in range(len(cnums[i])):
                if not cnums[i][j] in combined[i]:
                    combined[i][cnums[i][j]] = []
                combined[i][cnums[i][j]].append(pardict[i][j])
                
        for i in pardict:
            pvals = np.asarray(pvaldict[i]).astype(float)
            meanedpardict[i].append([np.nanmean(np.asarray(pardict[i]).astype(float)), 
                                     np.nanmean(np.asarray(pvaldict[i]).astype(float))])
                                     #(float(pvals[pvals<.05].size)/pvals.size)*100])
        
    for par in combined:
        for cat in combined[par]:
            combined[par][cat] = np.nanmean(np.asarray(combined[par][cat]).astype(float))
    
    del meanedpardict['Intercept']
    empties = []
    for par in meanedpardict:
        meanedpardict[par] = np.asarray(meanedpardict[par])
        meanedpardict[par] = meanedpardict[par][:, 0]
        if meanedpardict[par][~np.isnan(meanedpardict[par])].size == 0:
            empties.append(par)
    for par in empties:
        del meanedpardict[par]
        
        
    lis = meanedpardict.keys()
    print lis
    def plot():
        for param in lis:
            for subj in range(len(meanedpardict[param])):
                color = 'w'
                
                if fs:
                    pval = data['fstat_pvals'][subj][lis.index(param)]
                else:
                    pval = data['pvalues'][subj][lis.index(param)+1]
                
                if pval < .05:
                    color = 'k'
                plt.plot(lis.index(param), meanedpardict[param][subj], 
                        marker='o', color=color, markersize=4, markeredgecolor='k')
                
            mean = np.nanmean(meanedpardict[param])
            x = np.arange(lis.index(param)-.1, lis.index(param)+.15,.1)
            y = [mean]*len(x)
            plt.plot(x, y, color='k', linewidth=1.4)
            y = max(meanedpardict[param])+.1
            text = ''

            ttest = scipy.stats.ttest_ind(meanedpardict[param], 
                [0]*len(meanedpardict[param]))
            for k in [.001, .01, .05]:
                if ttest.pvalue < k:
                    text += ''#'*'
            #print ttest.pvalue
            fnt = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 12,
                }
            plt.text(lis.index(param)-.1, .2, text, fontdict=fnt)
        plt.plot(np.arange(-2,len(lis)+1), [0]*(len(lis)+3), 
                 linewidth=2, color ='k', linestyle = ':')
        mxticklabels=lis
        print mxticklabels, lis
        sys.stdout.flush()
        
        try: #replace pirt labels
            mxticklabels[mxticklabels.index(
                    get_pirt_string(1))]=get_pirt_ucode(1)
            mxticklabels[mxticklabels.index(
                    get_pirt_string(2))]=get_pirt_ucode(2)
                
        except ValueError: #excluding pirts for analyses
            pass
        
        plt.xticks(np.arange(0,len(mxticklabels)), 
                   mxticklabels, rotation='45', ha='right')
        plt.xlim([-1, len(lis)])
        plt.ylabel('Beta Coeficient')
        #plt.ylim(-.6,.3)
    misc_funcs.plt_std_graph(plot, 'beta_dist'+sn1)
        
        
    for par in meanedpardict:
        ttest = scipy.stats.ttest_1samp(meanedpardict[par], 0)
        #pvals = ttest[1]
        
        meanedpardict[par] = [np.nanmean(meanedpardict[par]), 
                     misc_funcs.sterr(meanedpardict[par]),
                     ttest[1]]
        
    rlabels = meanedpardict.keys()# ['OP', 'R', 'LAG', 'SIM', 'SESS', 'SIM x LAG', 'PREV_IRT', 'PREV_IRT_2']
    
    for par in meanedpardict:
        for i in [.05]:
            if meanedpardict[par][2] <= i:
                rlabels[meanedpardict.keys().index(par)] += '*'
    
    clabels = [u'Avg \xdf', u'SEM \xdf']
    mydata = np.asarray([i[1][0:2] for i in meanedpardict.items()])
    mydata = np.round(mydata, 10)
    #data = (data * 10000000).astype(int).astype(float)/10000000
    
    misc_funcs.plotTable(mydata, clabels, rlabels,
           'Beta_Table'+sn1)
    
    r2_hist(data['r^2'], sn=sn1)
    
    return meanedpardict


def r2_hist(r2s, sn=''):
    ##----------R^2
    r2s = np.asarray(r2s)
    def plot():
        bins = np.arange(np.round(r2s.min(), 2)-0.01, np.round(r2s.max(), 2)+0.01, .01)
        hist, bins = np.histogram(r2s, bins)
        plt.xlabel('R^2 values')
        plt.ylabel('Frequency')
        #plt.xticks(np.arange(r2.min(), r2.max(), .1))
        plt.bar(bins[:-1], hist, 
                color ='k', edgecolor='w', 
                width = .01, align='edge')
    misc_funcs.plt_std_graph(plot, 'r^2_hist'+sn)
    print np.nanmean(r2s), misc_funcs.sterr(r2s)
    return np.nanmean(r2s), misc_funcs.sterr(r2s)
    

def f_test_for_previrts(sample_size=1000):

    num_pirts = 3
    
    parameters = ['OP', 'R', 'LAG', 'SIM', 'SESS', 
                  'SIMxLAG']
    #parameters = ['OP']
    
    for i in range(num_pirts):
        parameters.append('PREVIRT%d' % (i+1))

    pvals = [[] for i in range(len(parameters))]
    fstats = [[] for i in range(len(parameters))]
    
    for fname in misc_funcs.get_ltpFR2_filenames():
        print fname
        vs = get_vars_1subj(fname, num_pirts=num_pirts)
        if len(vs['irts']) < 100:
            continue
        
        #get actual fstat
        mainfit = get_fit_wvars(vs, exclude=[])
        fits = [get_fit_wvars(vs, exclude=[i]) for i in parameters]
        for i in range(len(pvals)):
            fstat, pval, idk = mainfit.compare_f_test(fits[i])
            pvals[i].append(pval)
            fstats[i].append(fstat)
        
        #get permeated fstats
        
    
    pvals = np.asarray(pvals)
    
    def plot():
        x = np.arange(0, pvals.shape[0])
        y = [i[i<0.05].size/float(i.size) for i in pvals]
        plt.bar(x, y, color='w', width=.8, edgecolor='k')
        plt.xticks(x)
        xlim = plt.xlim()
        plt.plot(xlim, [.5] * 2, linestyle='--', linewidth=3)
        plt.xlim(xlim)
        plt.axes().set_xticklabels(parameters, rotation=45, ha='right')
        plt.xlabel('Parameter Removed')
        plt.ylabel('Percent significant loss')
        
    misc_funcs.plt_std_graph(plot, 'Significant_Loss_Measure')
    
def f_test_for_previrts2(sample_size=1000):

    num_pirts = 4
    
    
    parameters = ['OP', 'R', 'LAG', 'SIM', 'SESS', 
                  'SIMxLAG']
    #parameters = ['OP']
    
    for i in range(num_pirts):
        parameters.append('PREVIRT%d' % (i+1))
    
    
    os.chdir(misc_funcs.get_scriptdir())
    
    pref = 'test1_'
    permed_fstats = np.vstack([misc_funcs.readArrayFile(i) for i in glob.glob(pref+'permed_fstats*')])
    os.chdir(misc_funcs.get_scriptdir())
    fstats = np.loadtxt(glob.glob(pref+'fstats*')[6])
    sample_size = permed_fstats.shape[0]
    permed_fstats = np.sqrt(permed_fstats) 
    fstats = np.sqrt(fstats) 
    #permed_fstats = np.transpose(permed_fstats)
    #permed_fstats = np.nanmean(np.transpose(permed_fstats), axis=0)
    
    plt.rcParams.update({'font.size': 8})
    f, axs = plt.subplots(len(parameters)/2, 2)
    f.set_size_inches(6.5, 5.5)
    for i in range(len(parameters)):
        tohist = permed_fstats[:, i, :].flatten()
        hist, bins = np.histogram(tohist[~np.isnan(tohist)], bins=int(sample_size/20), normed=1)
        axs[i/2, i%2].plot(bins[:-1], hist)
        axs[i/2, i%2].tick_params(direction='in')
        this_fstat = np.nanmean(fstats[i])
        ylim = axs[i/2, i%2].get_ylim()
        axs[i/2, i%2].vlines(this_fstat, ylim[0], ylim[1]+.5)
        axs[i/2, i%2].text(.03, .8, parameters[i], transform=axs[i/2, i%2].transAxes, weight='bold')
        pval = permed_fstats[:, i, :][permed_fstats[:, i, :]>this_fstat].size/float(permed_fstats[:, i, :].size)
        axs[i/2, i%2].text(.6, .3, 'P=%f' % pval, transform=axs[i/2, i%2].transAxes)
        axs[i/2, i%2].set_ylim(ylim[0], ylim[1]+.5)
    #axs[-1, -1].set_visible(False)
    f.savefig(misc_funcs.get_graphdir()+'/fstat_perms.eps', format='eps')
    f.savefig(misc_funcs.get_graphdir()+'/fstat_perms.png', format='png')
    f.show()
        
    
    
def cat_vs_nocat():
    #CAT
    cat_params = find_all_params(reevaluate=True, normalized=True, exclude = ['PREVIRT', 'SESS', 'BLOCK', 'TRIAL'], catagorized=True)
    table_w_params(cat_params, sn1='cat')
    no_cat_params = find_all_params(reevaluate=True, normalized=True, exclude = ['PREVIRT', 'SESS', 'BLOCK', 'TRIAL'], catagorized=False)
    table_w_params(no_cat_params, sn1='nocat')
    
    cat_paramsp = find_all_params(reevaluate=True, normalized=True, exclude = ['SESS'], catagorized=True)
    table_w_params(cat_paramsp, sn1='catwp', sn2='catwp')
    no_cat_paramsp = find_all_params(reevaluate=True, normalized=True, exclude = ['SESS'], catagorized=False)
    table_w_params(no_cat_paramsp, sn1='nocatwp', sn2='nocatwp')

    
def two_stage_mod(fdata, catagorized=False):
    data = {}
    al_pars = []
    al_pvals = []
    subjs = []
    
    data['r2'] = [] 
    data['fstat_pvals'] = []
    
    for subj in np.unique(fdata['subject']):
        
        dat = get_subj_from_fdata(fdata, subj)
        
        print subj, 
        sys.stdout.flush()
        try:
            vs = get_vars(dat, resids=True, 
                                resid_exclude=['SESS', 'BLOCK', 'TRIAL'], resid_cat=catagorized)
            results = get_fit_wvars(vs, 
                                 normalized=True, exclude=['OP', 'R', 'LAG', 'SIM', 'SIMxLAG', 'PREVIRT'],
                                 catagorized=catagorized)
            fstat_pvals = []
            mainfit = get_fit_wvars(vs, exclude=[], catagorized=catagorized)
            for param in param_names():                
                exl_fit = get_fit_wvars(vs, exclude=[param], catagorized=catagorized)
                fstat_pvals.append(mainfit.compare_f_test(exl_fit)[1])
        except ValueError:
            print 'ValueError'
            continue
        except IndexError:
            print 'IndexError'
            continue
            #print results.params
        al_pars.append(results.params)
        al_pvals.append(results.pvalues)
        subjs.append(fname)
        data['r2'].append(results.rsquared)
        data['fstat_pvals'].append(fstat_pvals)
        print results.rsquared,
    
    data = {'params': al_pars, 'pvalues': al_pvals, 
            'r^2': data['r2'], 'subj':subjs, 
            'fstat_pvals': data['fstat_pvals']}    
    
    table_w_params(data, sn1='_twostage', fs=False)

    
if __name__ == '__main__':  
#    test1par('OP', '_LTP325')
    #f_test_for_previrts2(sample_size=1000)
    data = np.load(misc_funcs.get_resourcedir()+'/data.npy').tolist()
    if raw_input('Procede with main (y/n)? ') == 'y':
#            #graphParams(find_all_params(True, True))
            print 'FULL'
            params = find_all_params_wdata(data, 
                          normalized=True, exclude = ['BLOCK', 'TRIAL'], 
                          catagorized=False, bcx=True)
            table_w_params(params, sn1='_full', fs=False)
            
            params = find_all_params_wdata(data, normalized=True, 
                          exclude = ['BLOCK', 'TRIAL', 'SESS', 'PREVIRT'], 
                          catagorized=False, bcx=True)
            table_w_params(params, sn1='_base', fs=False)
            
            params = find_all_params_wdata(data, normalized=True, 
                          exclude = ['BLOCK', 'TRIAL', 'SESS'], 
                          catagorized=False, bcx=True)
            table_w_params(params, sn1='_pirt', fs=False)
                #pirt_cat // justbase_cat // pirt_nocat // justbase_nocat
            two_stage_mod(data, catagorized=False)
#    pass
