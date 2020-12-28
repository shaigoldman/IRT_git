#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 09:36:27 2018

@author: shait
"""
import misc_funcs as mf


def R_crl(irts, graphit=True):
    import numpy as np
    from matplotlib import pyplot as plt
    Rs = np.asarray([len(trial[~np.isnan(trial)]) for trial in irts])
    #irts[:, :4] = np.nan
    irts_by_R = []
    errs = []
    x = np.arange(4, Rs.max()+1)
    for i in x:
        irts_by_R.append(np.nanmean(irts[Rs==i]))
        errs.append(mf.sterr(irts[Rs==i]))
        try:
            if not irts[Rs==i]:
                irts_by_R.append(np.array(0))
        except ValueError:
            pass
    def plot():
        plt.errorbar(x, irts_by_R, yerr=errs, linewidth=3, color='k', marker='o')
        plt.xlabel('Total Recalls')
        plt.ylabel('Inter-Response Time (s)')
    mf.plt_std_graph(plot, 'R_crl')
    
    
def vs_R_crl(np, plt, vs):
    irts_by_R = []
    for R in np.sort(np.unique(vs['R'])):
        irts_by_R.append([])
        
    irts_by_R
    for R in np.sort(np.unique(vs['R'])):
        irts_by_R[np.sort(np.unique(vs['R'])).tolist().index(R)].append(vs['irts'][vs['R']==R])
        
    irts_by_R
    plt.plot([np.nanmean(i) for i in irts_by_R])
    
    
def lag_crl(data, lag_num=23, graphit=True, savename='', sv=True, logged=False, bcx=False):
    
    recalls = data['recalls']
    irts = data['irts']
    if bcx:
        irts = data['bcx']
    subject = data['subject']
    
    import numpy as np
    subjects = np.unique(subject)
        
    result = [[0] * (2 * lag_num + 1) 
            for count in range(len(subjects))]
    for subject_index in range(len(subjects)):
        lag_bin_count = [0] * (2 * lag_num + 1)
        irt = [0] * (2 * lag_num + 1)

        for trial in range(len(subject)):
            if subject[trial] == subjects[subject_index]:
                for serial_pos in range(1, len(recalls[0])):
                    if (~np.isnan(irts[trial][serial_pos])):
                        
                        lag = recalls[trial][serial_pos].astype(np.int64) - recalls[trial][serial_pos - 1].astype(
                            np.int64)
                            
                        if 0 <= lag_num + lag <= 2 * lag_num:
                            lag_bin_count[lag_num + lag] += 1
                            irt[lag_num + lag] += (irts[trial][serial_pos])
                            
        for index in range(2 * lag_num + 1):
            if lag_bin_count[index] == 0:
                lag_bin_count[index] = 1
            result[subject_index][index] = irt[index] / float(lag_bin_count[index])
        #print(result[subject_index])
    result = np.asarray(result)

    result[:, lag_num] = np.nan

    if graphit:
        def plot():
            from matplotlib import pyplot as plt
            extra_space = 3
            ax = plt.axes()
            x = np.arange(-lag_num, lag_num+1+extra_space)
            if logged:
                x = np.log(np.absolute(x))
            y = np.nanmean(result, axis=0)
            y = np.insert(y, lag_num, [np.nan]*extra_space)
            err = mf.sterr(result, axis=0)
            err = np.insert(err, lag_num, [np.nan]*extra_space)
            ax.errorbar(x, y, 
                         yerr=err, linewidth=3, marker='o', color='k')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Inter-Response Time (s)')

            
            if logged:
                ax.set_xlim([-lag_num-1, lag_num+1])
            else:
                ax.set_xlim([-25, 25+extra_space])
                if not bcx and not data['model']:
                    ax.set_ylim([1.3, 6])
                ax.set_xticks([-23, -18, -12,  -6, -1, 
                  1+extra_space, 6+extra_space, 12+extra_space, 18+extra_space, 23+extra_space])
                
                ax.set_xticklabels(['-23', '-18', '-12',  '-6', '-1', '+1', '+6', '+12', '+18', '+23'])
            
            print (ax.get_ylim()[1], ax.get_xlim()[1])
        fname = savename + bool(data['model'])*'model_'+ logged*'logged_'+ 'lag_crl'
        if not sv:
            fname = 'temp'
        mf.plt_std_graph(plot, fname=fname, letter='B')
    err = mf.sterr(result, axis=0)
    return result, err;

def sem_crl(data, num_simBins=10, graphit=True, w2v=mf.get_w2v(),
           typ='W2V', savename='', sv=True):
    
    import os
    import numpy as np
    import scipy.stats as ss
    os.chdir(mf.get_scriptdir())
#    from semcrl import semcrl
#    bins, crl, err = semcrl(data['rec_itemnos'], 
#                            data['recalls'], data['rectimes'],
#                            w2v, num_simBins)

    ys = []
    for subj in np.unique(data['subject']):
        print (' subj: %s' % subj),
        thisdata = mf.get_sub_data(data, subj)
        irts, lags, sims = mf.get_flat_irts_lags_and_sims(thisdata)
        
        simbins = mf.flat_arr_to_bins(np.unique(w2v), num_simBins)       
        binned_sims = np.asarray([
                mf.which_bin(simbins, sim) for sim in sims])
        
        irts_by_sim = []
        for i in range(num_simBins):
            irts_by_sim.append(irts[binned_sims==i])
        
    
        irts_by_sim = np.asarray(irts_by_sim)
              
        
        irts_by_sim = np.asarray([i for i in irts_by_sim])
        ys.append(np.asarray([np.nanmean(i) for i in irts_by_sim]))
    y = np.nanmean(ys, axis=0)
    x = np.asarray([np.nanmean(i) for i in simbins])
    err = ss.sem(ys, axis=0)
    
    if graphit:
        def plot():
            from matplotlib import pyplot as plt
            plt.errorbar(x, y, yerr=err, 
                         linewidth=3, marker='o', 
                         color='k')
            plt.xlabel('Semantic Relatedness')
            plt.ylabel('Inter-Response Time (s)')
            plt.xticks([0, 0.05, .1, .15, .2, .25, .3])
            if not data['model']:
                plt.yticks([1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2])
            plt.xlim([-.02, .31])
            if not data['model']:
                plt.ylim([1.6, 3.4])
            #plt.xlim([-0.05, .37])
        fname = savename+bool(data['model'])*'model_'+'semcrl'
        if not sv:
            fname = 'temp'
        mf.plt_std_graph(plot, fname=fname, letter='C')
    
    return x, y, err


def opR_crl(data, minR=9, max_output=24, 
                    graphit=True, colors=True, savename='',
                    bcx=False):
    import numpy as np
    #irts = data['irts']
    
    irts_by_R_all_subj = []
    for subj in np.unique(data['subject']):
        irts = data['irts'][np.where(data['subject']==subj)]
        if bcx:
            irts = data['bcx'][np.where(data['subject']==subj)]
        Rs = np.asarray([len(trial[~np.isnan(trial)]) for trial in irts])
        irts[irts == np.inf] = np.nan
        
        if data['model']:
            Rs += 3
            irts = irts[:, 3:]
          
        irts_by_R = []
        for i in range(minR, max_output+1):
            irts_by_R.append(irts[Rs==i])
        
    
        irts_by_R = np.asarray(irts_by_R)
        irts_by_R = irts_by_R[0:len(irts_by_R):2]
    
   
        irts_by_R = np.array([irts_by_R[i][:, 1:] for i in range(len(irts_by_R))])
        irts_by_R_all_subj.append(np.asarray(
                [np.nanmean(irt[~np.isnan(irt[:,0])], axis=0) for irt in irts_by_R]))
    
    irts_by_R_all_subj = np.asarray(irts_by_R_all_subj)
    toGraph = np.nanmean(irts_by_R_all_subj, axis=0)
    errs = mf.sterr(irts_by_R_all_subj, axis=0)
    
    linestyles = []
    markers = []
    for i in np.arange(0, len(toGraph)+1, 2):
        linestyles.extend(['-', '--', '-.', ':'])
        markers.extend(['o', 's'])
    if graphit:
        from matplotlib import pyplot as plt
        def plot():
            ax = plt.subplot(111)
            for i in range(0,len(toGraph)):
                graph = toGraph[i]
                err = errs[i]
                
                x = np.arange(1,len(graph[~np.isnan(graph)])+1)
                
                y = graph[~np.isnan(graph)]
                error = err[~np.isnan(graph)]
                
                if colors:
                    ax.errorbar(x, y, yerr=error, linewidth=2, 
                                 color='k',
                                 marker=markers[i], linestyle=linestyles[i], 
                                 label=len(y)+1)
                else:
                    ax.errorbar(x, y, yerr=error, linewidth=2, 
                                 marker=markers[i], linestyle=linestyles[i], 
                                 label=len(y)+1)
                    
            plt.xlabel('Output Position')
            plt.ylabel('Inter-Response Time (s)')
            plt.xticks(range(1, max_output+1, 2))
            plt.tight_layout()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, 
                             box.width * 0.7, box.height])
            handles, labels = ax.get_legend_handles_labels() # remove the errorbars 
            handles = [h[0] for h in handles] 
            ax.legend(handles, labels, loc='center left', 
                      bbox_to_anchor=(1.0, 0.5), title='Total Recalls')
    
        nfname = 'opR_crl'
        
        if bool(data['model']):
            nfname = 'model_' + nfname
        mf.plt_std_graph(plot, fname=savename+nfname, tight=False, show=True, letter='A')
        
    return toGraph, errs


def SimbyLag_crl(data, w2v=mf.get_w2v(), simBins=5,
                    graphit=True, colors=True, savename=''):
    import numpy as np

    irts, lags, sims = mf.get_flat_irts_lags_and_sims(data)
    
    #define the lag bins
    minLag = 1
    maxLag = 6
    lags = np.absolute(lags)
    lags[lags==5] = 4
    lags[np.logical_and(lags>=5,lags<=8)] = 5
    lags[np.logical_and(lags>=9,lags<=12)] = 6
      
    binned_sims = mf.flat_arr_to_bins(np.unique(w2v), simBins)       
      
    irts_by_lag = []
    sims_by_lag = []
    for i in range(minLag, maxLag+1):
        irts_by_lag.append(irts[lags==i])
        sims_by_lag.append(sims[lags==i])
    

    irts_by_lag = np.asarray(irts_by_lag)
    sims_by_lag = np.asarray(sims_by_lag)

    
    #---now back to the actual graphs
    
    #irts_by_lag = irts_by_lag[0:len(irts_by_lag):2] #skip every other so we can read better
    #sims_by_lag = sims_by_lag[0:len(sims_by_lag):2] #skip every other so we can read better
    errs=[]   
    sizes=[]    
    
    for i in range(len(irts_by_lag)):
        simlvs = np.asarray([mf.which_bin(binned_sims, sim) for sim in sims_by_lag[i]])
        by_simlv = np.asarray([irts_by_lag[i][simlvs==binn] for binn in range(simBins)])
        irts_by_lag[i] = [np.nanmean(decible) for decible in by_simlv]
        errs.append([mf.sterr(np.asarray(decible)) for decible in by_simlv])
        sizes.append([len(np.asarray(decible)) for decible in by_simlv])
    
    irts_by_lag = np.asarray([i for i in irts_by_lag])
    errs=np.asarray(errs)
    
    toGraph = irts_by_lag
    print (toGraph)
    
    linestyles = []
    markers = []
    for i in np.arange(0, len(toGraph)+1, 2):
        linestyles.extend(['-', '--', '-.', ':'])
        markers.extend(['o', 's', '^'])
    if graphit:
        from matplotlib import pyplot as plt
        def plot():
            ax = plt.subplot(111)
            for i in range(0,len(toGraph)):
                graph = toGraph[i]
                err = errs[i]#np.zeros(errs[i].shape)#
                
                x = np.arange(1,len(graph[~np.isnan(graph)])+1)
                x = np.array([np.nanmean(simlv) for simlv in binned_sims])
                
                y = graph[~np.isnan(graph)]
                error = err[~np.isnan(graph)]
                
                if colors:
                    plt.errorbar(x, y, yerr=error, linewidth=2, 
                                 color='k',
                                 marker=markers[i], linestyle=linestyles[i])
                else:
                    plt.errorbar(x, y, yerr=error, linewidth=2, 
                                 marker=markers[i], linestyle=linestyles[i])
                    
            plt.xlabel('Semanic similarity')
            plt.ylabel('Inter-Response Time (s)')
            plt.ylim([1.1, 5.9])
            plt.tight_layout()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, 
                             box.width * 0.75, box.height])
            try:
                plt.legend(['1', '2', '3', '4-5', '6-8', '9-12'],
                        bbox_to_anchor=(1.05, 1), loc=2, 
                        borderaxespad=0., title='Lag')
            except IndexError:
                pass
        
        nfname = 'SimLag_crl'
        
        if bool(data['model']):
            nfname = 'model_' + nfname
        mf.plt_std_graph(plot, fname=nfname, tight=False, show=True, letter='D')
        
    
    return toGraph, errs    
    

def prevIrt_crl(data, num_irt_bins=10, num_pirts=3):
    import numpy as np
    irts = data['irts'][~np.isnan(data['irts'])]
    irt_bins = mf.flat_arr_to_bins(irts, num_irt_bins)
#    binned_irts = np.asarray([which_bin(irt_bins, irt) for irt in irts])
#    binned_irts = binned_irts.reshape(binned_irts.shape[0])
    
    irts_by_previrts = []
    for i in range(num_pirts+1):
        irts_by_previrts.append([[] for i in range(num_irt_bins)])
        
    
    for trial in range(data['irts'].shape[0]):
        for resp in range(1, data['irts'].shape[1]):
            if np.isnan(data['irts'][trial, resp]):
                continue
            for i in range(len(irts_by_previrts)):
                if resp>i:
                    irts_by_previrts[i][
                            mf.which_bin(irt_bins, data['irts'][trial, resp-i])
                            ].append(data['irts'][trial, resp])
        print (trial)

    x = np.asarray([np.nanmean(i) for i in irt_bins])
    
    ys = [np.asarray([np.nanmean(i) for i in irts_by_previrt]) for irts_by_previrt in irts_by_previrts]
    errs = [np.asarray([mf.sterr(i) for i in irts_by_previrt]) for irts_by_previrt in irts_by_previrts]
    
    styles = ['-', '--', ':', '-.'] * (len(ys)/2 + 1)
        
    def plot():
        from matplotlib import pyplot as plt
        ax = plt.subplot(111)
        plt.xlabel('Previous Inter-Response Time (s)')
        plt.ylabel('Inter-Response Time (s)')
        for i in range(1, len(ys)):
            ax.errorbar(x, ys[i], linewidth=2, marker='o',
                         label='Lag=%d'%i, linestyle=styles[i-1], color='k')
            ax.fill_between(x, ys[i]+errs[i], ys[i]-errs[i], alpha=.5, color='Grey')
        if len(ys)-1>3:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, 
                                 box.width * 0.7, box.height])
            ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        else:
            axbox = plt.axes().get_position()
            loc = (axbox.x0 + 0, axbox.y1 - .028 * (len(ys)+1))
            ax.legend(loc=2)
        #plt.show()
    mf.plt_std_graph(plot, fname='previrt_crl', tight=False, letter='')
    


def LagbySim_crl(data, w2v=mf.get_w2v(), num_simBins=5, minLag=-10, maxLag=10,
                    graphit=True, colors=False, savename=''):
    import numpy as np
    
    irts, lags, sims = mf.get_flat_irts_lags_and_sims(data)
      
    simbins = mf.flat_arr_to_bins(np.unique(w2v), num_simBins)       
    binned_sims = np.asarray([mf.which_bin(simbins, sim) for sim in sims])
    
    irts_by_sim = []
    lags_by_sim = []
    for i in range(num_simBins):
        irts_by_sim.append(irts[binned_sims==i])
        lags_by_sim.append(lags[binned_sims==i])
    

    irts_by_sim = np.asarray(irts_by_sim)
    lags_by_sim = np.asarray(lags_by_sim)
          
    errs=[]   
    sizes=[] 
    for i in range(len(irts_by_sim)):
        irts_by_lag = np.asarray([irts_by_sim[i][lags_by_sim[i]==lag] for lag in np.arange(minLag, maxLag+1)])
        irts_by_sim[i] = [np.nanmean(lag) for lag in irts_by_lag]
        errs.append([mf.sterr(np.asarray(lag)) for lag in irts_by_lag])
        sizes.append([len(np.asarray(lag)) for lag in irts_by_lag])
    
    irts_by_sim = np.asarray([i for i in irts_by_sim])
    errs = np.asarray(errs)
    
    toGraph = irts_by_sim
    
    linestyles = []
    markers = []
    for i in np.arange(0, len(toGraph)+1, 2):
        linestyles.extend(['-', '--'])
        markers.extend(['o', 's'])
    if graphit:
        from matplotlib import pyplot as plt
        def plot():
            ax = plt.subplot(111)
            for i in range(0,len(toGraph)):
                graph = toGraph[i]
                err = errs[i]#np.zeros(errs[i].shape)#
                
                x = np.arange(minLag,maxLag+1)
                
                y = graph
                error = err
                
                if colors:
                    plt.errorbar(x, y, yerr=error, linewidth=2, 
                                 color='k',
                                 marker=markers[i], linestyle=linestyles[i])
                else:
                    plt.errorbar(x, y, yerr=error, linewidth=2, 
                                 marker=markers[i], linestyle=linestyles[i])
                    
            plt.xlabel('Serial Lag')
            plt.ylabel('Inter-Response Time (s)')
            #plt.xticks(range(minLag, maxLag+1, 2))
            plt.tight_layout()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, 
                             box.width * 0.7, box.height])
            try:
                plt.legend(np.arange(num_simBins),
                        bbox_to_anchor=(1.05, 1), loc=2, 
                        borderaxespad=0., title='Similarity Bin')
            except IndexError:
                pass
            #plt.text(max_output+1.3, toGraph[-1][~np.isnan(toGraph[-1])][-3]-.3, 'Total Recalls')
        
        nfname = 'LagSim_crl'
        
        if bool(data['model']):
            nfname = 'model_' + nfname
        mf.plt_std_graph(plot, fname=nfname, tight=False, show=True)
        
    
    return toGraph, errs


def intrasess_crl(data, savename=''):
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    sessions = data['session']
    avged_irts = [[] for i in range(sessions.max())]

    for trial in range(0, irts.shape[0]):
        avged_irts[sessions[trial]-1].append(np.nanmean(irts[trial]))
    
    
    def plot():
        x = range(1,24)
        y = np.asarray([np.nanmean(i) for i in avged_irts])
        error = np.asarray([mf.sterr(np.asarray(i)) for i in avged_irts])
        
#        plt.plot(x, y, linestyle='-', marker='o',color='k')
#        plt.fill_between(x, y-error, y+error, color = '#969696')
        
        plt.errorbar(x, y, yerr=error, marker='o', 
                     color='k', linewidth=3)
        
        plt.ylabel('Inter-Response Time (s)')
        plt.xlabel('Session')
        plt.xlim([-.5,24])
        plt.ylim([plt.ylim()[0], 3.75])
        plt.xticks(np.append(1, np.append(np.arange(5, 24, 5), 23)))
        #plt.ylim([1200, 2500])
    
    mf.plt_std_graph(plot, fname=savename+(bool(data['model']) * 'model_')+'sess_crl', letter='A')
    
    
def block_crl(data, savename=''):
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    blocks = data['sestrialnums'] % 3
    avged_irts = [[] for i in range(blocks.max()+1)]

    for trial in range(0, irts.shape[0]):
        avged_irts[blocks[trial]].append(np.nanmean(irts[trial]))
    
    
    def plot():
        x = range(1,4)
        y = np.asarray([np.nanmean(i) for i in avged_irts])
        print (y)
        error = np.asarray([mf.sterr(np.asarray(i)) for i in avged_irts])
        
#        plt.plot(x, y, linestyle='-', marker='o',color='k')
#        plt.fill_between(x, y-error, y+error, color = '#969696')
        
        plt.errorbar(x, y, yerr=error, marker='o', 
                     color='k', linewidth=3)
        
        plt.ylabel('Inter-Response Time (s)')
        plt.xlabel('Session')
        #plt.xlim([-.5,24])
        #plt.ylim([plt.ylim()[0], 3.75])
        plt.xticks(x)
        #plt.ylim([1200, 2500])
    
    mf.plt_std_graph(plot, fname=savename+(bool(data['model']) * 'model_')+'block_crl', letter='')
    

def tri_crl(data, savename=''):
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    blocks = data['sestrialnums'] / 3
    avged_irts = [[] for i in range(blocks.max()+1)]

    for trial in range(0, irts.shape[0]):
        avged_irts[blocks[trial]].append(np.nanmean(irts[trial]))
    
    
    def plot():
        x = range(1,9)
        y = np.asarray([np.nanmean(i) for i in avged_irts])
        print (y, x)
        print (y)
        error = np.asarray([mf.sterr(np.asarray(i)) for i in avged_irts])
        
#        plt.plot(x, y, linestyle='-', marker='o',color='k')
#        plt.fill_between(x, y-error, y+error, color = '#969696')
        
        plt.errorbar(x, y, yerr=error, marker='o', 
                     color='k', linewidth=3)
        
        plt.ylabel('Inter-Response Time (s)')
        plt.xlabel('Session')
        #plt.xlim([-.5,24])
        #plt.ylim([plt.ylim()[0], 3.75])
        plt.xticks(x)
        #plt.ylim([1200, 2500])
    
    mf.plt_std_graph(plot, fname=savename+(bool(data['model']) * 'model_')+'block_crl', letter='')


def intersess_crl(data, savename=''):
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    sessions = data['session']
    sessions = data['sestrialnums']
    avged_irts = [[] for i in range(24)]

    for trial in range(0, irts.shape[0]):
        avged_irts[sessions[trial]].append(np.nanmean(irts[trial]))
    
    
    def plot():
        x = np.arange(1,25).reshape([3,8])
        y = np.asarray([np.nanmean(i) for i in avged_irts])
        y = y.reshape([3, 8])
        error = np.asarray([mf.sterr(np.asarray(i)) for i in avged_irts])
        error = error.reshape([3, 8])
        
#        plt.plot(x, y, linestyle='-', marker='o',color='k')
#        plt.fill_between(x, y-error, y+error, color = '#969696')
        
        for i in range(0,3):
            plt.errorbar(x[i], y[i], yerr=error[i], marker='o', 
                         color='k', linewidth=3)
        
        plt.ylabel('Inter-Response Time (s)')
        plt.xlabel('Trial')
        plt.xlim([0,25])
        plt.xticks(np.append(1, np.append(np.arange(5, 24, 5), 24)))
        #plt.ylim([1200, 2500])
    
    mf.plt_std_graph(plot, fname=savename+(bool(data['model']) * 'model_')+'intersess_crl', letter='B')

def intra_and_inter_sess_crl(data):
    tps = 24 # trials per session
    spp = 24 # sessions per paradigm
    wpl = 24 # words per list
    rpt = 3 # rewets per session
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    trials = data['trialnums']
    avged_irts = [[] for i in range(tps * spp)] 

    for lst in range(0, irts.shape[0]):
        avged_irts[trials[lst]-1].append(np.nanmean(irts[lst]))
    
    
    def plot():
        x = np.arange(1, tps*spp+1).reshape([rpt*wpl, tps*spp/(rpt*wpl)])
        y = np.asarray([np.nanmean(i) for i in avged_irts])
        y = y.reshape([rpt*wpl, tps*spp/(rpt*wpl)])
        error = np.asarray([mf.sterr(np.asarray(i)) for i in avged_irts])
        error = error.reshape([rpt*wpl, tps*spp/(rpt*wpl)])
        
#        plt.plot(x, y, linestyle='-', marker='o',color='k')
#        plt.fill_between(x, y-error, y+error, color = '#969696')
        
        plt.ylabel('Inter-Response Time (s)')
        plt.xlabel('Trial')
        plt.xlim([0,tps*spp+1-tps]) # the -tps is cuz we discount data from session 24
        
        for i in range(0, rpt*wpl):
            plt.errorbar(x[i], y[i], yerr=error[i]/(10.), marker='o', 
                         color='k', linewidth=3)
        ylim = plt.ylim()
        for i in range(1, spp):
            plt.plot([(i*tps)+.5]* 2, ylim, color='grey')
        plt.ylim(ylim)
        
        
        #plt.ylim([1200, 2500])
    
    mf.plt_std_graph(plot, fname=(bool(data['model']) * 'model_')+'intra-inter_sess_crl', width=50)

def wordlen_crl(data):
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    recwords = data['rec_items']
    wordlens = np.asarray(
            [[len(word) for word in trial] for trial in recwords])
    
    avged_irts = [[] for i in range(wordlens.max()-1)]
    
    for trial in range(0, irts.shape[0]):
        for word in range(1, irts.shape[1]):
            if ~np.isnan(irts[trial, word-1]):
                avged_irts[wordlens[trial, word-1]-2].append(
                        irts[trial, word])
    def plot():
        x = range(wordlens[wordlens>0].min(), wordlens.max()+1)[1:]
        y = np.asarray([np.nanmean(i) for i in avged_irts])#[1:]
        error = np.array([mf.sterr(np.asarray(i)) for i in avged_irts])#[1:]
        
        plt.errorbar(x, y, yerr=error, linestyle='-', marker='o', color='k', linewidth=3)
        #plt.fill_between(x, y-error, y+error, color = '#969696')
        plt.xlim(plt.xlim()[0]-1, plt.xlim()[1]+1)
        plt.xlabel('Word Length')
        plt.ylabel('Inter-Response Time (s)')
        
    mf.plt_std_graph(plot, fname=(bool(data['model'])*'model_')+'wlen_crl')
    

def wordfreq_crl(data, bin_num=10):
    import numpy as np
    from matplotlib import pyplot as plt
    
    irts = data['irts']
    rec_itemnos = data['rec_itemnos']
    Fs = mf.get_wfreq()

    
    wordfreqbins = np.sort(Fs)
    wordfreqbins = wordfreqbins[[~np.isnan(wordfreqbins)]]
    
    endpiece = wordfreqbins[
            int(wordfreqbins.size/bin_num)*bin_num:
            ]

    wordfreqbins = wordfreqbins[
            :int(wordfreqbins.size/bin_num)*bin_num
            ].reshape([bin_num, wordfreqbins.size/bin_num]).tolist()
    
    wordfreqbins.append(endpiece.tolist())
    for i in range(len(wordfreqbins)):
        wordfreqbins[i] = np.asarray(wordfreqbins[i])
    
    def whichbin(wordfreq, wordfreqbins):
        for i in range(len(wordfreqbins)):
            if wordfreq in wordfreqbins[i]:
                return i
        return np.nan
    
    recalledwordfreqs = np.asarray(
            [[whichbin(Fs[num], wordfreqbins) for num in lst] 
            for lst in rec_itemnos]).astype(float)
        
    recalledwordfreqs[np.isnan(irts)] = np.nan
    
    avged_irts = [irts[(np.where(recalledwordfreqs==i))] for i in range(len(wordfreqbins))]

    def plot():
        x = range(0, bin_num+1)
        y = np.asarray([np.nanmean(i) for i in avged_irts])
        error = np.asarray([mf.sterr(np.asarray(i)) for i in avged_irts])
        
#        plt.plot(x, y, linestyle='-', marker='o',color='k')
#        plt.fill_between(x, y-error, y+error, color = '#969696')
        
        plt.errorbar(x, y, yerr=error, marker='o', 
                     color='k', linewidth=3)
        
        plt.ylabel('Inter-Response Time (s)')
        plt.xlabel('Wordfreqbin')
        plt.xlim(plt.xlim()[0]-1, plt.xlim()[1]+1)
        
    mf.plt_std_graph(plot, fname='wfreq_crl')