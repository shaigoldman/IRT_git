from matplotlib import pyplot as plt
import numpy as np


def semcrl(data_r_item, data_r, data_t, LSA, BIN_NUM=10):
    all_val = []
    bin_total = [0] * BIN_NUM
    bin_count = [0] * BIN_NUM
    bin_val = [0] * BIN_NUM
    
    
    def chunkIt(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
    
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
    
        return out
    
    """find all values in LSA matrix"""
    for item1 in range(len(LSA)):
        for item2 in range(len(LSA[0])):
            all_val.append(LSA[item1][item2])
    all_val = np.sort(all_val)
    all_val = all_val[~np.isnan(all_val)]
    
    all_val = list(chunkIt(all_val, BIN_NUM))
    
    """helper"""
    def find_bin(lsa_val, bin):
        for index in range(len(bin)):
            if lsa_val >= bin[index][0] and lsa_val <= bin[index][len(bin[index]) - 1]:
                return index
    
    times = [[] for i in range(BIN_NUM)]
    
    def add_to_bin(lsa_val, bin, subj, sp):
        bin_val[find_bin(lsa_val, bin)] += lsa_val
        bin_count[find_bin(lsa_val, bin)] += 1
        bin_total[find_bin(lsa_val, bin)] += (data_t[subj][sp+1] - data_t[subj][sp])/ float(1000)
        if np.isnan(data_t[subj][sp+1] - data_t[subj][sp]):
            print 'aha', subj, sp
        times[find_bin(lsa_val, bin)].append((data_t[subj][sp+1] - data_t[subj][sp])/ float(1000))

    
    for subj in range(len(data_r)):
        for sp in range(len(data_r[0])):
            if sp + 1 < len(data_r[0]):
                if (data_r[subj][sp] > 0 and data_r[subj][sp+1] > 0 and 
                    not (np.isnan(data_t[subj][sp]) or np.isnan(data_t[subj][sp+1])) and
                    not (np.isnan(LSA[data_r_item[subj][sp] - 1][data_r_item[subj][sp + 1] - 1]))):
                        add_to_bin(LSA[data_r_item[subj][sp] - 1][data_r_item[subj][sp + 1] - 1], all_val , subj, sp)
    
    bin_mean = [0] * BIN_NUM
    crl = [0] * BIN_NUM
    std = [0] * BIN_NUM
    
    for index in range(BIN_NUM):
        total = 0
        bin_mean[index] = np.divide(bin_val[index], float(bin_count[index]))
        crl[index] = np.divide(bin_total[index], float(bin_count[index]))
        for num in range(bin_count[index]):
            total += (times[index][num] - crl[index]) ** 2
        std[index] = 1.96 * (np.math.sqrt(total / (float(bin_count[index])))) / float(np.sqrt(bin_count[index]))
    
    return bin_mean, np.asarray(crl)*1000, np.asarray(std)*1000
    
    
    print(bin_mean)
    print(crl)
    sem_crp, = plt.plot(bin_mean, crl, 'ro-', label='sem-crp')
    plt.errorbar(bin_mean, crl, yerr=std, linestyle="", marker="o", color="k")
    
    print(std)
    print("done")
    plt.axis([-0.05, 0.38, 0.0, 4.0])
    
    plt.xlabel('Semantic Relatedness')
    plt.ylabel('Conditional-Response Latency (sec)')

    plt.show()