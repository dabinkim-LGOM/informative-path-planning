# !/usr/bin/python

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
from matplotlib import cm
import os
import pdb
import copy
import gpmodel_library as gplib 

from analysis_utils import *

######### MAIN LOOP ###########
if __name__ == '__main__':
    # seed_numbers = range(5100, 10000, 100)
    seed_numbers = range(0, 5000, 100)
    # seed_naive = range(0, 5000, 100)
    # seed_numbers = range(0, 2400, 100)
    # seed_numbers.remove(5300)
    print len(seed_numbers)
    # seed_numbers = [0, 100, 200, 400, 500, 700, 800, 900, 1000, 1200, 1300, 1400, 1600, 1700, 1800, 1900]
    seeds = ['seed'+ str(x) + '-' for x in seed_numbers]
    seeds_lawnmower = ['mower'+ str(x) + '/' for x in seed_numbers]

    SUFFIX  = 'FREE' # FREE or CLUTTERED
    TRIAL = 'plumes' # naive or plumes  or all

    if TRIAL == 'naive':
        fileparams = ['pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
                     'pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
                     'pathsetdubins-nonmyopicFalse-' + SUFFIX,
                     'pathsetdubins-nonmyopicFalse-' + SUFFIX]

        trials = ['naive', 'naive_value', 'naive', 'naive_value']
        labels = ['LOC-MCTS', 'VAL-MCTS', 'LOC-Myopic', 'VAL-Myopic.']
    elif TRIAL == 'all':
        fileparams = ['pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
                     'pathsetdubins-nonmyopicTrue-treebelief-' + SUFFIX,
                     'pathsetdubins-nonmyopicFalse-' + SUFFIX,
                     'lawnmower',
                     'pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
                     'pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
                     'pathsetdubins-nonmyopicFalse-' + SUFFIX,
                     'pathsetdubins-nonmyopicFalse-' + SUFFIX]

        trials = ['mes', 'mean', 'mes', 'lawnmower', 'naive', 'naive_value', 'naive', 'naive_value']
        labels = ['PLUMES', 'UCB-MCTS', 'UCB-MYOPIC', 'BOUSTRO.', 'LOC-MCTS', 'VAL-MCTS', 'LOC-Myopic', 'VAL-Myopic.']

    elif TRIAL == 'plumes':
	    fileparams = ['pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
			'pathsetdubins-nonmyopicTrue-treebelief-' + SUFFIX,
			'pathsetdubins-nonmyopicFalse-' + SUFFIX,
			'lawnmower']

	    trials = ['mes', 'mean', 'mean', 'lawnmower']
	    labels = ['PLUMES', 'UCB-MCTS', 'UCB-MYOPIC', 'BOUSTRO.']

    else:
        fileparams = ['pathsetdubins-nonmyopicTrue-treedpw-' + SUFFIX,
                    'pathsetdubins-nonmyopicTrue-treebelief-' + SUFFIX,
                    'pathsetdubins-nonmyopicFalse-' + SUFFIX]
        trials = ['mes', 'mean', 'mean']
        labels = ['PLUMES', 'UCB-MCTS', 'UCB-MYOPIC']

    file_start = 'ral-naive-trials'

    # path= '/home/vpreston/Documents/IPP/informative-path-planning/experiments/'
    # path= '/home/genevieve/mit-whoi/informative-path-planning/experiments/'
    # path = '/media/genevieve/WINDOWS_COM/IROS_2019/cluttered_experiments/experiments/'
    # path = '/media/genevieve/WINDOWS_COM/IROS_2019/experiments/'
    path = '/media/genevieve/WINDOWS_COM/iros_submission_simulations/reviewed_submission/free_world_experiments/'
    # path = '/media/genevieve/WINDOWS_COM/iros_submission_simulations/reviewed_submission/naive_experiments/'
    # path= '/home/genevieve/mit-whoi/informative-path-planning/final_naive_experiments/'

    # variables for making dataframes
    column_names = ['time', 'info_gain','aqu_fun', 'MSE', 'hotspot_error','max_loc_error', 'max_val_error', 
                        'simple_regret', 'sample_regret_loc', 'sample_regret_val', 'regret', 'info_regret',
                        'current_highest_obs', 'current_highest_obs_loc_x', 'current_highest_obs_loc_y',
                        'robot_loc_x', 'robot_loc_y', 'robot_loc_a', 'distance', 'max_value_info']

    #get the data files
    all_dfs = []
    all_sample_dfs = []
    all_props = []
    all_propsy = []
    all_labels = []
    dist_dfs = []
    dist_samples_dfs = []
    dist_props = []
    dist_propsy = []
    dist_ids = []
    dist_err_x = []
    dist_err_z = []
    dist_dist_x = []
    dist_dist_z = []
    dist_entropy_x = []
    dist_entropy_z = []

    max_val = []
    max_loc = []

    for param, label, trial in zip(fileparams, labels, trials):
        values = []
        samples = []
        # p_mean = []
        # p_mes = []
        # p_mean_samples = []
        # p_mes_samples = []

        print trial == 'naive'
        if trial == 'naive':
            other_trial = 'naive_value'
        else:
            other_trial = "xxx"

        print "Adding for:", param, label, trial
        for root, dirs, files in os.walk(path):
            for name in files:
                if 'metrics' in name and 'star' not in name and trial in root and other_trial not in root and param in root and SUFFIX in root or (('lawnmower' in param and 'lawnmower' in root)):
                   for s in seeds:
                       if s in root:
                           values.append(root+"/"+name)
                # else:
                #     print "Not adding:", root 

                if 'robot_model' in name and '#' not in name and ((trial in root and other_trial not in root and param in root and SUFFIX in root) or ('lawnmower' in param and 'lawnmower' in root)):
                    if 'lawnmower' in root:
                        for s in seeds_lawnmower:
                            if s in root + '/':
                                samples.append(root+"/"+name)
                                print root+'/'+name
                    else:
                        for s in seeds:
                            if s in root:
                                samples.append(root+"/"+name)
                                print root+'/'+name
                # if 'log' in name and (('mean' in root and 'UCB-MCTS' in param) or ('mes' in root and 'COMPOSIT' in param)) and param in root and 'FREE' in root:
                # if 'log' in name and param in root and SUFFIX in root:
                if 'log' in name and trial == trials[0] and trial in root and other_trial not in root and param in root and SUFFIX in root:
                    for s in seeds:
                        if s in root:
                            ls = []
                            temp = open(root+'/'+name, "r")
                            for l in temp.readlines():
                                if "max value" in l:
                                    ls.append(l)
                            max_val.append(float(ls[0].split(" ")[3]))
			
                            print root+'/'+name
                            # For Genevieve
                            try:
                                max_loc.append((float(ls[-1].split(" ")[7].split("[")[0]), float(ls[-1].split(" ")[9].split("]")[0])))
				print "Genevieve's"
				print max_loc[-1]	
				print s
                            # For Victoria
                            except:
                                max_loc.append((float(ls[0].split(" ")[6].split("[")[1]), float(ls[0].split(" ")[7].split("]")[0])))
				print "Victoria's"
				print max_loc[-1]	
      
	print "Length of max locs:", len(max_loc) 
	print "Length of max vals:", len(max_val) 
        if 'dpw' in param:
            old_values = copy.copy(values)
        
        if 'lawnmower' in param:
            values = copy.copy(old_values)

        print max_loc
        data = make_df(values, column_names)
        all_dfs.append(data)

        sdata, prop, propy, err_x, err_z, dist_x, dist_z, ent_x, ent_z = make_samples_df(samples, ['x', 'y', 'z'], max_loc = max_loc, max_val = max_val, xthresh = 1.5, ythresh = 1.0)
        all_sample_dfs.append(sdata)
        all_props.append(prop)
        all_propsy.append(propy)
        all_labels.append(label)

        if 'lawnmower' in param:
            dist_data, dist_sdata, d_props, d_propsy, ids, d_err_x, d_err_z, d_dist_x, d_dist_z, d_hx, d_hz = make_dist_dfs(values, samples, column_names, max_loc, max_val, ythresh = 3.0, xthresh = 1.5, dist_lim = 200.0, lawnmower = True)
        else:
            dist_data, dist_sdata, d_props, d_propsy, ids, d_err_x, d_err_z, d_dist_x, d_dist_z, d_hx, d_hz = make_dist_dfs(values, samples, column_names, max_loc, max_val, ythresh = 3.0, xthresh = 1.5, dist_lim = 200.0)

        dist_dist_x.append(d_dist_x)
        dist_dist_z.append(d_dist_z)
        dist_entropy_x.append(d_hx)
        dist_entropy_z.append(d_hz)

        dist_dfs.append(dist_data)
        dist_samples_dfs.append(dist_sdata)
        dist_props.append(d_props)
        dist_propsy.append(d_propsy)
        dist_ids.append(ids)
        dist_err_x.append(d_err_x)
        dist_err_z.append(d_err_z)


    if SUFFIX == 'FREE':
        # all_labels = ['PLUMES', 'UCB-MCTS', 'UCB-MYOPIC', 'BOUSTRO.', 'LOC-MCTS', 'VAL-MCTS', 'LOC-Myopic', 'VAL-Myopic.']#['frpd', 'frgd', 'frgo', 'frpo', 'my', 'plumes']
        labels = labels
        # all_labels = ['PLUMES', 'LAWNMOWER']#['frpd', 'frgd', 'frgo', 'frpo', 'my', 'plumes']
    elif SUFFIX == 'NOISE':
        # labels = ['LOC-MCTS', 'VAL-MCTS', 'LOC-Myopic', 'VAL-Myopic.']
        # labels = ['LOC-MCTS', 'VAL-MCTS', 'LOC-Myopic', 'VAL-Myopic.']
        labels = labels
    else:
        all_labels = ['PLUMES', 'UCB-MCTS', 'UCB-MYOPIC']#['frpd', 'frgd', 'frgo', 'frpo', 'my', 'plumes']

    # generate_stats(all_dfs, all_labels, ['distance', 'MSE', 'max_loc_error', 'max_val_error', 'max_value_info', 'info_regret'], 149, file_start + '_stats.txt')
    generate_dist_stats(dist_dfs, all_labels, ['distance', 'MSE', 'max_loc_error', 'max_val_error', 'max_value_info', 'info_regret'], dist_ids, file_start + '_dist_stats.txt')

    # generate_histograms(all_sample_dfs, all_props, all_labels, title='All Iterations', figname=file_start, save_fig=False)

    generate_histograms(dist_samples_dfs, dist_props, all_labels, title='200$m$ Budget X Samples', figname=file_start, save_fig=False)
    generate_histograms(dist_samples_dfs, dist_propsy, all_labels, title='200$m$ Budget Y Samples', figname=file_start, save_fig=False)

    generate_histograms(dist_samples_dfs, dist_err_x, all_labels, title='200$m$ Budget X Error', figname=file_start, save_fig=False, ONLY_STATS = True)
    generate_histograms(dist_samples_dfs, dist_err_z, all_labels, title='200$m$ Budget Z Error', figname=file_start, save_fig=False, ONLY_STATS = True)

    generate_histograms(dist_samples_dfs, dist_dist_x, all_labels, title='200$m$ Budget X Star Dist', figname=file_start, save_fig=False, ONLY_STATS = True)
    generate_histograms(dist_samples_dfs, dist_dist_z, all_labels, title='200$m$ Budget Z Star Dist', figname=file_start, save_fig=False, ONLY_STATS = True)

    generate_histograms(dist_samples_dfs, dist_entropy_x, all_labels, title='200$m$ Budget X Star Entropy', figname=file_start, save_fig=False, ONLY_STATS = True)
    generate_histograms(dist_samples_dfs, dist_entropy_z, all_labels, title='200$m$ Budget Z Star Entropy', figname=file_start, save_fig=False, ONLY_STATS = True)

    # # def planning_iteration_plots(dfs, labels, param, title, end_time=149, d=20, plot_confidence=False, save_fig=False, fname='')
    # planning_iteration_plots(all_dfs, all_labels, 'MSE', 'Averaged MSE', 149, len(seeds), True, False, file_start+'_avg_mse.png')
    # planning_iteration_plots(all_dfs, all_labels, 'max_val_error', 'Val Error', 149, len(seeds), True, False, file_start+'_avg_rac.png')
    # planning_iteration_plots(all_dfs, all_labels, 'max_loc_error', 'Loc Error', 149, len(seeds), True, False, file_start+'_avg_ireg.png')

    # (dfs, sdfs, labels, param, title, dist_lim=150., granularity=10, d=20, plot_confidence=False, save_fig=False, fname=''):
    distance_iteration_plots(dist_dfs, dist_ids, all_labels, 'MSE', 'Averaged MSE', 200., 100, len(seeds), True, False, '_avg_mse_dist.png' )
    distance_iteration_plots(dist_dfs, dist_ids, all_labels, 'max_value_info', 'Reward Accumulation', 200., 100, len(seeds), True, False, '_avg_rac_dist.png' )
    distance_iteration_plots(dist_dfs, dist_ids, all_labels, 'info_regret', 'Info Regret', 200., 100, len(seeds), True, False, '_avg_ireg_dist.png' )
    distance_iteration_plots(dist_dfs, dist_ids, all_labels, 'max_loc_error', 'Loc Error', 200., 100, len(seeds), True, False, '_avg_locerr_dist.png' )


    plt.show()
