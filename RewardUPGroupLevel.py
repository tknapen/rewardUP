from __future__ import division

#!/usr/bin/env python
# encoding: utf-8
"""
EyeLinkSession.py

Created by Tomas Knapen on 2011-04-27.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import os, sys, datetime, pickle
import math

import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as pl
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import glob
import seaborn as sn
import mne

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from joblib import Parallel, delayed
import itertools
from itertools import chain

import logging, logging.handlers, logging.config

sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.log import *
from Tools.Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data
from Tools.Operators.CommandLineOperator import ExecCommandLine
from Tools.other_scripts.plotting_tools import *
from Tools.other_scripts import functions_jw as myfuncs

from Tools.other_scripts import savitzky_golay as savitzky_golay

from IPython import embed as shell

def bootstrap(data, nr_reps = 1000, full_output = False, threshold = 0.0):
	"""data is a subjects by value array
	bootstrap returns the p-value of the difference with zero, and optionally the full distribution.
	"""
	
	permutes = np.random.randint(0, data.shape[0], size = (data.shape[0], nr_reps))
	bootstrap_distro = data[permutes].mean(axis = 0)
	p_val = 1.0 - ((bootstrap_distro > threshold).sum() / float(nr_reps))
	if full_output:
		return p_val, bootstrap_distro
	else:
		return p_val

def permutation_test(data, nr_reps = 10000, full_output = False):
	"""permutation_test assumes a subjects by value by condition array.
	"""
	actual_difference = np.mean(data[:,0] - data[:,1])

	# cond_1 = np.random.randint(2,size = (nr_reps, data.shape[0]))
	# cond_2 = 1-cond_1
	# permuted_distro = np.array([np.array([data[i,d] for i, d in enumerate(cond_1[i])])-np.array([data[i,d] for i, d in enumerate(cond_2[i])]) for i in range(nr_reps)]).mean(axis = 1)

	cond = np.array(np.random.randint(2,size = (nr_reps, data.shape[0])), dtype = bool)
	cond = np.array([cond, -cond]).transpose(1,2,0)
	permuted_distro = np.array([data[c]-data[-c] for c in cond]).mean(axis = 1)

	p_val = 1.0 - ((permuted_distro > actual_difference).sum() / float(nr_reps))
	if full_output:
		return p_val, permuted_distro
	else:
		return p_val





class RewardUPGroupLevel(object): 
	"""Instances of RewardUPGroupLevel can be used to do group analyses on PupilPredErrSession experiments"""

	def __init__ (self, sessions, data_folder, loggingLevel = logging.DEBUG):
		self.sessions = sessions 
		self.data_folder = data_folder
		self.grouplvl_dir = os.path.join(data_folder, 'group_level')
		self.grouplvl_data_dir = os.path.join(data_folder, 'group_level' , 'data')
		self.grouplvl_plot_dir = os.path.join(data_folder, 'group_level', 'figs')
		self.grouplvl_log_dir = os.path.join(data_folder, 'group_level', 'log')
		self.hdf5_filename = os.path.join(self.grouplvl_data_dir, 'all_data.hdf5')
		for directory in [self.grouplvl_dir, self.grouplvl_data_dir, self.grouplvl_plot_dir, self.grouplvl_log_dir]:
			try: 
				os.mkdir(directory)
			except OSError: 
				pass 

		###some stuff about logging I don't understand###
		# add logging for this session
		# sessions create their own logging file handler
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger( self.__class__.__name__ )
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( os.path.join(self.grouplvl_log_dir, 'groupLogFile.log'), when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		self.logger.info('starting analysis in ' + self.grouplvl_log_dir)

	def gather_data_from_hdfs(self, group = 'basic_pupil_deconvolution', data_type = 'time_points', hdf_file = 'eye'):
		"""gather_data_from_hdfs takes arbitrary data from hdf5 files for all self.sessions.
		arguments:  group - the folder in the hdf5 file from which to take data
					data_type - the type of data to be read.
		returns a numpy array with all the data.
		"""
		gathered_data = []
		for s in self.sessions:
			if hdf_file == 'eye':
				h5f = s.hdf5_eye_filename
			elif hdf_file == 'mri':
				h5f = s.hdf5_mri_filename
			with pd.get_store(h5f) as h5_file:
				gathered_data.append(np.array(h5_file.get("/%s/%s"%(group, data_type))))

		return np.array(gathered_data)

	def basic_deconv_results_across_subjects(self, data_type = 'pupil_bp', sig_level = 0.05, hdf_file = 'eye', which_data = 'pupil', smooth_width = 5, baseline = False):
		"""basic_deconv_results_across_subjects takes timepoints from deconvolve_colour_sound, and averages across subjects.
		"""

		self.logger.info('plotting across-subjects deconvolution results of the %s event-related responses.'%which_data)

		dec_time_courses_1 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_1', hdf_file = hdf_file)
# 		dec_time_courses_2 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_2', hdf_file = hdf_file)
		dec_time_courses_4 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_4', hdf_file = hdf_file)
		dec_time_courses_5 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_5', hdf_file = hdf_file)

		dec_time_courses_6 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_7', hdf_file = hdf_file)

		
		time_points = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'time_points', hdf_file = hdf_file).mean(axis = 0)
		baseline_times = (time_points < 0.0) * (time_points > -1.0)

		if baseline:
			dec_time_courses_1_s = np.array([[myfuncs.smooth(dec_time_courses_1[i,:,j], window_len=smooth_width) - dec_time_courses_1[i,baseline_times,j].mean() for i in range(dec_time_courses_1.shape[0])] for j in range(dec_time_courses_1.shape[-1])]).transpose((1,2,0))
# 			dec_time_courses_2_s = np.array([[myfuncs.smooth(dec_time_courses_2[i,:,j], window_len=smooth_width) - dec_time_courses_2[i,baseline_times,j].mean() for i in range(dec_time_courses_2.shape[0])] for j in range(dec_time_courses_2.shape[-1])]).transpose((1,2,0))
			dec_time_courses_4_s = np.array([[myfuncs.smooth(dec_time_courses_4[i,:,j], window_len=smooth_width) - dec_time_courses_4[i,baseline_times,j].mean() for i in range(dec_time_courses_4.shape[0])] for j in range(dec_time_courses_4.shape[-1])]).transpose((1,2,0))
			dec_time_courses_5_s = np.array([[myfuncs.smooth(dec_time_courses_5[i,:,j], window_len=smooth_width) - dec_time_courses_5[i,baseline_times,j].mean() for i in range(dec_time_courses_5.shape[0])] for j in range(dec_time_courses_5.shape[-1])]).transpose((1,2,0))
			dec_time_courses_6_s = np.array([[myfuncs.smooth(dec_time_courses_6[i,:,j], window_len=smooth_width) - dec_time_courses_6[i,baseline_times,j].mean() for i in range(dec_time_courses_6.shape[0])] for j in range(dec_time_courses_6.shape[-1])]).transpose((1,2,0))
			dec_time_courses_7_s = np.array([[myfuncs.smooth(dec_time_courses_7[i,:,j], window_len=smooth_width) - dec_time_courses_7[i,baseline_times,j].mean() for i in range(dec_time_courses_7.shape[0])] for j in range(dec_time_courses_7.shape[-1])]).transpose((1,2,0))


		else:
			dec_time_courses_1_s = np.array([[myfuncs.smooth(dec_time_courses_1[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1.shape[0])] for j in range(dec_time_courses_1.shape[-1])]).transpose((1,2,0))
# 			dec_time_courses_2_s = np.array([[myfuncs.smooth(dec_time_courses_2[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_2.shape[0])] for j in range(dec_time_courses_2.shape[-1])]).transpose((1,2,0))
			dec_time_courses_4_s = np.array([[myfuncs.smooth(dec_time_courses_4[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_4.shape[0])] for j in range(dec_time_courses_4.shape[-1])]).transpose((1,2,0))	
			dec_time_courses_5_s = np.array([[myfuncs.smooth(dec_time_courses_5[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5.shape[0])] for j in range(dec_time_courses_5.shape[-1])]).transpose((1,2,0))
			dec_time_courses_6_s = np.array([[myfuncs.smooth(dec_time_courses_6[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6.shape[0])] for j in range(dec_time_courses_6.shape[-1])]).transpose((1,2,0))
			dec_time_courses_7_s = np.array([[myfuncs.smooth(dec_time_courses_7[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7.shape[0])] for j in range(dec_time_courses_7.shape[-1])]).transpose((1,2,0))
			
		# cis = np.linspace(68, 33, 2)
		sn.set(style="ticks")

		f = pl.figure(figsize = (5.5,12))
		ax1 = f.add_subplot(411)
		ax1.set_title('%s data stimulus responses and blinks' % which_data)
		conds = pd.Series(['blink','microsaccade','stimulus on','stimulus off'], name="conditions")
		pal = sn.dark_palette("palegreen", 4)

		sn.tsplot(dec_time_courses_1_s, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z')

		ax2 = f.add_subplot(412)
		ax2.set_title('%s data responses to transitions'%which_data)
		conds = pd.Series(['transition offset','transition onset'], name="conditions")
		pal = sn.dark_palette("red", 2)

		sn.tsplot(dec_time_courses_4_s[:,:,:2], err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax2.set_ylabel('Z')
		
		# now for a per-subject difference score between onset and offset. 
		dec_time_courses_smooth_differences = dec_time_courses_4_s[:,:,1] - dec_time_courses_4_s[:,:,0]
		# pvals_bootstrap = np.array([bootstrap(d) for d in dec_time_courses_smooth_differences.T])

		pvals_permute = np.array([permutation_test(d) for d in dec_time_courses_4_s.transpose((1,0,2))])

		# sig_timepoints_bootstrap = time_points[((pvals_bootstrap < sig_level) + (pvals_bootstrap > (1.0-sig_level))) * (time_points < 0)]
		# sig_timepoints_permute = time_points[((pvals_permute < sig_level) + (pvals_permute > (1.0-sig_level))) * (time_points < 0) * (time_points > -2.5)]
		# sig_timepoints_permute = time_points[(pvals_permute > (1.0-sig_level))] #  * (time_points < 0) * (time_points > -2.5)
		# clusters = mne.stats.permutation_cluster_1samp_test(np.array(dec_time_courses_2_s))[1]
		clusters = mne.stats.permutation_cluster_1samp_test(dec_time_courses_smooth_differences)[1]
		# shell()
		if len(clusters) == 0:
			sig_timepoints_permute = time_points[np.zeros(pvals_permute.shape[0], dtype = bool)]
		# elif len(clusters) == 1:
		# 	sig_timepoints_permute = time_points[np.array(clusters[0]).astype(bool)[:,0]]
		else:
			# sig_timepoints_permute = time_points[np.array(clusters).sum(axis = 0)[:,0].astype(bool)]
			sig_indices = np.concatenate([np.arange(dec_time_courses_smooth_differences.shape[1])[c] for c in clusters])
			# sig_timepoints_permute = np.zeros(dec_time_courses_smooth_differences.shape[1], dtype = bool)
			# sig_timepoints_permute[sig_indices] = True
			sig_timepoints_permute = time_points[sig_indices]


		self.logger.info(data_type + '_' + which_data + ' # time points sig: %i, fdr corrected minimum : %1.3f'%(sig_timepoints_permute.sum(), np.min(mne.stats.fdr_correction(1-pvals_permute, 0.05)[1])))
		ax3 = f.add_subplot(413)
		ax3.set_title('differential transition onset/offset %s data responses' % which_data)

		# if which_data == 'pupil':
		ax3.plot(sig_timepoints_permute, np.zeros(len(sig_timepoints_permute)), 'ks', ms = 3, alpha = 0.3)
		sn.tsplot(dec_time_courses_smooth_differences, err_style="ci_band", time = time_points, color = 'b') # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax3.set_xlabel('time [s]')
		ax3.set_ylabel('Z')
		# if which_data == 'pupil':
			# ax.annotate('p<%1.3f\n@\n%1.3f s'%(sig_level, sig_timepoints_permute[0]), (sig_timepoints_permute[0], -0.025), xytext = (time_points[0], -0.125), arrowprops = {'width':1.5,'headwidth':7,'frac':0.15,'color':'k'})
		
		# ax = f.add_subplot(414)
		# ax.set_title('%s data responses to on+off, IT'%which_data)
		# conds = pd.Series(['Trans on + off','IT'], name="conditions")
		# pal = sn.dark_palette("blue", 2)
		
 	# 	combined = np.array([np.squeeze([dec_time_courses_4_s[:,:,0]+dec_time_courses_4_s[:,:,1]]),dec_time_courses_4_s[:,:,3]]).transpose(1,2,0)
		# sn.tsplot(combined, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 

		ax4 = f.add_subplot(414)
		ax4.set_title('%s data responses to halfway transition'%which_data)
		conds = pd.Series(['Trans halfway'], name="conditions")
		pal = sn.dark_palette("blue", 1)
		sn.tsplot(dec_time_courses_5_s[:,:,0], err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 

		
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax4.set_ylabel('Z')
		
		
		pl.tight_layout()
		self.logger.info(os.path.join(self.grouplvl_plot_dir, data_type + '_' + which_data + '_deconv_diff_filtered_4.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, data_type + '_' + which_data + '_deconv_diff_filtered_4.pdf'))
		# shell()


	def bar_charts_from_deconvolutions(self, regions = ['Thalamus', 'V1', 'MT', 'combined_insula', 'IFC'], contrast = 'BR_gfeat_0mm_w_blinks_cope3_zstat', sign = 'pos', interval = [4,5.5]):
		""""""
		condition_labels = ['transition\nstart','percept\nstart']

		dec_time_courses = np.array([self.gather_data_from_hdfs(group = 'basic_' + 'mri' + '_deconvolution_' + 'mcf_phys_sgtf_Z_%s_%s_%s'%(r,sign, contrast), data_type = 'dec_time_course_2', hdf_file = 'mri') for r in regions])
		time_points = self.gather_data_from_hdfs(group = 'basic_' + 'mri' + '_deconvolution_' + 'mcf_phys_sgtf_Z_%s_%s_%s'%(regions[0],sign, contrast), data_type = 'time_points', hdf_file = 'mri')[0]
		selected_times = (time_points > interval[0]) * (time_points < interval[1])


		dec_time_courses_summed = dec_time_courses[:,:,selected_times].sum(axis = 2)

		dec_time_courses_summed_mean = np.mean(dec_time_courses_summed, axis = 1)
		dec_time_courses_summed_ci = np.std(dec_time_courses_summed, axis = 1) / math.sqrt(len(self.sessions)) * 1.96

		n_rows = np.prod(dec_time_courses_summed.shape)
		orig_sh = dec_time_courses_summed.shape
		flat_df = pd.DataFrame([[dec_time_courses_summed[r%orig_sh[0], r%orig_sh[1], r%orig_sh[2]], regions[r%orig_sh[0]], r%orig_sh[1], condition_labels[r%orig_sh[2]]] for r in range(n_rows)], columns = ['summed_response', 'area', 'sj_nr', 'condition'])


		# f = pl.figure(figsize = (9,4))
		g = sn.factorplot(x="area", y="summed_response", hue="condition",
			data=flat_df, kind="bar", legend_out = True, legend = False, x_order = regions, 
			size=3, aspect = 2, palette="Greens_d", ci = 90, estimator = np.mean)
		g.despine(offset=10, trim=True)
		g.set_ylabels("Summed Z scored\nBOLD response")
		pl.savefig(os.path.join(self.grouplvl_plot_dir,contrast + '_' + sign + '_deconv_bars.pdf'))
		# shell()

	def bar_charts_from_halfway_deconvolutions(self, regions = ['Thalamus', 'V1', 'MT', 'combined_insula', 'IFC'], contrast = 'BR_gfeat_0mm_halfway_trans_cope3_zstat', interval = [4,5.5]):
		""""""
		condition_labels = ['positive','negative']
		
		dec_time_courses_pos = np.array([self.gather_data_from_hdfs(group = 'basic_' + 'mri' + '_deconvolution_' + 'mcf_phys_tf_Z_%s_pos_%s'%(r, contrast), data_type = 'dec_time_course_2', hdf_file = 'mri') for r in regions])
		dec_time_courses_neg = np.array([self.gather_data_from_hdfs(group = 'basic_' + 'mri' + '_deconvolution_' + 'mcf_phys_tf_Z_%s_neg_%s'%(r, contrast), data_type = 'dec_time_course_2', hdf_file = 'mri') for r in regions])	
		
		time_points = self.gather_data_from_hdfs(group = 'basic_' + 'mri' + '_deconvolution_' + 'mcf_phys_sgtf_Z_%s_pos_%s'%(regions[0], contrast), data_type = 'time_points', hdf_file = 'mri')[0]
		selected_times = (time_points > interval[0]) * (time_points < interval[1])

		dec_time_courses_pos_summed = dec_time_courses_pos[:,:,selected_times,0].sum(axis = 2)
		dec_time_courses_neg_summed = dec_time_courses_neg[:,:,selected_times,0].sum(axis = 2)


		dec_time_courses_pos_summed_mean = np.mean(dec_time_courses_pos_summed, axis = 1)
		dec_time_courses_pos_summed_ci = np.std(dec_time_courses_pos_summed, axis = 1) / math.sqrt(len(self.sessions)) * 1.96

		dec_time_courses_neg_summed_mean = np.mean(dec_time_courses_neg_summed, axis = 1)
		dec_time_courses_neg_summed_ci = np.std(dec_time_courses_neg_summed, axis = 1) / math.sqrt(len(self.sessions)) * 1.96

# 		
# 		n_rows = np.prod(dec_time_courses_summed.shape)
# 		orig_sh = dec_time_courses_summed.shape
# 		flat_df = pd.DataFrame([[dec_time_courses_summed[r%orig_sh[0], r%orig_sh[1], r%orig_sh[2]], regions[r%orig_sh[0]], r%orig_sh[1], condition_labels[r%orig_sh[2]]] for r in range(n_rows)], columns = ['summed_response', 'area', 'sj_nr', 'condition'])
# 
# 	
# 		# f = pl.figure(figsize = (9,4))
# 		g = sn.factorplot(x="area", y="summed_response", hue="condition",
# 			data=flat_df, kind="bar", legend_out = True, legend = False, x_order = regions, 
# 			size=3, aspect = 2, palette="Greens_d", ci = 90, estimator = np.mean)
# 		g.despine(offset=10, trim=True)
# 		g.set_ylabels("Summed Z scored\nBOLD response")

		N = 2
		means = (dec_time_courses_pos_summed_mean[0], dec_time_courses_neg_summed_mean[0])
		cis = (dec_time_courses_pos_summed_ci[0],dec_time_courses_neg_summed_ci[0])
		
		ind = np.arange(N)  # the x locations for the groups
		inds = (ind[0],ind[1]-.5)
		width = 0.1       # the width of the bars

		fig, ax = pl.subplots()
		ax.bar(inds, means, width, color='r', yerr = cis)
		
		pl.savefig(os.path.join(self.grouplvl_plot_dir,contrast + r + '_halfway_pos_neg_deconv_bars.pdf'))

		
	def transition_blink_histogram_across_subjects(self, data_type = 'pupil_bp', hdf_file = 'eye', interval = 2.5):
		
		self.logger.info('plotting across-subjects transition-blinks results.')
		kde_time_courses = self.gather_data_from_hdfs(group = 'kde_blink_behavior_' + data_type, data_type = 'kde_time_courses', hdf_file = hdf_file)
		time_points = self.gather_data_from_hdfs(group = 'kde_blink_behavior_' + data_type, data_type = 'time_points', hdf_file = hdf_file).mean(axis = 0)
		
		# shell()
		kde_time_courses = kde_time_courses.transpose((0,2,1))

		assert np.max(time_points) >= interval

		# select times based on interval and Zscore these data.
		selected_times = (time_points > -interval) * (time_points < interval)
		m_d = np.median(kde_time_courses[:,selected_times], axis = 1)
		s_d = kde_time_courses[:,selected_times].std(axis = 1)
		z_scored_data = ((kde_time_courses[:,selected_times].transpose(1,0,2) - m_d) / s_d).transpose(1,0,2)
		time_points = time_points[selected_times]

		permute_sig_points = []
		for i in range(z_scored_data.shape[-1]): # loop across conditions
			self.logger.info('permutation stats on %s'%['trans_blinks','trans_ms','percept_blinks','percept_ms'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(z_scored_data[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)


		f = pl.figure(figsize = (9, 4))
		ax = f.add_subplot(121)
		ax.set_title('blinks & ms, transition onset')
		conds = pd.Series(['blink','microsaccade'], name="conditions")
		pal = sn.dark_palette("palegreen", 2)
		# pal = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

		for i in [0,1]:
			ax.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.1 * i, ['ks','gs'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(z_scored_data[:,:,:2], err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax.set_ylabel('Z-scored density [Z]')
		ax.set_xlabel('time relative to transition onset [s]')
		ax.set_ylim([-1.5,2.5])

		ax = f.add_subplot(122)
		ax.set_title('blinks & ms, percept onset')
		for i in [2,3]:
			ax.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.1 * i - 0.1, ['ks','gs'][i-2], ms = 4.5, alpha = 0.2)
		sn.tsplot(z_scored_data[:,:,2:], err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		ax.set_ylim([-1.5,2.5])
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax.set_xlabel('time relative to percept onset [s]')

		pl.tight_layout()
		pl.savefig(os.path.join(self.grouplvl_plot_dir,'blinks_ms_densities_filtered_transitions.pdf'))
		# pl.show()

	def basic_deconv_results_across_subjects_multiple_V1rois(self, data_type = 'mcf_phys_sgtf_Z_', roi='V1', mask_type = 'stim_on_mapper_Z_0mm', hdf_file = 'mri', which_data = 'mri', smooth_width = 7):
		"""basic_deconv_results_across_subjects takes timepoints from deconvolve_colour_sound, and averages across subjects.
		"""
		self.logger.info('plotting across-subjects deconvolution results of the %s event-related responses.'%which_data)

		data_type_tmp = [data_type + 'V1_pos_' + mask_type]
		dec_time_courses_1_pos = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5_pos = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6_pos = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7_pos = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_7', hdf_file = hdf_file)
	
		data_type_tmp = [data_type + 'V1_neg_' + mask_type]
		dec_time_courses_1_neg = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5_neg = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6_neg = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7_neg = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_7', hdf_file = hdf_file)

		data_type_tmp = [data_type + 'V1-4-1_all_']
		dec_time_courses_1_V1_1 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5_V1_1 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6_V1_1 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7_V1_1 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_7', hdf_file = hdf_file)


		data_type_tmp = [data_type + 'V1-4-2_all_']
		dec_time_courses_1_V1_2 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5_V1_2 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6_V1_2 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7_V1_2 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_7', hdf_file = hdf_file)


		data_type_tmp = [data_type + 'V1-4-3_all_']
		dec_time_courses_1_V1_3 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5_V1_3 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6_V1_3 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7_V1_3 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_7', hdf_file = hdf_file)

		
		data_type_tmp = [data_type + 'V1-4-4_all_']
		dec_time_courses_1_V1_4 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5_V1_4 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6_V1_4 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7_V1_4 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type_tmp[0], data_type = 'dec_time_course_7', hdf_file = hdf_file)



		time_points = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + 'mcf_phys_sgtf_Z_V1_pos_stim_on_mapper_Z_0mm', data_type = 'time_points', hdf_file = hdf_file).mean(axis = 0)
		baseline_times = (time_points < 0.0) * (time_points > -1.0)

		# Positive mapping region
		dec_time_courses_1_s_pos = np.array([[myfuncs.smooth(dec_time_courses_1_pos[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1_pos.shape[0])] for j in range(dec_time_courses_1_pos.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s_pos = np.array([[myfuncs.smooth(dec_time_courses_5_pos[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5_pos.shape[0])] for j in range(dec_time_courses_5_pos.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s_pos = np.array([[myfuncs.smooth(dec_time_courses_6_pos[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6_pos.shape[0])] for j in range(dec_time_courses_6_pos.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s_pos = np.array([[myfuncs.smooth(dec_time_courses_7_pos[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7_pos.shape[0])] for j in range(dec_time_courses_7_pos.shape[-1])]).transpose((1,2,0))

		# Negative mapping region
		dec_time_courses_1_s_neg = np.array([[myfuncs.smooth(dec_time_courses_1_neg[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1_neg.shape[0])] for j in range(dec_time_courses_1_neg.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s_neg = np.array([[myfuncs.smooth(dec_time_courses_5_neg[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5_neg.shape[0])] for j in range(dec_time_courses_5_neg.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s_neg = np.array([[myfuncs.smooth(dec_time_courses_6_neg[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6_neg.shape[0])] for j in range(dec_time_courses_6_neg.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s_neg = np.array([[myfuncs.smooth(dec_time_courses_7_neg[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7_neg.shape[0])] for j in range(dec_time_courses_7_neg.shape[-1])]).transpose((1,2,0))

		# All at first retinotopic region
		dec_time_courses_1_s_V1_1 = np.array([[myfuncs.smooth(dec_time_courses_1_V1_1[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1_V1_1.shape[0])] for j in range(dec_time_courses_1_V1_1.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s_V1_1 = np.array([[myfuncs.smooth(dec_time_courses_5_V1_1[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5_V1_1.shape[0])] for j in range(dec_time_courses_5_V1_1.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s_V1_1 = np.array([[myfuncs.smooth(dec_time_courses_6_V1_1[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6_V1_1.shape[0])] for j in range(dec_time_courses_6_V1_1.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s_V1_1 = np.array([[myfuncs.smooth(dec_time_courses_7_V1_1[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7_V1_1.shape[0])] for j in range(dec_time_courses_7_V1_1.shape[-1])]).transpose((1,2,0))

		# All at second retinotopic region
		dec_time_courses_1_s_V1_2 = np.array([[myfuncs.smooth(dec_time_courses_1_V1_2[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1_V1_2.shape[0])] for j in range(dec_time_courses_1_V1_2.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s_V1_2 = np.array([[myfuncs.smooth(dec_time_courses_5_V1_2[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5_V1_2.shape[0])] for j in range(dec_time_courses_5_V1_2.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s_V1_2 = np.array([[myfuncs.smooth(dec_time_courses_6_V1_2[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6_V1_2.shape[0])] for j in range(dec_time_courses_6_V1_2.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s_V1_2 = np.array([[myfuncs.smooth(dec_time_courses_7_V1_2[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7_V1_2.shape[0])] for j in range(dec_time_courses_7_V1_2.shape[-1])]).transpose((1,2,0))

		# All at third retinotopic region
		dec_time_courses_1_s_V1_3 = np.array([[myfuncs.smooth(dec_time_courses_1_V1_3[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1_V1_3.shape[0])] for j in range(dec_time_courses_1_V1_3.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s_V1_3 = np.array([[myfuncs.smooth(dec_time_courses_5_V1_3[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5_V1_3.shape[0])] for j in range(dec_time_courses_5_V1_3.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s_V1_3 = np.array([[myfuncs.smooth(dec_time_courses_6_V1_3[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6_V1_3.shape[0])] for j in range(dec_time_courses_6_V1_3.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s_V1_3 = np.array([[myfuncs.smooth(dec_time_courses_7_V1_3[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7_V1_3.shape[0])] for j in range(dec_time_courses_7_V1_3.shape[-1])]).transpose((1,2,0))

		# All at fourth retinotopic region
		dec_time_courses_1_s_V1_4 = np.array([[myfuncs.smooth(dec_time_courses_1_V1_4[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1_V1_4.shape[0])] for j in range(dec_time_courses_1_V1_4.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s_V1_4 = np.array([[myfuncs.smooth(dec_time_courses_5_V1_4[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5_V1_4.shape[0])] for j in range(dec_time_courses_5_V1_4.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s_V1_4 = np.array([[myfuncs.smooth(dec_time_courses_6_V1_4[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6_V1_4.shape[0])] for j in range(dec_time_courses_6_V1_4.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s_V1_4 = np.array([[myfuncs.smooth(dec_time_courses_7_V1_4[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7_V1_4.shape[0])] for j in range(dec_time_courses_7_V1_4.shape[-1])]).transpose((1,2,0))

		V1_stimulus_ret = np.array([dec_time_courses_1_s_V1_1[:,:,2],dec_time_courses_1_s_V1_2[:,:,2],dec_time_courses_1_s_V1_3[:,:,2],dec_time_courses_1_s_V1_4[:,:,2]]).transpose(1,2,0)
		V1_halfway_ret = np.array(np.squeeze([dec_time_courses_5_s_V1_1,dec_time_courses_5_s_V1_2,dec_time_courses_5_s_V1_3,dec_time_courses_5_s_V1_4])).transpose(1,2,0)
		V1_trans_on_ret = np.array(np.squeeze([dec_time_courses_6_s_V1_1,dec_time_courses_6_s_V1_2,dec_time_courses_6_s_V1_3,dec_time_courses_6_s_V1_4])).transpose(1,2,0)
		V1_trans_off_ret = np.array(np.squeeze([dec_time_courses_7_s_V1_1,dec_time_courses_7_s_V1_2,dec_time_courses_7_s_V1_3,dec_time_courses_7_s_V1_4])).transpose(1,2,0)

		V1_stimulus_mapper = np.array([dec_time_courses_1_s_pos[:,:,2],dec_time_courses_1_s_neg[:,:,2]]).transpose(1,2,0)
		V1_halfway_mapper = np.array(np.squeeze([dec_time_courses_5_s_pos,dec_time_courses_5_s_neg])).transpose(1,2,0)
		V1_trans_on_mapper = np.array(np.squeeze([dec_time_courses_6_s_pos,dec_time_courses_6_s_neg])).transpose(1,2,0)
		V1_trans_off_mapper = np.array(np.squeeze([dec_time_courses_7_s_pos,dec_time_courses_7_s_neg])).transpose(1,2,0)

		V1_blinks_ret = np.array([dec_time_courses_1_s_V1_1[:,:,0],dec_time_courses_1_s_V1_2[:,:,0],dec_time_courses_1_s_V1_3[:,:,0],dec_time_courses_1_s_V1_4[:,:,0]]).transpose(1,2,0)
		V1_ms_ret = np.array([dec_time_courses_1_s_V1_1[:,:,1],dec_time_courses_1_s_V1_2[:,:,1],dec_time_courses_1_s_V1_3[:,:,1],dec_time_courses_1_s_V1_4[:,:,1]]).transpose(1,2,0)


		# Blinks and microsaccades
		sn.set(style="ticks")

		# Stimulus on and off
		f = pl.figure(figsize = (5,8))
		ax1 = f.add_subplot(211)
		ax1.set_title('%s data blinks responses' % which_data)
		conds = pd.Series(['V1-4-1','V1-4-2','V1-4-3','V1-4-4'], name="regions")

		permute_sig_points = []
		for i in range(V1_blinks_ret.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['v1-4-1','v1-4-1','v1-4-3','v1-4-4'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(V1_blinks_ret[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)
		
		for i in range(V1_blinks_ret.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.01 * i, ['g','r','k','c'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(V1_blinks_ret[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['green','red','black','cyan']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z-scored density [Z]')
		ax1.set_xlabel('time [s]')


		ax2 = f.add_subplot(212)
		ax2.set_title('%s data microsaccades responses' % which_data)

		permute_sig_points = []
		for i in range(V1_ms_ret.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['v1-4-1','v1-4-1','v1-4-3','v1-4-4'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(V1_ms_ret[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)
		
		for i in range(V1_ms_ret.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.01 * i, ['g','r','k','c'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(V1_ms_ret[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['green','red','black','cyan']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z-scored density [Z]')
		ax1.set_xlabel('time [s]')

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_blinks_ms_sig.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_blinks_ms_sig.pdf'))


# 		halfway_pos_neg_deco_timecourses = [dec_time_courses_5_s_pos[:,:,1],dec_time_courses_5_s_neg[:,:,1]]
# 		cis = np.linspace(68, 33, 2)
		sn.set(style="ticks")

		# Stimulus on and off
		f = pl.figure(figsize = (5,12))
		ax1 = f.add_subplot(411)
		ax1.set_title('%s data stimulus on responses' % which_data)
		conds = pd.Series(['V1-4-1','V1-4-2','V1-4-3','V1-4-4'], name="regions")
		pal = sn.color_palette("Set2", 4)

		sn.tsplot(V1_stimulus_ret, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z')


		# Transition halfway
		ax2 = f.add_subplot(412)
		ax2.set_title('%s data responses to halfway transitions'%which_data)
		conds = pd.Series(['V1-4-1','V1-4-2','V1-4-3','V1-4-4'], name="regions")
		pal = sn.dark_palette("green", 4)

		sn.tsplot(V1_halfway_ret, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax2.set_ylabel('Z')
		ax2.set_ylim(-0.05, .05 )


		# Transition on
		ax3 = f.add_subplot(413)
		ax3.set_title('%s data responses to onset transitions'%which_data)
		conds = pd.Series(['V1-4-1','V1-4-2','V1-4-3','V1-4-4'], name="regions")
		pal = sn.dark_palette("green", 4)

		sn.tsplot(V1_trans_on_ret, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax3.set_ylabel('Z')
		ax3.set_ylim(-0.05, .05 )


		# Transition on
		ax4 = f.add_subplot(414)
		ax4.set_title('%s data responses to offset transitions'%which_data)
		conds = pd.Series(['V1-4-1','V1-4-2','V1-4-3','V1-4-4'], name="regions")
		pal = sn.dark_palette("green", 4)

		sn.tsplot(V1_trans_off_ret, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax4.set_ylabel('Z')
		ax4.set_ylim(-0.05, .05 )

		# stim mapper
		f = pl.figure(figsize = (5,4))
		ax1 = f.add_subplot(111)
		pal = ['green','red','black','cyan']


		# shell()

		permute_sig_points = []
		for i in range(V1_stimulus_ret.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['v1-4-1','v1-4-1','v1-4-3','v1-4-4'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(V1_stimulus_ret[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)
		
		for i in range(V1_stimulus_ret.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.01 * i, ['g','r','k','c'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(V1_stimulus_ret[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['green','red','black','cyan']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z-scored density [Z]')
		ax1.set_xlabel('time [s]')
		# ax1.set_ylim([-1.5,2.5])

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_stimulus_sig.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_stimulus_sig.pdf'))

		# Trans halfway statistiek
		f = pl.figure(figsize = (5,4))
		ax1 = f.add_subplot(111)
		pal = sn.dark_palette("green", 4)

		permute_sig_points = []
		for i in range(V1_halfway_ret.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['v1-4-1','v1-4-1','v1-4-3','v1-4-4'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(V1_halfway_ret[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)
		
		for i in range(V1_halfway_ret.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.001 * i, ['ks','gs','cs','rs'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(V1_halfway_ret[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['black','green','cyan','red']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z-scored density [Z]')
		ax1.set_xlabel('time [s]')

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_halfway_sig.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_halfway_sig.pdf'))
		
		# Trans on statistiek
		f = pl.figure(figsize = (5,4))
		ax1 = f.add_subplot(111)
		pal = sn.dark_palette("green", 4)

		permute_sig_points = []
		for i in range(V1_trans_on_ret.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['v1-4-1','v1-4-1','v1-4-3','v1-4-4'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(V1_trans_on_ret[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)
		
		for i in range(V1_trans_on_ret.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.001 * i, ['ks','gs','cs','rs'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(V1_trans_on_ret[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['black','green','cyan','red']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z-scored density [Z]')
		ax1.set_xlabel('time [s]')

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_trans_on_sig.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_trans_on_sig.pdf'))
		
		# Trans off statistiek
		f = pl.figure(figsize = (5,4))
		ax1 = f.add_subplot(111)
		pal = sn.dark_palette("green", 4)

		permute_sig_points = []
		for i in range(V1_trans_off_ret.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['v1-4-1','v1-4-1','v1-4-3','v1-4-4'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(V1_trans_off_ret[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)
		
		for i in range(V1_trans_off_ret.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.001 * i, ['ks','gs','cs','rs'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(V1_trans_off_ret[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['black','green','cyan','red']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z-scored density [Z]')
		ax1.set_xlabel('time [s]')

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_trans_off_sig.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions_trans_off_sig.pdf'))
		# now for a per-subject difference score between onset and offset. 
		# dec_time_courses_smooth_differences = [dec_time_courses_5_s_pos[:,:,1] - dec_time_courses_5_s_neg[:,:,1]]
		# pvals_bootstrap = np.array([bootstrap(d) for d in dec_time_courses_smooth_differences.T])

# 		pvals_permute = np.array([permutation_test(d) for d in dec_time_courses_4_s_pos.transpose((1,0,2))])

		# sig_timepoints_bootstrap = time_points[((pvals_bootstrap < sig_level) + (pvals_bootstrap > (1.0-sig_level))) * (time_points < 0)]
		# sig_timepoints_permute = time_points[((pvals_permute < sig_level) + (pvals_permute > (1.0-sig_level))) * (time_points < 0) * (time_points > -2.5)]
		# sig_timepoints_permute = time_points[(pvals_permute > (1.0-sig_level))] #  * (time_points < 0) * (time_points > -2.5)
		# clusters = mne.stats.permutation_cluster_1samp_test(np.array(dec_time_courses_2_s))[1]
		# clusters = mne.stats.permutation_cluster_1samp_test(dec_time_courses_smooth_differences)[1]
		# shell()
		# if len(clusters) == 0:
			# sig_timepoints_permute = time_points[np.zeros(pvals_permute.shape[0], dtype = bool)]
		# elif len(clusters) == 1:
		# 	sig_timepoints_permute = time_points[np.array(clusters[0]).astype(bool)[:,0]]
		# else:
			# sig_timepoints_permute = time_points[np.array(clusters).sum(axis = 0)[:,0].astype(bool)]
			# sig_indices = np.concatenate([np.arange(dec_time_courses_smooth_differences.shape[1])[c] for c in clusters])
			# sig_timepoints_permute = np.zeros(dec_time_courses_smooth_differences.shape[1], dtype = bool)
			# sig_timepoints_permute[sig_indices] = True
			# sig_timepoints_permute = time_points[sig_indices]


		# self.logger.info(data_type + '_' + which_data + ' # time points sig: %i, fdr corrected minimum : %1.3f'%(sig_timepoints_permute.sum(), np.min(mne.stats.fdr_correction(1-pvals_permute, 0.05)[1])))
		# # f = pl.figure(figsize = (8,4))
		# ax = f.add_subplot(212)
		# ax.set_title('differential halfway transition pos/neg data responses')
		# # if which_data == 'pupil':
		# ax.plot(sig_timepoints_permute, np.zeros(len(sig_timepoints_permute)), 'ks', ms = 3, alpha = 0.3)
		# sn.tsplot(dec_time_courses_smooth_differences, err_style="ci_band", time = time_points, color = 'b') # ci=cis, 
		# pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		# pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		# sn.despine(offset=10, trim=True)
		# ax.set_xlabel('time [s]')
		# ax.set_ylabel('Z')
		# if which_data == 'pupil':
			# ax.annotate('p<%1.3f\n@\n%1.3f s'%(sig_level, sig_timepoints_permute[0]), (sig_timepoints_permute[0], -0.025), xytext = (time_points[0], -0.125), arrowprops = {'width':1.5,'headwidth':7,'frac':0.15,'color':'k'})
		
		# ax = f.add_subplot(414)
# 		ax.set_title('%s data responses to on+off, IT'%which_data)
# 		conds = pd.Series(['Trans on + off','IT'], name="conditions")
# 		pal = sn.dark_palette("blue", 2)
# 		
#  		combined = np.array([np.squeeze([dec_time_courses_4_s[:,:,0]+dec_time_courses_4_s[:,:,1]]),dec_time_courses_4_s[:,:,3]]).transpose(1,2,0)
# 		sn.tsplot(combined, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
# 		
# 		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
# 		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
# 		sn.despine(offset=10, trim=True)
# 		ax.set_ylabel('Z')
		
		
		# pl.tight_layout()
		
		# self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions.pdf'))
		# pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_retinoregions.pdf'))
		# # shell()

#		# Stimulus on and off
		f = pl.figure(figsize = (5,12))
		ax1 = f.add_subplot(411)
		ax1.set_title('%s data stimulus on responses' % which_data)
		conds = pd.Series(['positive','negative'], name="regions")
		pal = sn.color_palette("Set2", 2)

		shell()
		sn.tsplot(V1_stimulus_mapper, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z')

		# Transition halfway
		ax2 = f.add_subplot(412)
		ax2.set_title('%s data responses to halfway transitions'%which_data)
		conds = pd.Series(['positive','negative'], name="regions")
		pal = sn.dark_palette("red", 2)

		sn.tsplot(V1_halfway_mapper, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax2.set_ylabel('Z')
		ax2.set_ylim(-0.1, .1 )


		# Transition on
		ax3 = f.add_subplot(413)
		ax3.set_title('%s data responses to onset transitions'%which_data)
		conds = pd.Series(['positive','negative'], name="regions")
		pal = sn.dark_palette("red", 2)

		sn.tsplot(V1_trans_on_mapper, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax3.set_ylabel('Z')
		ax3.set_ylim(-0.1, .1 )


		# Transition on
		ax4 = f.add_subplot(414)
		ax4.set_title('%s data responses to offset transitions'%which_data)
		conds = pd.Series(['positive','negative'], name="regions")
		pal = sn.dark_palette("red", 2)
		sn.tsplot(V1_trans_off_mapper, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax4.set_ylabel('Z')
		ax4.set_ylim(-0.10, .1 )

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_mapperregions.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type + which_data + '_mapperregions.pdf'))

	def plot_rsq_across_subjects(self, data_type = 'mcf_phys_sgtf_Z_', roi='V1', mask_type = 'stim_on_mapper_Z_0mm', mask_direction='all', hdf_file = 'mri', which_data = 'mri', event_division = 10, nr_reps_for_btstr = 1000):

		all_rsq = []
		for i in ['V1-4-1','V1-4-2','V1-4-3','V1-4-4']:
			new_rsq = []

			rsq = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type + i + '_' + mask_direction + '_' + mask_type, data_type = 'rsq_' + str(event_division), hdf_file = hdf_file)
			rsq_boxcar = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type + i + '_' + mask_direction + '_' + mask_type, data_type = 'rsq_boxcar', hdf_file = hdf_file)
			for n_subject in range(np.shape(rsq)[-2]):
				new_rsq.append(np.hstack([rsq[n_subject],rsq_boxcar[n_subject]]))
			rsq = np.array(new_rsq).squeeze()
			rsq = rsq.T - rsq.mean(axis = 1)
			rsq /= rsq.std(axis = 0)
			rsq = rsq.T
			all_rsq.append(rsq)


		all_rsq = np.array(all_rsq)
		all_rsq = all_rsq.transpose((1,2,0))



		time_points = np.linspace(0,1,all_rsq.shape[1], endpoint = True)
		
		f = pl.figure(figsize=(5,8))
		ax1= f.add_subplot(211)
		ax1.set_title('%s r2 10 different transition responses' % which_data)
		conds = pd.Series([str(d) for d in ['V1-4-1','V1-4-2','V1-4-3','V1-4-4']], name="area")
		pal = sn.color_palette("Set2", 4)
		sn.tsplot(all_rsq, err_style="ci_band", time = time_points, condition = conds, color = pal ) 
		ax1.axhline(0, lw=0.25, alpha=0.25, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('normalized r-squared')
		ax1.set_xlabel('relative time of transition event within duration')
		pl.tight_layout()

		ax2 = f.add_subplot(212)

		all_btstr_res = np.zeros((4, nr_reps_for_btstr))
		for region in np.arange(all_rsq.shape[-1]):
			indices = np.random.randint(0,all_rsq.shape[0], size = (nr_reps_for_btstr, all_rsq.shape[0]))
			btstr_timelines = all_rsq[indices,:,region].mean(axis = 1)
			btstr_res = np.arctanh(np.array([sp.stats.spearmanr(np.linspace(0,1,event_division+2), bt) for bt in btstr_timelines])[:,0])
			pl.hist(btstr_res[(btstr_res != np.inf) * (btstr_res != -np.inf)], bins = 100, range = [-4,4], alpha = 0.4, cumulative = True, histtype = 'step', normed = True, linewidth = 2.5, color = pal[region])

			print btstr_res[(btstr_res != np.inf) * (btstr_res != -np.inf)].mean(), btstr_res[(btstr_res != np.inf) * (btstr_res != -np.inf)].std()
			all_btstr_res[region] = btstr_res
		pl.axhline(0.5, lw=0.25, alpha=0.25, color = 'k')
		pl.axvline(0.0, lw=0.25, alpha=0.25, color = 'k')
		ax2.set_ylim([-0.1, 1.1])
		ax2.set_xlim([-3.7, 3.7])
		sn.despine(offset=10, trim=True)
		ax2.set_ylabel('cumulative bootstrap distro')
		ax2.set_xlabel('Z-transformed correlation begin to end')
		pl.tight_layout()

		shell()
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_r2_2.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type + which_data + '_r2_2.pdf'))


		#plot the HRFs
		self.logger.info('plotting across-subjects deconvolution results of the %s event-related responses.'%which_data)

		smooth_width = 7
		for roi_name in ['V1-4-1','V1-4-2','V1-4-3','V1-4-4']:

			time_points = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + 'mcf_phys_sgtf_Z_V1_pos_stim_on_mapper_Z_0mm', data_type = 'time_points', hdf_file = hdf_file).mean(axis = 0)
			baseline_times = (time_points < 0.0) * (time_points > -1.0)

			for d in range(10+1):
				dec_time_courses_2 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type  + roi_name + '_' + mask_direction + '_' + mask_type, data_type = 'dec_time_course_2_' + str(d), hdf_file = hdf_file)
				
				time_points = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type + roi_name + '_' + mask_direction + '_' + mask_type, data_type = 'time_points', hdf_file = hdf_file).mean(axis = 0)
				
				baseline_times = (time_points < 0.0) * (time_points > -1.0)
				dec_time_courses_2_s = np.array([[myfuncs.smooth(dec_time_courses_2[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_2.shape[0])] for j in range(dec_time_courses_2.shape[-1])]).transpose((1,2,0))
				
				# cis = np.linspace(68, 33, 2)
				sn.set(style="ticks")

				f = pl.figure(figsize = (5,3))
				ax1 = f.add_subplot(111)
				ax1.set_title('%s data to %s / 10 of transition' % (which_data,d))
				conds = pd.Series(['transition response'], name="conditions")
				pal = sn.dark_palette("palegreen", 1)

				sn.tsplot(dec_time_courses_2_s, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
				pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
				pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
				sn.despine(offset=10, trim=True)
				ax1.set_ylabel('Z')
				ax1.set_xlabel('time [s]')

				# permute_sig_points = []
				# self.logger.info('permutation stats on division %s'%d)
				# clusters = mne.stats.permutation_cluster_1samp_test(dec_time_courses_2_s)[1]
				# if len(clusters) == 0:
				# 	sig_timepoints_permute = time_points[np.zeros(pvals_permute.shape[0], dtype = bool)]
				# else:
				# 	shell()
				# 	sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
				# 	sig_timepoints_permute = time_points[sig_indices]
				# 	permute_sig_points.append(sig_timepoints_permute)

				# ax1.plot(permute_sig_points, np.zeros(len(permute_sig_points)) + 0.01 * i, ['k'], ms = 4.5, alpha = 0.2)

				pl.tight_layout()
				
				self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + roi_name + '_eventdivision_'+str(d)+'.pdf'))
				pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + roi_name + '_eventdivision_'+str(d)+'.pdf'))

			#Get response to boxcar
			dec_time_courses_3 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type  + roi_name + '_' + mask_direction + '_' + mask_type, data_type = 'dec_time_course_3_boxcar', hdf_file = hdf_file)
			dec_time_courses_3_s = np.array([[myfuncs.smooth(dec_time_courses_3[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_3.shape[0])] for j in range(dec_time_courses_3.shape[-1])]).transpose((1,2,0))
			# cis = np.linspace(68, 33, 2)
			sn.set(style="ticks")

			f = pl.figure(figsize = (5,3))
			ax1 = f.add_subplot(111)
			ax1.set_title('%s data to whole transition' % which_data)
			conds = pd.Series(['whole (boxcar) transition response'], name="conditions")
			pal = sn.dark_palette("palegreen", 1)

			sn.tsplot(dec_time_courses_3_s, err_style="ci_band", condition = conds, time = time_points, color = pal) # ci=cis, 
			pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
			sn.despine(offset=10, trim=True)
			ax1.set_ylabel('Z')
			ax1.set_xlabel('time [s]')

			# permute_sig_points = []
			# self.logger.info('permutation stats on division %s'%d)
			# clusters = mne.stats.permutation_cluster_1samp_test(dec_time_courses_3_s)[1]
			# if len(clusters) == 0:
			# 	sig_timepoints_permute = time_points[np.zeros(pvals_permute.shape[0], dtype = bool)]
			# else:
			# 	sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			# 	sig_timepoints_permute = time_points[sig_indices]
			# 	permute_sig_points.append(sig_timepoints_permute)

			# ax1.plot(permute_sig_points, np.zeros(len(permute_sig_points)) + 0.01 * i, ['k'], ms = 4.5, alpha = 0.2)

			pl.tight_layout()
			
			self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + roi_name +'_whole_boxcar.pdf'))
			pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + roi_name + '_whole_boxcar.pdf'))



	def plot_rsq_across_subjects_pupil(self, data_type = 'pupil_bp', hdf_file = 'eye', which_data = 'pupil', event_division = 10, nr_reps_for_btstr = 1000):

		all_rsq = []
		
		rsq = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'rsq_' + str(event_division), hdf_file = hdf_file)
		rsq = np.array(rsq).squeeze()
		rsq = rsq.T - rsq.mean(axis = 1)
		rsq /= rsq.std(axis = 0)
		rsq = rsq.T
		all_rsq.append(rsq)



		all_rsq = np.array(all_rsq)
		all_rsq = all_rsq.transpose((1,2,0))

		time_points = np.linspace(0,1,all_rsq.shape[1], endpoint = True)
		pal = sn.husl_palette( 4)	


		f = pl.figure(figsize = (5,8))
		ax1 = f.add_subplot(211)
		ax1.set_title('%s r2 10 different transition responses' % which_data)
		conds = pd.Series('pupil response', name="data type")
		# pal = sn.color_palette("Set2", 4)
		sn.tsplot(all_rsq, err_style="ci_band", time = time_points, condition = conds, color = pal ) # ci=cis, 
		pl.axhline(0, lw=0.25, alpha=0.25, color = 'k')
		pl.axvline((all_rsq.shape[1]-1), lw=0.25, alpha=0.25, color = 'k')

		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('normalized r-squared')
		ax1.set_xlabel('relative time of transition event within duration ')

		

		# pl.figure(figsize = (10,4))
		ax2 = f.add_subplot(212)

		all_btstr_res = np.zeros((1, nr_reps_for_btstr))
		for region in np.arange(all_rsq.shape[-1]):
			indices = np.random.randint(0,all_rsq.shape[0], size = (nr_reps_for_btstr, all_rsq.shape[0]))
			btstr_timelines = all_rsq[indices,:,region].mean(axis = 1)
			btstr_res = np.arctanh(np.array([sp.stats.spearmanr(np.linspace(0,1,event_division+1), bt) for bt in btstr_timelines])[:,0])
			pl.hist(btstr_res[(btstr_res != np.inf) * (btstr_res != -np.inf)], bins = 100, range = [-4,4], alpha = 0.4, cumulative = True, histtype = 'step', normed = True, linewidth = 2.5, color = pal[region])

			print btstr_res[(btstr_res != np.inf) * (btstr_res != -np.inf)].mean(), btstr_res[(btstr_res != np.inf) * (btstr_res != -np.inf)].std()
			all_btstr_res[region] = btstr_res
		pl.axhline(0.5, lw=0.25, alpha=0.25, color = 'k')
		pl.axvline(0.0, lw=0.25, alpha=0.25, color = 'k')
		ax2.set_ylim([-0.1, 1.1])
		ax2.set_xlim([-3.7, 3.7])
		sn.despine(offset=10, trim=True)
		ax2.set_ylabel('cumulative bootstrap distro')
		ax2.set_xlabel('Z-transformed correlation begin to end')
		pl.tight_layout()

		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type + '_r2.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type + '_r2.pdf'))


	def basic_deconv_results_across_multiple_events_and_subjects_pupil(self, data_type = 'pupil_bp',  hdf_file = 'eye', which_data = 'pupil', smooth_width=7):
		"""basic_deconv_results_across_subjects takes timepoints from deconvolve_colour_sound, and averages across subjects.
		"""
		self.logger.info('plotting across-subjects deconvolution results of the %s event-related responses.'%which_data)

		dec_time_courses_1 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_1', hdf_file = hdf_file)
		dec_time_courses_5 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_5', hdf_file = hdf_file)
		dec_time_courses_6 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_6', hdf_file = hdf_file)
		dec_time_courses_7 = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'dec_time_course_7', hdf_file = hdf_file)

		time_points = self.gather_data_from_hdfs(group = 'basic_' + which_data + '_deconvolution_' + data_type, data_type = 'time_points', hdf_file = hdf_file).mean(axis = 0)
		baseline_times = (time_points < 0.0) * (time_points > -1.0)

		# Positive mapping region
		dec_time_courses_1_s = np.array([[myfuncs.smooth(dec_time_courses_1[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_1.shape[0])] for j in range(dec_time_courses_1.shape[-1])]).transpose((1,2,0))
		dec_time_courses_5_s = np.array([[myfuncs.smooth(dec_time_courses_5[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_5.shape[0])] for j in range(dec_time_courses_5.shape[-1])]).transpose((1,2,0))
		dec_time_courses_6_s = np.array([[myfuncs.smooth(dec_time_courses_6[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_6.shape[0])] for j in range(dec_time_courses_6.shape[-1])]).transpose((1,2,0))
		dec_time_courses_7_s = np.array([[myfuncs.smooth(dec_time_courses_7[i,:,j], window_len=smooth_width) for i in range(dec_time_courses_7.shape[0])] for j in range(dec_time_courses_7.shape[-1])]).transpose((1,2,0))

		time_courses = np.array([np.squeeze(dec_time_courses_5_s),np.squeeze(dec_time_courses_6_s),np.squeeze(dec_time_courses_7_s)]).transpose(1,2,0)

		sn.set(style="ticks")

		# Stimulus on and off
		f = pl.figure(figsize = (5,4))
		ax1 = f.add_subplot(111)
		ax1.set_title('%s data 3 transition responses' % which_data)

		permute_sig_points = []
		for i in range(time_courses.shape[-1]): # loop across regions
			self.logger.info('permutation stats on %s'%['trans start','trans halfway','trans end'][i])
			clusters = mne.stats.permutation_cluster_1samp_test(time_courses[:,:,i])[1]
			sig_indices = np.concatenate([np.arange(time_points.shape[0])[c] for c in clusters])
			sig_timepoints_permute = time_points[sig_indices]
			permute_sig_points.append(sig_timepoints_permute)

		conds = pd.Series(['Trans start','Trans halfway','Trans end'], name="condition")

		for i in range(time_courses.shape[-1]):
			ax1.plot(permute_sig_points[i], np.zeros(len(permute_sig_points[i])) + 0.01 * i, ['r','k','c'][i], ms = 4.5, alpha = 0.2)
		sn.tsplot(time_courses[:,:,:], err_style="ci_band", condition = conds, time = time_points, color = ['red','black','cyan']) # ci=cis, 
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		sn.despine(offset=10, trim=True)
		ax1.set_ylabel('Z')
		ax1.set_ylim([-0.1,0.1])

		pl.tight_layout()
		
		self.logger.info(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_transevents_sig.pdf'))
		pl.savefig(os.path.join(self.grouplvl_plot_dir, 'basic_deconvolution_' + data_type  + which_data + '_transevents_sig.pdf'))

	


