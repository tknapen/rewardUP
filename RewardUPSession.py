#!/usr/bin/env python
# encoding: utf-8

" BR TRANS Session.py "

import os, sys, datetime, pickle
import math
import re

import numpy as np
import scipy as sp
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as pl
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import seaborn as sn
import glob
import subprocess
import tables as tb
from nifti import NiftiImage

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from tables import NoSuchNodeError, open_file
from sklearn.neighbors import KernelDensity

from joblib import Parallel, delayed
import itertools
from itertools import chain

import logging, logging.handlers, logging.config

sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.log import *
from Tools.Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator, ImageOperator
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data
from Tools.Operators.CommandLineOperator import ExecCommandLine, FSLMathsOperator, FEATOperator, FlirtOperator, BETOperator, VolToSurfOperator, SurfToSurfOperator
from Tools.other_scripts.plotting_tools import *
from Tools.Sessions import Session
from Tools.Operators import *
from Tools.Operators.ImageOperator import *
from Tools.Operators.ArrayOperator import DeconvolutionOperator

from Tools.other_scripts import savitzky_golay as savitzky_golay

from IPython import embed as shell

class RewardUPSession(Session):
	"""RewardUPSession"""
	def __init__(self, ID, date, project, subject, parallelize = True, loggingLevel = logging.DEBUG):
		super(RewardUPSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel)
		self.hdf5_mri_filename = os.path.join(self.stageFolder(stage = 'processed/mri'), 'all.hdf5')
		self.hdf5_eye_filename = os.path.join(self.stageFolder(stage = 'processed/eye'), 'all.hdf5')
		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_eye_filename)

	def import_edf_data(self, aliases = []):
		"""import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
		try:
			subprocess.Popen('rm ' + self.hdf5_eye_filename, shell=True, stdout=subprocess.PIPE).communicate()
		except OSError:
			self.logger.info('file ' + self.hdf5_eye_filename + ' does not exist already.')
			pass
	
		for r in self.runList:
			if r.indexInSession in self.scanTypeDict['epi_bold']:
				run_name = os.path.split(self.runFile(stage = 'processed/eye', run = r, extension = ''))[-1]
				edf_file = subprocess.Popen('ls ' + self.runFolder(stage = 'processed/eye', run = r) + '/*.edf', shell=True, stdout=PIPE).communicate()[0].split('\n')[0]
				for ext in ['msg', 'gaz', 'gaz.gz', 'hdf5']:	# remove intermediate files - have them be created by the ho.
					try:
						subprocess.Popen('rm ' + os.path.splitext(edf_file)[0] + '.' + ext, shell=True, stdout=subprocess.PIPE).communicate()
					except OSError:
						self.logger.info('file ' + os.path.splitext(edf_file)[0] + '.' + ext + ' does not exist already.')
						pass
				self.ho.add_edf_file(edf_file)
				self.ho.edf_message_data_to_hdf(alias = run_name)
				self.ho.edf_gaze_data_to_hdf(alias = run_name, pupil_hp = 0.05, pupil_lp = 6)


	def orientation_events_from_pickle(self):
		"""orientation_events_from_pickle takes the pickle file of the ori_mapper condition and splits the events up depending on trial types.
		These events are then saved in fsl txt format, for later analysis.
		"""
		p_file = self.runFile(stage = 'processed/behavior', run = self.runList[self.conditionDict['ori_mapper'][0]], extension = '.dat')
		with open(p_file) as f:
			ori_event_data = pickle.load(f)
		parameters = ori_event_data['parameterArray']
		events = ori_event_data['eventArray']

		# shell()
		# when did this run start?
		run_start_time = float([ev for ev in events[0] if type(ev) == str and ev.split(' ')[2] == 'phase'][0].split(' ')[-1])

		# fill array with onset times and orientations
		trial_array = []
		for p, e in zip(parameters, events):
			trial_array.append([
				float(p['grating_orientation']),
				float([ev for ev in e if type(ev) == str and ev.split(' ')[2] == 'phase'][1].split(' ')[-1]) - run_start_time
			])

		trial_array = np.array(trial_array)

		which_orientations = np.unique(trial_array[:,0])
		np.sort(which_orientations)

		# shell()
		# save files
		for i,ori in enumerate(which_orientations):
			these_trials = trial_array[trial_array[:,0] == ori]
			these_trials_formatted = np.array([ these_trials[:,-1], np.ones(these_trials[:,-1].shape) * 0.5, np.ones(these_trials[:,-1].shape) ])
			np.savetxt(
				self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['ori_mapper'][0]], base = 'events/ori_%i'%i, extension = '.txt'),
				these_trials_formatted.T, 
				delimiter = '\t', fmt = '%3.1f' )

	def reward_events_from_pickle(self):
		"""orientation_events_from_pickle takes the pickle file of the ori_mapper condition and splits the events up depending on trial types.
		These events are then saved in fsl txt format, for later analysis.
		"""
		# shell()
		p_files = [self.runFile(stage = 'processed/behavior', run = self.runList[self.conditionDict[c][i]], extension = '.dat') for c in ['P','UP'] for i in [0,1]]
		op_file_format = [self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict[c][i]], base = 'events/XXXX', extension = '.txt') for c in ['P','UP'] for i in [0,1]]

		for p_file, op_ff in zip(p_files, op_file_format):
			with open(p_file) as f:
				ori_event_data = pickle.load(f)
			parameters = ori_event_data['parameterArray']
			events = ori_event_data['eventArray']

			# shell()
			# when did this run start?
			run_start_time = float([ev for ev in events[0] if type(ev) == str and ev.split(' ')[2] == 'phase'][0].split(' ')[-1])

			# fill array with onset times and orientations
			trial_array = []
			for p, e in zip(parameters, events):
				trial_array.append([
					int(float(p['grating_orientation'])),
					int(p['sound_index']==1),
					float([ev for ev in e if type(ev) == str and ev.split(' ')[2] == 'phase'][1].split(' ')[-1]) - run_start_time
				])

			trial_array = np.array(trial_array)

			trial_type_dict = {'CW-R':[45, 1], 'CW-NR':[45, 0], 'CCW-R':[135, 1], 'CCW-NR':[135, 0], 'F-R':[0, 1], 'F-NR':[0, 0]}

			# save files
			for tt in trial_type_dict.keys():
				these_trials = (trial_array[:,0] == trial_type_dict[tt][0]) * (trial_array[:,1] == trial_type_dict[tt][1])
				print tt, str(these_trials.sum())
				if these_trials.sum() > 0:
					these_trials = trial_array[these_trials]
					these_trials_formatted = np.array([ these_trials[:,-1], np.ones(these_trials[:,-1].shape) * 0.5, np.ones(these_trials[:,1].shape) ])
					np.savetxt(
						op_ff.replace('XXXX', tt),
						these_trials_formatted.T, 
						delimiter = '\t', fmt = '%3.1f' )


	def grab_B0_residuals(self):
		# copy:
		for er in self.scanTypeDict['epi_bold']:
			copy_in = self.runFile(stage = 'processed/mri', run = self.runList[er], postFix = ['NB'], extension = '.feat') + '/filtered_func_data.nii.gz'
			copy_out = self.runFile(stage = 'processed/mri', run = self.runList[er], postFix = ['B0'])
			subprocess.Popen('cp ' + copy_in + ' ' + copy_out, shell=True, stdout=subprocess.PIPE).communicate()[0]

	def grab_events_for_deco(self):
		
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['BR'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		if tr > 10:
			tr = tr / 1000.0
		run_duration = tr * nr_trs
		
		conds = ['percept_one_button','transition_button','percept_two_button']
		stim_labels = ['CCW_RG','CW_RG','CCW_GR','CW_GR']

		# check in the pupil data
		if not hasattr(self, 'pupil_data'):
			self.collect_pupil_data_from_hdf(condition = 'BR', event_types = conds, data_type = 'pupil_bp')
		if not hasattr(self,'h5file_mri_filename'):
			h5file = tb.open_file(self.hdf5_mri_filename, mode = 'r')
	
		event_data = []
		event_durations = []
		blink_events = []
		ms_events = []
		stim_events = []
		it_events =[]
		bit_events = []
		half_trans_events = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events')

			this_blink_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_Blinks.txt'))[:,0]
			this_blink_events += nr_runs * run_duration
			blink_events.extend(this_blink_events)
			this_ms_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_MicroSaccades.txt'))[:,0]
			this_ms_events += nr_runs * run_duration
			ms_events.extend(this_ms_events)
			this_it_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_instant_trans.txt'))[:,0]
			this_it_events += nr_runs * run_duration
			it_events.extend(this_it_events)
			this_half_trans_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_halfway_trans.txt'))[:,0]
			this_half_trans_events += nr_runs * run_duration
			half_trans_events.extend(this_half_trans_events)
		
			this_bit_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_blink_induced_trans.txt'))
			
			if np.shape(this_bit_events) == (3,):
				this_bit_events = this_bit_events[0]
				this_bit_events += nr_runs * run_duration
				bit_events.append(this_bit_events) 
			
			elif np.shape(this_bit_events) >= (1.,1.):
				this_bit_events = this_bit_events[:,0]
				this_bit_events += nr_runs * run_duration
				bit_events.extend(this_bit_events) 
			
			# Get and filter event data, so there are no instant and blink induced transitions
			this_run_durations = []
			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,0])	# toss out last and first event of each type to make sure there are no strange spill-over effects
				this_run_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
			stim_events.extend( [np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + stim_name + '.txt'))[[0,0]] + nr_runs * run_duration for stim_name in stim_labels] )
			event_data.append(this_run_events)
			event_durations.append(this_run_durations)
			
			nr_runs += 1
		
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		event_durations = [np.concatenate([e[i] for e in event_durations]) for i in range(len(event_durations[0]))]
		self.stim_events = np.array([se[0] for se in stim_events])
		
		# leave out dubble defined events 
		for ev in np.concatenate([it_events,bit_events]):
			if ev in event_data[0]:
				filter_me_idx = np.where(ev == event_data[0])[0][0]
				event_data[0][filter_me_idx] = 0.
				event_durations[0][filter_me_idx] = 0.
			
			elif ev in event_data[2]:
				filter_me_idx = np.where(ev == event_data[2])[0][0]
				event_data[2][filter_me_idx] = 0.
				event_durations[2][filter_me_idx] = 0.
			
			elif ev in event_data[1]:
				filter_me_idx = np.where(ev == event_data[1])[0][0]
				event_data[1][filter_me_idx] = 0.
				event_durations[1][filter_me_idx] = 0.
				
		event_data = [event_data[0][event_data[0] != 0.], event_data[1][event_data[1] != 0.], event_data[2][event_data[2] != 0.]]
		event_durations = [event_durations[0][event_durations[0] != 0.], event_durations[1][event_durations[1] != 0.], event_durations[2][event_durations[2] != 0.]]
		
		h5file.close()
		return event_data, bit_events, it_events, blink_events, ms_events, stim_events, event_durations, half_trans_events
		


	def grab_retroicor_residuals(self, conditions, postFix, nr_additions = 50):
		self.logger.info('grabbing retroicor residuals from %s, %s' % (str(conditions), str(postFix)))
		# maths_operators = []
		# second_order_maths_operators = [[] for i in range(10)]
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				# copy:
				copy_in = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '') + '.feat/stats/res4d.nii.gz'
				copy_out = self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['phys'])
				subprocess.Popen('cp ' + copy_in + ' ' + copy_out, shell=True, stdout=subprocess.PIPE).communicate()[0]

				# fix headers:
				nii_file_orig = NiftiImage(self.runFile(stage = 'processed/mri', run = r ))
				nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['phys']))
				nii_file.header = nii_file_orig.header
				nii_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['phys']))

				phys_data = nii_file.data
				add_data = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['meanvol'])).data
				new_data = phys_data + (nr_additions*add_data)

				add_file = NiftiImage(new_data)
				add_file.header = nii_file.header
				add_file.save(self.runFile(stage = 'processed/mri', run = r, postFix = postFix + ['phys','add']))				


	def register_feats(self, condition = 'mapper', postFix = ['mcf','phys','add','0','w_blinks']):
		"""run featregapply for all feat direcories in this session."""
		for run in [self.runList[i] for i in self.conditionDict[condition]]:
			feat_dir_name = self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat')
			self.setupRegistrationForFeat(feat_dir_name)

	def button_press_analysis_for_run(self, run):
		"""
		button_press_analysis takes a run object as argument and analyses:
		timing of button presses in general, and 
		how they relate to the perceptual sequence.
		This method saves its results in txt files that are readable by fsl.
		"""
		self.logger.info('running button press analysis for run %s'%str(run))
		alias = os.path.split(self.runFile(stage = 'processed/eye', run = run, extension = ''))[-1]
		trial_phases = self.ho.read_session_data(alias, 'trial_phases')
		trial_parameters = self.ho.read_session_data(alias, 'parameters')
		all_events = self.ho.read_session_data(alias, 'events')

		# labels for trials and event types
		stim_labels = ['CCW_RG','CW_RG','CCW_GR','CW_GR']
		event_labels = ['percept_one_button', 'transition_button', 'percept_two_button']

		# divide up trials based on parameters 
		motion_indices = [np.array(trial_parameters['motion_direction'] == -1), np.array(trial_parameters['motion_direction'] == 1)]
		stim_eye_correspondence_indices = [np.array(trial_parameters['stim_eye_correspondence'] == -1), np.array(trial_parameters['stim_eye_correspondence'] == 1)]
		label_trial_index_dict = {}
		for i, label in enumerate(stim_labels):
			label_trial_index_dict.update({label: np.arange(len(trial_parameters))[ motion_indices[i%2] * stim_eye_correspondence_indices[int(math.floor(i/2.0))] ] })

		# general times for trials, and button presses
		run_start_time = np.array(trial_phases[trial_phases['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0]
		trial_start_times = np.array(trial_phases[trial_phases['trial_phase_index'] == 2]['trial_phase_EL_timestamp'])
		trial_end_times = np.array(trial_phases[trial_phases['trial_phase_index'] == 3]['trial_phase_EL_timestamp'])
		button_presses_down =  all_events[(all_events['up_down'] == 'Down') & 
								((all_events['scancode'] == 5) | (all_events['scancode'] == 16) | (all_events['scancode'] == 11))][['scancode','EL_timestamp']]
		#
		# 	re-phrase the events. This is now done arbitrary - I haven't looked up which finger is which number here.
		#	as long as it's done consistently, right?
		#
		for key, new_key in zip([5,16,11],[-1,0,1]):
			button_presses_down.loc[button_presses_down['scancode'] == key, 'scancode'] = new_key

		# Loop across presentations to create figures for future reference.
		f = pl.figure(figsize = (16,8))
		button_times = [[],[],[]]
		for i in range(len(trial_parameters)):
			this_trial_button_events = np.array(button_presses_down[(button_presses_down['EL_timestamp'] < trial_end_times[i]) & (button_presses_down['EL_timestamp'] > trial_start_times[i])])
			# re-zero times in reference to stimulus appearance
			this_trial_button_events_for_plot = np.repeat(this_trial_button_events, 2, axis = 0)
			this_trial_button_events_for_plot = this_trial_button_events_for_plot[1:,1] - trial_start_times[i], this_trial_button_events_for_plot[:-1,0]

			for k, percept in enumerate([-1,0,1]):
				start_times = np.arange( (this_trial_button_events[:,0] == percept).shape[0] )[this_trial_button_events[:,0] == percept]
				end_times = np.arange( (this_trial_button_events[:,0] == percept).shape[0] )[this_trial_button_events[:,0] == percept] + 1
				button_times[k].append( [this_trial_button_events[start_times[:-1],-1] - run_start_time, this_trial_button_events[end_times[:-1],-1] - run_start_time] ) # drop last event of each type from the trial's timeslist

			# plot
			s = f.add_subplot(len(trial_parameters), 1, i+1)
			pl.plot(this_trial_button_events_for_plot[0], this_trial_button_events_for_plot[1], 'r', linewidth = 2.0)
			s.axis([0,150000,-1.1,1.1])
			s.axhline(0, color = 'k', linewidth = 0.25)

		fn = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = ['perception'], extension = '.pdf'))[-1]
		f.savefig(os.path.join(self.stageFolder(stage = 'processed/behavior/figs'), fn))

		stacked_button_times = [np.hstack(button_times[i]) for i in [0,1,2]]
		stacked_stim_on_times = np.array([trial_start_times - run_start_time, trial_end_times - run_start_time]).T

		# try to write these files away. First try to create a directory for the events.
		event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = run), 'events')
		try:
			os.mkdir(event_directory)
		except OSError:
			self.logger.info('event directory exists already.')

		for i, event_name in enumerate(event_labels):
			# print os.path.join(event_directory, str(run.ID) + '_' + event_name + '.txt')
			these_events = np.array([stacked_button_times[i][0,:] / 1000.0, (stacked_button_times[i][1,:] - stacked_button_times[i][0,:]) / 1000.0, np.ones(len(stacked_button_times[i][0]))]).T
			np.savetxt(os.path.join(event_directory, str(run.ID) + '_' + event_name + '.txt'), these_events, fmt = '%4.2f', delimiter='\t')

		trial_onset_times = (trial_start_times - run_start_time) / 1000.0
		for i, stim_name in enumerate(stim_labels):
			this_trial_np_array = np.array([[trial_onset_times[label_trial_index_dict[stim_name][0]], 150.0, 1]])
			np.savetxt(os.path.join(event_directory, str(run.ID) + '_' + stim_name + '.txt'), this_trial_np_array, fmt = '%4.2f', delimiter='\t')

		# Add blinks to events folder
		
		#load in blink data
		session_start_EL_time = np.array(trial_phases[trial_phases['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
		session_stop_EL_time = np.array(trial_end_times)[-1]
		eye = self.ho.eye_during_period([session_start_EL_time, session_stop_EL_time], alias)
				
		eyelink_blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
		eyelink_blink_data_L = eyelink_blink_data[eyelink_blink_data['eye'] == eye] #only select data from left eye
		b_start_times = np.array(eyelink_blink_data_L.start_timestamp)
		b_end_times = np.array(eyelink_blink_data_L.end_timestamp)
 
 		#other stuff we need
 		self.sample_rate = 1000.0
 		blink_durations = []
 		
		#evaluate only blinks that occur after start and before end experiment
		b_indices = (b_start_times>session_start_EL_time)*(b_end_times<session_stop_EL_time) 
		b_start_times_t = (b_start_times[b_indices] - session_start_EL_time) #valid blinks (start times) 
		b_end_times_t = (b_end_times[b_indices] - session_start_EL_time) 
		
		blinks = np.array(b_start_times_t /self.sample_rate)
		blink_durations.append(((b_end_times_t-b_start_times_t) /self.sample_rate ))
		blink_regressor = np.array(np.vstack([blinks,  blink_durations[0], np.ones(len(blinks))])).T
		
		np.savetxt(os.path.join(event_directory, str(run.ID) + '_Blinks.txt'), blink_regressor, fmt = '%4.2f', delimiter='\t')

	def button_press_analysis(self, conditions = ['BR']):
		for cond in conditions:

			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				self.button_press_analysis_for_run(r)


	def feat_analysis_rivalry_run(self, run, postFix = ['mcf','phys','add','sgtf'], analysis_type = 'trans_feat', feat_name = '_no_blinks_stim', smooth_mm = 0, run_feat = True):

		self.logger.info('running feat analysis of type %s to produce feat dir %s'%
							(analysis_type, self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat')))

		feat_dir_name = self.runFile(stage = 'processed/mri', run = run, postFix = postFix + [str(smooth_mm)] + [feat_name], extension = '.feat')
		fsf_file_name = self.runFile(stage = 'processed/mri', run = run, postFix = postFix + [str(smooth_mm)] + [feat_name], extension = '.fsf')

		self.button_press_analysis_for_run(run)

		if run_feat:
			try:
				os.system('rm -rf ' + feat_dir_name)
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + fsf_file_name)
			except OSError:
				pass
			self.logger.info('running feat transition analysis for run %s'%str(run))

			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
			input_feat_file = '/home/kupers/code/git/BR_transition/feats/' + analysis_type + '.fsf'
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = run), 'events')

			REDict = {
			'---NII_FILE---': 				self.runFile(stage = 'processed/mri', run = run, postFix = postFix), 
			'---NR_TRS---':					str(NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = postFix)).timepoints),
			'---MCF_PAR---': 				self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks','T']), 	
			'---CW_RG_FILE---': 			os.path.join(event_directory, str(run.ID) + '_' + 'CW_RG' + '.txt'), 	
			'---CW_GR_FILE---': 			os.path.join(event_directory, str(run.ID) + '_' + 'CW_GR' + '.txt'), 	
			'---CCW_RG_FILE---': 			os.path.join(event_directory, str(run.ID) + '_' + 'CCW_RG' + '.txt'), 	
			'---CCW_GR_FILE---': 			os.path.join(event_directory, str(run.ID) + '_' + 'CCW_GR' + '.txt'), 	
			'---PERCEPT_ONE_BUTTON---': 	os.path.join(event_directory, str(run.ID) + '_' + 'percept_one_button' + '.txt'), 
			'---PERCEPT_TWO_BUTTON---': 	os.path.join(event_directory, str(run.ID) + '_' + 'percept_two_button' + '.txt'), 
			'---TRANSITION_BUTTON---': 		os.path.join(event_directory, str(run.ID) + '_' + 'transition_button' + '.txt'), 
			'---SMOOTH_MM---': 				str(smooth_mm), 
			'---OUTPUT_DIR---':				feat_dir_name,
			'---BLINKS_FILE---':			os.path.join(event_directory, str(run.ID) + '_' + 'Blinks' + '.txt')
			}
			featOp = FEATOperator(inputObject = input_feat_file)
			# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
			if run == [self.runList[i] for i in self.conditionDict['BR']][-1]:
				featOp.configure( REDict = REDict, featFileName = fsf_file_name, waitForExecute = True )
			else:
				featOp.configure( REDict = REDict, featFileName = fsf_file_name, waitForExecute = False )
			self.logger.debug('Running feat from ' + input_feat_file + ' as ' + fsf_file_name)
			# run feat
			featOp.execute()

	def feat_analysis_rivalry(self, postFix = ['mcf','phys','add'], analysis_type = 'trans_clean_stim_blinks_feat', feat_name = '_no_blinks_stim',smooth_mm = 0):
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			self.feat_analysis_rivalry_run(r, postFix = postFix, analysis_type = analysis_type, feat_name = feat_name, smooth_mm = smooth_mm)
			
			
	def feat_analysis_mapper_run(self, run, postFix = ['mcf','phys','add', 'sgtf'], analysis_type = 'mapper_feat_v2', smooth_mm = 0, run_feat = True):

		self.logger.info('running feat analysis of type %s to produce feat dir %s'%
							(analysis_type, self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat')))

		feat_dir_name = self.runFile(stage = 'processed/mri', run = run, postFix = postFix + [str(smooth_mm) + '_mapper'], extension = '.feat')
		fsf_file_name = self.runFile(stage = 'processed/mri', run = run, postFix = postFix + [str(smooth_mm) + '_mapper'], extension = '.fsf')

# 		self.button_press_analysis_for_run(run)

		if run_feat:
			try:
				os.system('rm -rf ' + feat_dir_name)
 				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix, extension = '.feat'))
				os.system('rm -rf ' + fsf_file_name)
			except OSError:
				pass
			self.logger.info('running feat transition analysis for run %s'%str(run))

			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# the order of the REs here, is the order in which they enter the feat. this can be used as further reference for PEs and the like.
			input_feat_file = '/home/kupers/code/git/BR_transition/feats/' + analysis_type + '.fsf'
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = run), 'events')

			REDict = {
			'---NII_FILE---': 				self.runFile(stage = 'processed/mri', run = run, postFix = postFix), 
			'---NR_TRS---':					str(NiftiImage(self.runFile(stage = 'processed/mri', run = run, postFix = postFix)).timepoints),
 			'---MCF_PAR---': 				self.runFile(stage = 'processed/mri', run = run, extension = '.txt', postFix = ['blinks','T']), 	
			'---CW_Red_R_FILE---': 			os.path.join(event_directory, str(run.ID) + '_' + 'CW_Red_R' + '.txt'), 	
			'---CW_Green_R_FILE---': 		os.path.join(event_directory, str(run.ID) + '_' + 'CW_Green_R' + '.txt'), 	
			'---CCW_Red_R_FILE---': 		os.path.join(event_directory, str(run.ID) + '_' + 'CCW_Red_R' + '.txt'), 	
			'---CCW_Green_R_FILE---': 		os.path.join(event_directory, str(run.ID) + '_' + 'CCW_Green_R' + '.txt'), 
			'---CW_Red_L_FILE---': 			os.path.join(event_directory, str(run.ID) + '_' + 'CW_Red_L' + '.txt'), 	
			'---CW_Green_L_FILE---': 		os.path.join(event_directory, str(run.ID) + '_' + 'CW_Green_L' + '.txt'), 	
			'---CCW_Red_L_FILE---': 		os.path.join(event_directory, str(run.ID) + '_' + 'CCW_Red_L' + '.txt'), 	
			'---CCW_Green_L_FILE---': 		os.path.join(event_directory, str(run.ID) + '_' + 'CCW_Green_L' + '.txt'),
			'---BLINKS_FILE---':			os.path.join(event_directory, str(run.ID) + '_' + 'Blinks' + '.txt'),
			'---SMOOTH_MM---': 				str(smooth_mm), 
			'---OUTPUT_DIR---':				feat_dir_name
			}
			featOp = FEATOperator(inputObject = input_feat_file)
			# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
			if run == [self.runList[i] for i in self.conditionDict['mapper']][-1]:
				featOp.configure( REDict = REDict, featFileName = fsf_file_name, waitForExecute = True )
			else:
				featOp.configure( REDict = REDict, featFileName = fsf_file_name, waitForExecute = False )
			self.logger.debug('Running feat from ' + input_feat_file + ' as ' + fsf_file_name)
			# run feat
			featOp.execute()

	def feat_analysis_mapper(self, postFix = ['mcf','phys','add'], analysis_type = 'mapper_feat_v2', smooth_mm = 0):
		for r in [self.runList[i] for i in self.conditionDict['mapper']]:
			self.feat_analysis_mapper_run(r, postFix = postFix, analysis_type = analysis_type, smooth_mm = smooth_mm)


	def gfeat_analysis_rivalry(self, postFix = ['mcf','phys','add'], analysis_type = 'trans_gfeat', smooth_mm = 0, feat_name = 'blinks'):

		input_feat_file = '/home/kupers/code/git/BR_transition/feats/' + analysis_type + '.fsf'
		gfeat_dir_name = os.path.join(self.stageFolder(stage = 'processed/mri/BR'), 'gfeat_%imm_'%smooth_mm + (feat_name))
		gfeat_fsf_name = os.path.join(self.stageFolder(stage = 'processed/mri/BR'), 'gfeat_%imm_'%smooth_mm + (feat_name) + '.fsf')
		feat_dir_names = [self.runFile(stage = 'processed/mri', run = run, postFix = postFix + [str(smooth_mm)] + [feat_name], extension = '.feat') for run in [self.runList[i] for i in self.conditionDict['BR']]]
		nr_rivalry_runs = len([self.runList[i] for i in self.conditionDict['BR']])
		self.logger.info('starting gfeat combination across runs for %s into %s using %s'%(str(feat_dir_names), gfeat_dir_name, input_feat_file))

		# set the strings to be filled into the fsf file
		# set feat_files(INDEX) "FEAT_DIR"
		feat_definitions = '\n'.join(['set feat_files(%i) "%s"'%(i+1, fdn) for i, fdn in enumerate(feat_dir_names)])
		# set fmri(evgINDEX.1) 1
		ev_value_definitions = '\n'.join(['set fmri(evg%i.1) 1'%(i+1) for i in range(nr_rivalry_runs)]) 
		# set fmri(groupmem.INDEX) 1
		group_membership_definitions = '\n'.join(['set fmri(groupmem.%i) 1'%(i+1) for i in range(nr_rivalry_runs)])

		REDict = {
			'---NR_RIVALRY_RUNS---': 				str(nr_rivalry_runs),
			'---OUTPUT_DIR---':						gfeat_dir_name,
			'---FEAT_DEFINITIONS---': 				feat_definitions, 
			'---EV_VALUE_DEFINITIONS---': 			ev_value_definitions, 	
			'---GROUP_MEMBERSHIP_DEFINITIONS---': 	group_membership_definitions, 	
			}

		try:
			os.system('rm -rf ' + gfeat_dir_name+'.gfeat')
			os.system('rm -rf ' + gfeat_fsf_name)
			os.system('rm -rf ' + gfeat_dir_name+'+.gfeat')
			os.system('rm -rf ' + gfeat_dir_name+'++.gfeat')
		except OSError:
			pass

		featOp = FEATOperator(inputObject = input_feat_file)
		# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
		featOp.configure( REDict = REDict, featFileName = gfeat_fsf_name, waitForExecute = False )
		self.logger.debug('Running feat from ' + input_feat_file + ' as ' + gfeat_fsf_name)
		# run feat
		featOp.execute()
		
	def gfeat_analysis_mapper(self, postFix = ['mcf','phys','add'], analysis_type = 'mapper_gfeat', smooth_mm = 0):

		input_feat_file = '/home/kupers/code/git/BR_transition/feats/' + analysis_type + '.fsf'

		gfeat_dir_name = os.path.join(self.stageFolder(stage = 'processed/mri/mapper'), 'gfeat_%imm_mapper'%smooth_mm)
		gfeat_fsf_name = os.path.join(self.stageFolder(stage = 'processed/mri/mapper'), 'gfeat_%imm_mapper.fsf'%smooth_mm)
		feat_dir_names = [self.runFile(stage = 'processed/mri', run = run, postFix = postFix + [str(smooth_mm), '_mapper'], extension = '.feat') for run in [self.runList[i] for i in self.conditionDict['mapper']]]
		nr_mapper_runs = len([self.runList[i] for i in self.conditionDict['mapper']])
		self.logger.info('starting gfeat combination across runs for %s into %s using %s'%(str(feat_dir_names), gfeat_dir_name, input_feat_file))

		# set the strings to be filled into the fsf file
		# set feat_files(INDEX) "FEAT_DIR"
		feat_definitions = '\n'.join(['set feat_files(%i) "%s"'%(i+1, fdn) for i, fdn in enumerate(feat_dir_names)])
		# set fmri(evgINDEX.1) 1
		ev_value_definitions = '\n'.join(['set fmri(evg%i.1) 1'%(i+1) for i in range(nr_mapper_runs)])
		# set fmri(groupmem.INDEX) 1
		group_membership_definitions = '\n'.join(['set fmri(groupmem.%i) 1'%(i+1) for i in range(nr_mapper_runs)])

		REDict = {
			'---NR_RIVALRY_RUNS---': 				str(nr_mapper_runs),
			'---OUTPUT_DIR---':						gfeat_dir_name,
			'---FEAT_DEFINITIONS---': 				feat_definitions, 
			'---EV_VALUE_DEFINITIONS---': 			ev_value_definitions, 	
			'---GROUP_MEMBERSHIP_DEFINITIONS---': 	group_membership_definitions, 	
			}

		try:
			os.system('rm -rf ' + gfeat_dir_name+'.gfeat')
			os.system('rm -rf ' + gfeat_fsf_name)
			os.system('rm -rf ' + gfeat_dir_name+'+.gfeat')
			os.system('rm -rf ' + gfeat_dir_name+'++.gfeat')
		except OSError:
			pass

		featOp = FEATOperator(inputObject = input_feat_file)
		# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
		featOp.configure( REDict = REDict, featFileName = gfeat_fsf_name, waitForExecute = False )
		self.logger.debug('Running feat from ' + input_feat_file + ' as ' + gfeat_fsf_name)
		# run feat
		featOp.execute()


	def take_stats_to_session_space(self, condition = 'mapper', gfeat_name = 'gfeat_%imm', smooth_mm = 0, clear_all_stats = False):
		""""""
		# clean everything up.
		if clear_all_stats:
			os.system('rm -rf ' + self.stageFolder(stage = 'processed/mri/masks/stat/') + '*')
		# always clean up this gfeat's results.
		os.system('rm ' + os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), condition + '_' + (gfeat_name % smooth_mm) + '/*.nii.gz '))
		try:
			os.mkdir(self.stageFolder(stage = 'processed/mri/masks/stat/' + condition + '_' + (gfeat_name % smooth_mm)))
		except OSError:
			self.logger.info('event directory exists already.')

		gfeat_dir_name = os.path.join(self.stageFolder(stage = 'processed/mri'), condition, (gfeat_name % smooth_mm) + '.gfeat')
		which_copes = subprocess.Popen('ls -d ' + gfeat_dir_name + '/cope*.feat', shell=True, stdout=PIPE).communicate()[0].split('\n')[:-1]

		for which_cope in which_copes:
			for which_stat in ['zstat', 'tstat', 'pe', 'cope']:
				fO = FlirtOperator(os.path.join(which_cope, 'stats', '%s1.nii.gz'%which_stat), 
									referenceFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'example_func.nii.gz' ))

				this_cope_stat_name = os.path.split(which_cope)[-1].split('.')[0]
				fO.configureApply(transformMatrixFileName = os.path.join(self.stageFolder(stage = 'processed/mri/reg/feat'),'standard2example_func.mat' ), 
									outputFileName = self.runFile(stage = 'processed/mri/masks/stat/' + condition + '_' + (gfeat_name % smooth_mm), base = this_cope_stat_name, postFix = [which_stat] ), sinc = True)
				fO.execute()

	def stats_to_surface(self, which_smoothing_widths = [0,5]): 

		try:
			os.mkdir(self.stageFolder(stage = 'processed/mri/masks/stat/surf/'))
		except OSError:
			pass

		which_stat_mask_dirs = subprocess.Popen('ls ' + self.stageFolder(stage = 'processed/mri/masks/stat/'), shell=True, stdout=PIPE).communicate()[0].split('\n')[:-1]
		stat_files = {}
		for mask_dir in which_stat_mask_dirs:
			mask_files = subprocess.Popen('ls ' + os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), mask_dir, '*.nii.gz') , shell=True, stdout=PIPE).communicate()[0].split('\n')[:-1]
			for m in mask_files:
				stat_files.update({ mask_dir + '_' + os.path.split(m)[-1][:-7]: m})

		deco_files = subprocess.Popen('ls ' + os.path.join(self.stageFolder(stage = 'processed/mri/BR/deco/'), '*.nii.gz') , shell=True, stdout=PIPE).communicate()[0].split('\n')[:-1]
		for df in deco_files:
				stat_files.update({ 'deco_' + os.path.split(df)[-1][:-7]: df})

		run_type = 'mapper'
		r = [self.runList[i] for i in self.conditionDict['mapper']][0]
		for mm in which_smoothing_widths:
			feat_post_fix = 'mapper'
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','phys','add'] + [str(mm)] + [feat_post_fix], extension = '.feat')
			stat_files.update({
						# I have not seen how these regressors are counted, but these are the most important ones

						'stim_on_mapper_T' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'tstat13.nii.gz'),
						'stim_on_mapper_Z' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'zstat13.nii.gz'),
						'stim_on_mapper_cope' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'cope13.nii.gz'),
		
						'eye_T' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'tstat9.nii.gz'),
						'eye_Z' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'zstat9.nii.gz'),
						'eye_cope' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'cope9.nii.gz'),

						'color_T' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'tstat11.nii.gz'),
						'color_Z' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'zstat11.nii.gz'),
						'color_cope' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'cope11.nii.gz'),
			
						'motion_T' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'tstat10.nii.gz'),
						'motion_Z' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'zstat10.nii.gz'),
						'motion_cope' + '_%imm'%mm: os.path.join(this_feat, 'stats', 'cope10.nii.gz'),
			
						})

		#make per-subject surfaces
		for sf in stat_files.keys():
			vO = VolToSurfOperator(stat_files[sf])
			vO.configure(frames = { sf: 0}, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat'), 
							outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/surf/'), '') )
			vO.execute()

			# make average subjects surfaces
			for hemi in ['lh','rh']:
				sO = SurfToSurfOperator(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/surf/'), sf + '-' + hemi + '.mgz'))
				sO.configure(fsSourceSubject = self.subject.standardFSID,  fsTargetSubject = 'BR_avg_9s', hemi = hemi, outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/surf/'), sf + '_avg_subject-' + hemi + '.mgz'))
				sO.runcmd += ' &'
				sO.execute()
		# shell()


	
	def mapper_events_for_run(self, run):
		""" Behavior for mapper run """
		
		session_time = 0
		
		alias = os.path.split(self.runFile(stage = 'processed/eye', run = run, extension = ''))[-1]
				
		all_events = self.ho.read_session_data(alias, 'events')
		trial_times = self.ho.read_session_data(alias, 'trials')
		trial_phases = self.ho.read_session_data(alias, 'trial_phases')
		trial_parameters = self.ho.read_session_data(alias, 'parameters')
		
		trial_start_times = np.array(trial_phases[trial_phases['trial_phase_index'] == 2]['trial_phase_EL_timestamp'])
		trial_end_times = np.array(trial_phases[trial_phases['trial_phase_index'] == 3]['trial_phase_EL_timestamp'])

		# divide up trials based on parameters 
		motion_indices = [np.array(trial_parameters['motion_direction'] == -1), np.array(trial_parameters['motion_direction'] == 1)]
		stim_eye_correspondence_indices = [np.array(trial_parameters['stim_eye_correspondence'] == -1), np.array(trial_parameters['stim_eye_correspondence'] == 1)]
		which_eye_stimulated_indices = [np.array(trial_parameters['which_eye_stimulated'] == -1), np.array(trial_parameters['which_eye_stimulated'] == 1)]					
		
		condition_name = []
		motion_array = []
		color_array = []
		eye_array = []
		
		CCW_Red_L = []
		CW_Red_L = []
		
		CCW_Green_L = []
		CW_Green_L = []
		
		CCW_Red_R = []
		CW_Red_R = []
		
		CCW_Green_R = []
		CW_Green_R = []
		
		motion_indices = motion_indices[0]
		stim_eye_correspondence_indices = stim_eye_correspondence_indices[0]
		which_eye_stimulated_indices = which_eye_stimulated_indices[0]
		
		run_start_time = np.array(trial_phases[trial_phases['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0]
		
		for i in range(len(motion_indices)):
			motion_array.append(motion_indices[i]*1)
			color_array.append(stim_eye_correspondence_indices[i]*1)
			eye_array.append(which_eye_stimulated_indices[i]*1)
			
			
			if motion_array[i] == True and color_array[i] == True and eye_array[i] == True:
				CCW_Red_L.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			elif motion_array[i] == False and color_array[i] == True and eye_array[i] == True:
				CW_Red_L.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			
			elif motion_array[i] == True and color_array[i] == False and eye_array[i] == True:
				CCW_Green_L.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			elif motion_array[i] == False and color_array[i] == False and eye_array[i] == True:
				CW_Green_L.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			
			elif motion_array[i] == True and color_array[i] == True and eye_array[i] == False:
				CCW_Red_R.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			elif motion_array[i] == False and color_array[i] == True and eye_array[i] == False:
				CW_Red_R.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			
			elif motion_array[i] == True and color_array[i] == False and eye_array[i] == False:
				CCW_Green_R.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			elif motion_array[i] == False and color_array[i] == False and eye_array[i] == False:
				CW_Green_R.append(np.array([(trial_start_times[i]-run_start_time) / 1000.0, 20.0, 1.0]))
			
			
			
# 		shell()
# 		for i, m in enumerate(motion_indices):
# 			for j, c in enumerate(stim_eye_correspondence_indices):
# 				for k, e in enumerate(which_eye_stimulated_indices):

 					

		#load in blink data
		
		session_start_EL_time = np.array(trial_phases[trial_phases['trial_phase_index'] == 1]['trial_phase_EL_timestamp'])[0] # np.array(trial_times['trial_start_EL_timestamp'])[0]#
		session_stop_EL_time = np.array(trial_times['trial_end_EL_timestamp'])[-1]
		eye = self.ho.eye_during_period([session_start_EL_time, session_stop_EL_time], alias)
		
# 		session_time += session_stop_EL_time - session_start_EL_time
		
		eyelink_blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
		eyelink_blink_data_L = eyelink_blink_data[eyelink_blink_data['eye'] == eye] #only select data from left eye
		b_start_times = np.array(eyelink_blink_data_L.start_timestamp)
		b_end_times = np.array(eyelink_blink_data_L.end_timestamp)
 
 		#other stuff we need
 		self.sample_rate = 1000.0
 		blink_durations = []
 		
 		
		#evaluate only blinks that occur after start and before end experiment
		b_indices = (b_start_times>session_start_EL_time)*(b_end_times<session_stop_EL_time) 
		b_start_times_t = (b_start_times[b_indices] - session_start_EL_time) #valid blinks (start times) 
		b_end_times_t = (b_end_times[b_indices] - session_start_EL_time) 
		
		blinks = np.array(b_start_times_t /self.sample_rate)
		blink_durations.append(((b_end_times_t-b_start_times_t) /self.sample_rate ))
		blink_regressor = np.array(np.vstack([blinks,  blink_durations[0], np.ones(len(blinks))])).T
			
		conDict = {'CCW_Red_L': CCW_Red_L,
			'CW_Red_L': 		CW_Red_L,
			'CCW_Green_L': 		CCW_Green_L,
			'CW_Green_L': 		CW_Green_L,
			'CCW_Red_R': 		CCW_Red_R,
			'CW_Red_R': 		CW_Red_R,
			'CCW_Green_R': 		CCW_Green_R,
			'CW_Green_R': 		CW_Green_R,
			'Blinks':			blink_regressor}
			
		for i in range(len(conDict)):
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = run), 'events')
			try:
	 			os.mkdir(event_directory)
	 		except OSError:
				self.logger.info('event directory exists already.')
			
 			np.savetxt(os.path.join(event_directory, str(run.ID) + '_' + conDict.keys()[i] + '.txt'), conDict.values()[i], fmt = '%4.2f', delimiter='\t')
		
	def mapper_events(self, conditions = ['mapper']):
		for cond in conditions:
			for r in [self.runList[i] for i in self.conditionDict[cond]]:
				self.mapper_events_for_run(r)
				
				
	def blink_transition_behavior(self, interval = 5, data_type = 'pupil_bp'):
		
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['BR'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		if tr > 10:
			tr = tr / 1000.0
		run_duration = tr * nr_trs		
		
		conds = ['percept_one_button', 'transition_button', 'percept_two_button']
		
		event_data = []
		blink_events = []
		ms_events = []
		raw_percept_durations = []
		raw_trans_durations = []
		
		nr_runs = 0
		# get event files
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events')
			this_run_events = []
			these_percept_durations = []
			for cond in conds:
				this_run_events.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,0])	# toss out last and first event of each type to make sure there are no strange spill-over effects
				if cond in ['percept_one_button', 'percept_two_button']:
					these_percept_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
				elif cond == 'transition_button':
					raw_trans_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
			raw_percept_durations.append(these_percept_durations)
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
# 			stim_events.extend( [np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + stim_name + '.txt'))[[0,0]] + nr_runs * run_duration for stim_name in stim_labels] )
			event_data.append(this_run_events)
			this_blink_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_Blinks.txt'))[:,0]
			this_blink_events += nr_runs * run_duration
			blink_events.extend(this_blink_events)
			this_ms_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_MicroSaccades.txt'))[:,0]
			this_ms_events += nr_runs * run_duration
			ms_events.extend(this_ms_events)
			nr_runs += 1
		
		
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]
		percept_durations = [np.concatenate([e[i] for e in raw_percept_durations]) for i in range(len(raw_percept_durations[0]))]
		blink_events = np.hstack(blink_events)
		ms_events = np.hstack(ms_events)
		trans_durations = np.hstack(raw_trans_durations)
		
		X_plot = np.linspace(0, 10.0, 1000)[:, np.newaxis]
		kde_T = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(trans_durations)
		kde_P0 = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(percept_durations[0])
		kde_P1 = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(percept_durations[1])

		# dens = [kde.score_samples(X_plot) for kde in (kde_T, kde_P0, kde_P1)]

		cp = sn.color_palette("Set1", 3)
		bins_max = [15,15,15]
		bws = [0.25, 1.0, 1.0]
		nr_bins = 500
		f = pl.figure(figsize = (4,8))
		# f = pl.figure(figsize = (6,3))
		# s = f.add_subplot(111)
		with sn.axes_style("white"):
			for i, kde in enumerate([trans_durations, percept_durations[0], percept_durations[1]]):
				s = f.add_subplot(3,1,i+1)
				# sn.rugplot(kde, color=cp[i])#, ax = s)
				# pl.hist(kde, bins = np.linspace(0,bins_max[i],nr_bins), color = cp[i], alpha = 0.25, normed = True, histtype='step', cumulative = True, lw = 3.0)
				sn.kdeplot(kde, shade=True, color=cp[i], bw = bws[i], lw = 3.0, clip=(0,bins_max[i]))#, ax = s)
				# s.axvline(0, lw=0.25, alpha=0.5, color = 'k')
				s.set_xlim([0,15])
				s.axhline(0, lw=0.25, alpha=0.5, color = 'k')
				simpleaxis(s);		spine_shift(s)
			pl.tight_layout()
		fn = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['percept','histogram'], extension = '.pdf'))[-1]
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/behavior/figs'), fn))

		# f = pl.figure(figsize = (6,6))
		# nr_percepts = np.min([len(percept_durations[0]), len(percept_durations[1])])
		# with sn.axes_style("white"):
		# 	sn.jointplot(percept_durations[0][:nr_percepts], percept_durations[1][:nr_percepts], kind="hex")
		# fn = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['percept','hex'], extension = '.pdf'))[-1]
		# pl.savefig(os.path.join(self.stageFolder(stage = 'processed/behavior/figs'), fn))
		# shell()	
			# sn.distplot(kde, kde=False, fit=stats.gamma, bins = np.linspace(0,15,100), 
			# 	hist_kws={"histtype": "stepfilled", "color": ("slategray","dodgerblue")[i], "alpha":0.25}, 
			# 	kde_kws={"color": ("slategray","dodgerblue")[i], "lw": 3, "label": ("R","G")[i]});


		# if statement to get the blinks that happend just before transitions, and those that did not..
		# Get all the transition events within 5 seconds after the blink event, with their corresponding durations
		
		trans_due_to_blinks = []
		trans_due_to_ms = []
		percept_due_to_blinks = []
		percept_due_to_ms = []
		
		for blink_onset_event in blink_events:
			for trans_index, trans_onset_event in enumerate(event_data[1]):
				if blink_onset_event-interval <= trans_onset_event < blink_onset_event+interval:
					trans_due_to_blinks.append([blink_onset_event-trans_onset_event,trans_durations[trans_index]])
			for percept_index, percept_onset_event in enumerate(np.concatenate([event_data[0],event_data[2]],axis=0)):
				if blink_onset_event-interval <= percept_onset_event < blink_onset_event+interval:
					percept_due_to_blinks.append([blink_onset_event-percept_onset_event])
					

		for trans_onset_event in event_data[1]:
			for ms_index, ms_event in enumerate(ms_events):
				if trans_onset_event-interval <= ms_event < trans_onset_event+interval:
					trans_due_to_ms.append([ms_event-trans_onset_event,.3])
	
		for percept_index, percept_onset_event in enumerate(np.concatenate([event_data[0],event_data[2]],axis=0)):
			for ms_index, ms_event in enumerate(ms_events):
				if percept_onset_event-interval <= ms_event < percept_onset_event+interval:
					percept_due_to_ms.append([ms_event-percept_onset_event])
		
		# these are not yet KDEs so best not call them that!
		trans_due_to_blinks_joined = np.vstack(trans_due_to_blinks)[:,0]
		trans_due_to_ms_joined = np.vstack(trans_due_to_ms)[:,0]
		percept_due_to_blinks_joined = np.vstack(percept_due_to_blinks)[:,0]
		percept_due_to_ms_joined = np.vstack(percept_due_to_ms)[:,0]
		
		f = pl.figure(figsize = (12,4))
		s = f.add_subplot(111)
		with sn.axes_style("white"):
			
			sn.kdeplot(trans_due_to_blinks_joined, shade=False, color=cp[0], bw = 0.1, lw = 3.0, clip=(-interval-0.5,interval + 0.5), alpha=0.5, ax = s, label = 'blinks vs trans')			
			sn.kdeplot(trans_due_to_ms_joined, shade=True, color=cp[1], bw = 0.1, lw = 3.0, clip=(-interval-0.5,interval + 0.5), alpha=0.5, ax = s, label = 'ms vs trans')
			s.set_xlim([-interval+1,interval-1])
			s.axhline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.legend()
			simpleaxis(s)
			spine_shift(s)
			pl.xlabel('Time after transition [s] # of ms: %i' % len(ms_events))
			pl.ylabel('Proportion of start transion events around a blink')
			pl.tight_layout()
		fn = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['trans_vs_blinks','histogram'], extension = '.pdf'))[-1]
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/behavior/figs'), fn))
		
		f = pl.figure(figsize = (12,4))
		s = f.add_subplot(111)
		with sn.axes_style("white"):
			
			sn.kdeplot(percept_due_to_blinks_joined, shade=False, color=cp[0], bw = 0.1, lw = 3.0, clip=(-interval-0.5,interval + 0.5), alpha=0.5, ax = s, label = 'blinks vs percept')			
			sn.kdeplot(percept_due_to_ms_joined, shade=True, color=cp[1], bw = 0.1, lw = 3.0, clip=(-interval-0.5,interval + 0.5), alpha=0.5, ax = s, label = 'ms vs percept')
			s.set_xlim([-interval+1,interval-1])
			s.axhline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.legend()
			simpleaxis(s)
			spine_shift(s)
			pl.xlabel('Time after transition [s] # of ms: %i' % len(ms_events))
			pl.ylabel('Proportion of start percept events around a blink')
			pl.tight_layout()
		fn = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['percept_vs_blinks','histogram'], extension = '.pdf'))[-1]
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/behavior/figs'), fn))

		# 100 Hz should be enough :)
		time_points = np.linspace(-interval,interval,1000)
		
		# the gaussian_kde estimates its own bandwidth, so we have to see what it comes up with, and make it the same across subjects.
		# having run them all across the different methods, I decide to try numbers out and converge on 0.05
		kde_bandwidth = 0.05
		kde_results = []
		for dt, name in zip([trans_due_to_blinks_joined, trans_due_to_ms_joined, percept_due_to_blinks_joined, percept_due_to_ms_joined],
								['trans_due_to_blinks_joined', 'trans_due_to_ms_joined', 'percept_due_to_blinks_joined', 'percept_due_to_ms_joined']):
			kde = stats.gaussian_kde(dt)
			self.logger.info('automatically defined kde bandwidth factor for %s is %f'%(name, kde.factor))
			kde = stats.gaussian_kde(dt, kde_bandwidth)
			self.logger.info('setting kde bandwidth factor for %s to %f'%(name, kde.factor))
			evaluation = kde.evaluate(time_points)
			kde_results.append(evaluation.T)

		with pd.get_store(self.ho.inputObject) as h5_file:
			h5_file.put("/%s/%s"%('kde_blink_behavior_' + data_type, 'time_points'), pd.Series(time_points))
			h5_file.put("/%s/%s"%('kde_blink_behavior_' + data_type, 'kde_time_courses'), pd.DataFrame(kde_results))		
		
	
	def make_blinks_induced_trans_evfile(self, interval=1):
			
		conds = ['percept_one_button', 'transition_button', 'percept_two_button']
		
		# get event files
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
		
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events')
			run_events = []
			percept_durations = []
			blink_events = []
			ms_events = []
			percept_durations = []
			trans_durations = []
			
			for cond in conds:
				run_events.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,0])	# toss out last and first event of each type to make sure there are no strange spill-over effects
				if cond in ['percept_one_button', 'percept_two_button']:
					percept_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
				elif cond == 'transition_button':
					trans_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
			percept_durations.append(percept_durations)
			
			blink_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_Blinks.txt'))[:,0]
			ms_events = np.loadtxt(os.path.join(event_directory, str(r.ID) + '_MicroSaccades.txt'))[:,0]
			
			trans_due_to_blinks = []
			
			for blink_onset_event in blink_events:
				for trans_index, trans_onset_event in enumerate(run_events[1]):
					if blink_onset_event-interval <= trans_onset_event < blink_onset_event:
						trans_due_to_blinks.append([trans_onset_event,trans_durations[0][trans_index]])
			
			if np.shape(trans_due_to_blinks) >= (1.,1.):
				trans_due_to_blinks = np.vstack(trans_due_to_blinks)
				blink_induced_trans_regressor = np.array(np.vstack([trans_due_to_blinks[:,0], trans_due_to_blinks[:,1], np.ones(len(trans_due_to_blinks))])).T
			else:
				blink_induced_trans_regressor = np.array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]).T
				
			np.savetxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events', str(r.ID) + '_blink_induced_trans.txt'), blink_induced_trans_regressor, fmt = '%4.2f', delimiter='\t')
				

			

	
	def find_instant_trans(self, inst_duration = 0.5):
		
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['BR'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		if tr > 10:
			tr = tr / 1000.0
		run_duration = tr * nr_trs		
		
		conds = ['percept_one_button', 'transition_button', 'percept_two_button']
		
		instant_vs_all_events = []

		nr_runs = 0
		# get event files
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events')
			this_run_events = []
			these_percept_durations = []
			raw_percept_durations = []
			raw_trans_durations = []
			for cond in conds:
				this_run_events.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,0])	# toss out last and first event of each type to make sure there are no strange spill-over effects
				if cond in ['percept_one_button', 'percept_two_button']:
					these_percept_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
				elif cond == 'transition_button':
					raw_trans_durations.append(np.loadtxt(os.path.join(event_directory, str(r.ID) + '_' + str(cond) + '.txt'))[1:-1,1])
			raw_percept_durations.append(these_percept_durations)
			this_run_events = np.array(this_run_events) + nr_runs * run_duration
		
			trans_durations = np.hstack(raw_trans_durations)	
			instant_event = []
			
			percept1_array = np.array([this_run_events[0],these_percept_durations[0],np.ones(float(len(this_run_events[0])))]).T
			percept2_array = np.array([this_run_events[2],these_percept_durations[1],np.ones(float(len(this_run_events[2])))]).T
			trans_array = np.array([this_run_events[1],trans_durations,np.zeros(float(len(this_run_events[1])))]).T
			
			all_array = np.vstack([percept1_array,percept2_array,trans_array])
			indices = np.argsort(all_array[:,0],axis=0)
			ordered_events = all_array[indices,:]

			for index, event in enumerate(ordered_events[:-1,0]):
				if ordered_events[index,2] == ordered_events[index+1,2]:
					instant_event.append(event)
			 			
			instant_regressor = np.array(np.vstack([instant_event, 0.1*np.ones(len(instant_event)), np.ones(len(instant_event))])).T
			np.savetxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events', str(r.ID) + '_instant_trans.txt'), instant_regressor, fmt = '%4.2f', delimiter='\t')
			
			instant_vs_all_events.append([instant_event,this_run_events[1]])
		
		inst = float(len(instant_vs_all_events[0][0]))
		all = float(len(instant_vs_all_events[0][1]))
		percentage_trans =  (inst / all)*100
		
		
		cp = sn.color_palette("Set1", 3)
		f = pl.figure(figsize = (12,4))
		s = f.add_subplot(111)
		with sn.axes_style("white"):
			sn.kdeplot(trans_durations, shade=True, color=cp[1], bw = 0.1, lw = 3.0, clip=(0,5), alpha=0.5, ax = s, label = 'Transitions')
			s.set_xlim([0,5])
			s.axhline(0, lw=0.25, alpha=0.5, color = 'k')
# 			s.axvline(inst_duration, lw=1.0, alpha=0.5, color = 'k')
			pl.legend()
			simpleaxis(s)
			spine_shift(s)
			pl.xlabel('Durations of transition [s] / percentage instantaneous: %i ' % percentage_trans)
			pl.ylabel('Proportion of transion events')
			pl.tight_layout()
		fn = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = ['percentage_intantaneous'], extension = '.pdf'))[-1]
# 		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/behavior/figs'), fn))
		
		
		
	def create_rhifc_mask(self):
		
		roi_list = ['S_circular_insula_sup','Lat_Fis_ant_Horizont','Lat_Fis_ant_Vertical','G_front_inf_Opercular','G_and_S_subcentral','Lat_Fis_post','G_front_inf_Triangul']
		
		mask_path = os.path.join(self.stageFolder('processed/mri/masks/anat'))
		all_rois = []
		rois_combined = np.zeros((17,128,128)) # Is this the size of the largest ROI in this list   ?
		for ii in roi_list:
			# It seems to be the case that a dash interferes with running exec core. I changed the dash into underscores for those 7 ROIs (but this is not a permanent solution).
# 			re.sub('-', '_', os.path.join(mask_path, 'rh' + ii + '.nii.gz'))
			
			
			if os.path.isfile(os.path.join(mask_path,'rh.' + str(ii) + '.nii.gz')):
				exec("rh_%s = np.array(NiftiImage(os.path.join(mask_path,'rh.%s.nii.gz')).data,dtype=bool)"%((ii),(ii)))
				all_rois.append('rh_'+str(ii))
				rois_combined += eval('rh_' + str(ii))
		
		right_ifc = np.zeros(np.shape(rh_S_circular_insula_sup))
		right_ifc[rois_combined!=0] =1 

		new_nifti = NiftiImage(right_ifc)
		new_nifti.header = NiftiImage(os.path.join(mask_path, 'rh.S_circular_insula_sup.nii.gz')).header
		new_nifti.save(os.path.join(mask_path, 'rh.IFC.nii.gz'))
		
	def deconvolve_and_regress_trials_roi(self, roi, threshold = 3.5, mask_type = 'stim_on_mapper_Z_5mm', which_data =  'mri', mask_direction = 'pos', signal_type = 'mean', data_type = 'mcf_phys_sgtf_Z'):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['BR'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		if tr > 10:
			tr = tr / 1000.0
		run_duration = tr * nr_trs
		
		conds = ['percept_one_button','transition_button','percept_two_button']
		stim_labels = ['CCW_RG','CW_RG','CCW_GR','CW_GR']
		
		# check in the pupil data
		if not hasattr(self, 'pupil_data'):
			self.collect_pupil_data_from_hdf(condition = 'BR', event_types = conds, data_type = 'pupil_bp')
		
		h5file = tb.open_file(self.hdf5_mri_filename, mode = 'r')
					
		roi_data = []
		mocos = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			elif type(roi) == str:
				roi_data.append(self.roi_data_from_hdf(h5file, r, roi, data_type, postFix = []))
				roi_name = roi
			else: # roi is a list?
				roi_data.append(np.vstack([self.roi_data_from_hdf(h5file, r, this_roi, data_type, postFix = []) for this_roi in roi]))
				roi_name = '_'.join(roi)
			mocos.append(np.loadtxt(self.runFile(stage = 'processed/mri', run = r, extension = '.par', postFix = ['mcf'])))
			nr_runs += 1

		
		[event_data, bit_events, it_events, blink_events, ms_events, stim_events] = self.grab_events_for_deco()
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		mocos = np.vstack(mocos)
		
		# mapping data
		if 'mapper' in mask_type:
			if type(roi) == str:
				mapping_data = self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type, postFix = [])
			else: # roi is a list?
				mapping_data =  np.vstack([self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['mapper'][0]], this_roi, mask_type, postFix = []) for this_roi in roi])
		else:
			if type(roi) == str:
				mapping_data = self.roi_data_from_hdf(h5file, 'gfeat_stats', roi, mask_type, postFix = [])
			else: # roi is a list?
				mapping_data =  np.vstack([self.roi_data_from_hdf(h5file, 'gfeat_stats', this_roi, mask_type, postFix = []) for this_roi in roi])

		h5file.close()
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].mean(axis = 0)')
		
		time_signals = []
		interval = [-10.0,20.0]
		nuissance_events = [np.array(blink_events) + interval[0], np.array(ms_events) + interval[0], np.array(stim_events) + interval[0], np.array(stim_events) + interval[0] + 150.0]
		# nuisance version?
# 		nuisance_design = Design(timeseries.shape[0] * 2, tr/2.0 )
# 		nuisance_design.configure(nuissance_events)
# 		nuisance_design.configure(np.array(np.hstack([blink_events, ms_events, stim_events])))
		full_nuisance_design = r_[nuisance_design.designMatrix, np.repeat(mocos,2, axis = 0).T].T
		
		deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1], run = False)
		deco.runWithConvolvedNuisanceVectors(full_nuisance_design)
		deco.residuals()
		# mean trans on response:
		bit_it_end_start_events = [np.concatenate([event_data[i] + interval[0] for i in [0,2]]), event_data[1] + interval[0], np.array(bit_events) + interval[0], np.array(it_events) + interval[0]]
		deco1 = ArrayOperator.DeconvolutionOperator( inputObject = np.squeeze(np.array(deco.residuals)).T, 
							eventObject = bit_it_end_start_events, TR = tr/subsampling, deconvolutionSampleDuration = tr/subsampling, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		
		trans_on_resp = deco1.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('trans_on')]
		# mean trans off response:
		trans_off_resp = deco1.deconvolvedTimeCoursesPerEventTypeNuisance[conds.index('trans_off')]
		# instantaneous
		
		if True:
			f = pl.figure(figsize = (6,3))
			s = f.add_subplot(1,1,1)
			s.set_title(roi + ' ' + 'transition on and offset')
			pl.plot(np.linspace(interval[0], interval[1], trans_on_resp.shape[0]), trans_on_resp, 'k', label = 'trans on')
			pl.plot(np.linspace(interval[0], interval[1], trans_off_resp.shape[0]), trans_off_resp, 'r', label = 'trans off')
			s.set_xlabel('time [s]')
			s.set_ylabel('% signal change')
			# s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
			leg = s.legend(fancybox = True)
			leg.get_frame().set_alpha(0.5)
			if leg:
				for t in leg.get_texts():
					t.set_fontsize('small')	# the legend text fontsize
				for l in leg.get_lines():
					l.set_linewidth(3.5)  # the legend line width
			simpleaxis(s)
			spine_shift(s)
			# s.set_ylim([-2,2])
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '_' + mask_direction + '_template_deconvolutions.pdf'))
			
		
		rounded_event_array = np.array([np.array(((ev / 1.5) * 2.0), dtype = int) for ev in [np.concatenate([event_data[i] for i in [0,2]]), event_data[1]]])
		rounded_event_types = np.array([np.ones(ev.shape) * i for i, ev in enumerate([np.concatenate([event_data[i] for i in [0,2]]), event_data[1]])])
		
		nr_trials = np.concatenate(rounded_event_array).shape[0]
		per_trial_design_matrix = np.zeros((nr_trials * 2, timeseries.shape[0] * 2))
		
		for i in range(nr_trials):
			# trans on regressors:
			per_trial_design_matrix[i][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i] = np.correlate(per_trial_design_matrix[i], trans_on_resp, 'same')
			# trans off regressors:
			per_trial_design_matrix[i + nr_trials][np.concatenate(rounded_event_array)[i]] = 1.0
			per_trial_design_matrix[i + nr_trials] = np.correlate(per_trial_design_matrix[i], trans_off_resp, 'same')
		
		full_per_trial_design_matrix = np.mat(np.vstack((per_trial_design_matrix, full_nuisance_design.T))).T
		full_per_trial_betas = ((full_per_trial_design_matrix.T * full_per_trial_design_matrix).I * full_per_trial_design_matrix.T) * np.mat(deco.residuals).T
		full_per_trial_betas_no_nuisance = np.array(full_per_trial_betas[:nr_trials*2].reshape(2,-1).T).squeeze()
		
		trial_info = pd.DataFrame({'trans_on_betas': full_per_trial_betas_no_nuisance[:,0], 'trans_off_betas': full_per_trial_betas_no_nuisance[:,1], 'event_times': np.concatenate(rounded_event_array), 'event_types': np.concatenate(rounded_event_types)})
		
		h5file.close()
		with pd.get_store(self.hdf5_filename) as h5_file: # hdf5_filename is now the reward file as that was opened last
			h5_file.put("/per_trial_glm_results/%s"% roi + '_' + mask_type + '_' + mask_direction + '_' + data_type, trial_info)


	def deconvolve_and_regress_trials(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4'], signal_type = 'mean', data_type = 'mcf_phys_tf_Z'):
		"""docstring for deconvolve_and_regress_trials_roi"""
		for roi in rois:
			self.deconvolve_and_regress_trials_roi(roi, threshold = threshold, mask_type = 'stim_on_mapper_Z_5mm', mask_direction = 'pos', signal_type = signal_type, data_type = data_type)
			self.deconvolve_and_regress_trials_roi(roi, threshold = -threshold, mask_type = 'stim_on_mapper_Z_5mm', mask_direction = 'neg', signal_type = signal_type, data_type = data_type)



	def simulate_and_regress(self, roi = 'V1', threshold = 12.5, mask_type = 'stim_on_mapper_Z_5mm', mask_direction = 'pos', data_type = 'mcf_phys_tf_Z', interval = [-5.0,15.0], subsampling = 2.0):
		
		"""Function to simulate different event related parts of a BR run (with different gains),
		deconvolve, regress those timeseries out and look at beta weights for the different simulated parts of events"""
		
		# Get standard hrf
		def doubleGamma_with_d(x, a1 = 6, a2 = 12, b1 = 0.9, b2 = 0.9, c = 0.35,d1=5.4,d2=10.8):
			return np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])
			
		# Grab events
		
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['BR'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		if tr > 10:
			tr = tr / 1000.0
		run_duration = tr * nr_trs
		
		conds = ['percept_one_button','transition_button','percept_two_button']
		stim_labels = ['CCW_RG','CW_RG','CCW_GR','CW_GR']
		
		h5file = tb.open_file(self.hdf5_mri_filename, mode = 'r')
					
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			if 'residuals' in data_type:
				roi_data[-1] = roi_data[-1] ** 2
			elif type(roi) == str:
				roi_data.append(self.roi_data_from_hdf(h5file, r, roi, data_type, postFix = []))
				roi_name = roi
			else: # roi is a list?
				roi_data.append(np.vstack([self.roi_data_from_hdf(h5file, r, this_roi, data_type, postFix = []) for this_roi in roi]))
				roi_name = '_'.join(roi)
			nr_runs += 1

		[event_data, bit_events, it_events, blink_events, ms_events, stim_events, event_durations] = self.grab_events_for_deco()
		
		# Grab roi data, deconvolve with nuisance regressors and take residuals
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		
		# mapping data
		if 'mapper' in mask_type:
			if type(roi) == str:
				mapping_data = self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type, postFix = [])
			else: # roi is a list?
				mapping_data =  np.vstack([self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['mapper'][0]], this_roi, mask_type, postFix = []) for this_roi in roi])
		else:
			if type(roi) == str:
				mapping_data = self.roi_data_from_hdf(h5file, 'gfeat_stats', roi, mask_type, postFix = [])
			else: # roi is a list?
				mapping_data =  np.vstack([self.roi_data_from_hdf(h5file, 'gfeat_stats', this_roi, mask_type, postFix = []) for this_roi in roi])

		h5file.close()
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].mean(axis = 0)')
		
		events = [np.array(blink_events) + interval[0], np.array(ms_events) + interval[0], np.array(stim_events) + interval[0], np.array(stim_events) + interval[0] + 150.0]
		do1 = ArrayOperator.DeconvolutionOperator( inputObject = timeseries,
							eventObject = events, TR = tr, deconvolutionSampleDuration = tr/subsampling, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		do1.residuals()
		
		sample_duration = tr / do1.ratio
		hrf_times = np.arange(0,25, sample_duration)
		# Model different parts of events during a run, give them different weights
		hrf = doubleGamma_with_d(hrf_times) # , a1 = 4.5, a2 = 10, d1 = 5.0, d2 = 10.0
		hrf = hrf/np.sum(hrf)

		sample_times = np.arange(do1.residuals.shape[1])

		trans_ons = (np.round(event_data[1] / sample_duration)).astype(int)
		trans_ends = (trans_ons + np.round(event_durations[1] / sample_duration)).astype(int)
		
		# periods are their own duration, events modeled as 300 ms 'events'
		when_trans_periods = np.array([(sample_times>=to) * (sample_times<=te) for to, te in zip(trans_ons, trans_ends)]).sum(axis = 0)
		when_trans_ends = np.array([(sample_times==te) for to, te in zip(trans_ons, trans_ends)]).sum(axis = 0)
		when_trans_starts = np.array([(sample_times==to) for to, te in zip(trans_ons, trans_ends)]).sum(axis = 0)

		when_trans_periods_pt = np.array([(sample_times>to) * (sample_times<te) for to, te in zip(trans_ons, trans_ends)])
		when_trans_ends_pt = np.array([(sample_times==te) for to, te in zip(trans_ons, trans_ends)])
		when_trans_starts_pt = np.array([(sample_times==to) for to, te in zip(trans_ons, trans_ends)])

		# convolve these simulated parts with HRF
		signal_periods = np.array([fftconvolve(tp_pt, hrf, 'full')[:sample_times.shape[0]] for tp_pt in when_trans_periods_pt])
		signal_ends = np.array([fftconvolve(te_pt, hrf, 'full')[:sample_times.shape[0]] for te_pt in when_trans_ends_pt])
		signal_starts = np.array([fftconvolve(ts_pt, hrf, 'full')[:sample_times.shape[0]] for ts_pt in when_trans_starts_pt])

		signal_periods -= signal_periods.mean()
		signal_ends -= signal_ends.mean()
		signal_starts -= signal_starts.mean()

		# # gains of different signal parts
		# period_gain = 0.5
		# start_gain = 0.0
		# end_gain = 10.0
		
		model = np.vstack([signal_starts/signal_starts.std(), signal_periods/signal_periods.std(), signal_ends/signal_ends.std()])

		# Regress out, get betas
		lsq = sm.OLS(np.squeeze(do1.residuals).T,model.T)
		results = lsq.fit()
		
		res_reshaped = results.params.reshape((3,results.params.shape[0]/3))
		res_summed = res_reshaped.sum(axis = 0)
		res_norm = res_reshaped / res_summed 


		pl.hist(res_norm[0], range = [0,1], bins = 50, cumulative = True, alpha = 0.7, histtype='step', linewidth = 2.0, color = 'r')
		pl.hist(res_norm[1], range = [0,1], cumulative = True, bins = 50, alpha = 0.7, histtype='step', linewidth = 2.0, color = 'g')
		pl.hist(res_norm[2], range = [0,1], cumulative = True, bins = 50, alpha = 0.7, histtype='step', linewidth = 2.0, color = 'b')

		# plot these timecourses
		pl.figure(figsize = (16,3))
		pl.plot(signal_periods*period_gain, 'r', signal_ends*end_gain, 'b', signal_starts*start_gain, 'g')
		pl.legend(['period', 'ends', 'starts'])
		
		
	def shift_event_get_rsquared(self, roi = 'V1', threshold = 12.5, mask_type = 'stim_on_mapper_Z_5mm', mask_direction = 'pos', data_type = 'mcf_phys_tf_Z', interval = [-5.0,21.0], subsampling = 2.0, event_division = 10):
		
		self.logger.info('starting basic fmri roi deconvolution with data of type %s and mask of type %s, in the interval %s' % (data_type, mask_type, str(interval)))

		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['BR'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		if tr > 10:
			tr = tr / 1000.0
		run_duration = tr * nr_trs
		
		conds = ['percept_one_button','transition_button','percept_two_button']
		stim_labels = ['CCW_RG','CW_RG','CCW_GR','CW_GR']

		# check in the pupil data
		if not hasattr(self, 'pupil_data'):
			self.collect_pupil_data_from_hdf(condition = 'BR', event_types = conds, data_type = 'pupil_bp')
			
		[event_data, bit_events, it_events, blink_events, ms_events, stim_events, event_durations, half_trans_events] = self.grab_events_for_deco()

 		h5file = tb.open_file(self.hdf5_mri_filename, mode = 'r')
		
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['BR']]:
			if type(roi) == str:
				roi_data.append(self.roi_data_from_hdf(h5file, r, roi, data_type, postFix = []))
				roi_name = roi
			else: # roi is a list?
				roi_data.append(np.vstack([self.roi_data_from_hdf(h5file, r, this_roi, data_type, postFix = []) for this_roi in roi]))
				roi_name = '_'.join(roi)
			event_directory = os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'events')
		
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )

		roi_data_per_run = demeaned_roi_data

		roi_data = np.hstack(demeaned_roi_data)
		
		if len(mask_type) == 0:
			mapping_data = np.ones(np.shape(roi_data))
		# In case, map data with mask_type contrast
		elif 'mapper' in mask_type:
			if type(roi) == str:
				mapping_data = self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['mapper'][0]], roi, mask_type, postFix = [])
			else: # roi is a list?
				mapping_data =  np.vstack([self.roi_data_from_hdf(h5file, self.runList[self.conditionDict['mapper'][0]], this_roi, mask_type, postFix = []) for this_roi in roi])
		else:
			if type(roi) == str:
				mapping_data = self.roi_data_from_hdf(h5file, 'gfeat_stats', roi, mask_type, postFix = [])
			else: # roi is a list?
				mapping_data =  np.vstack([self.roi_data_from_hdf(h5file, 'gfeat_stats', this_roi, mask_type, postFix = []) for this_roi in roi])

		h5file.close()

		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		elif mask_direction == 'all':
			mapping_mask = np.ones(mapping_data[:,0].shape, dtype = bool)
		elif mask_direction == 'neg':
			mapping_mask = mapping_data[:,0] < threshold
		
		timeseries = eval('roi_data[mapping_mask,:].mean(axis = 0)')
		
		fig = pl.figure(figsize = (12, 4))
		s = fig.add_subplot(211)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		events = [np.array(blink_events) + interval[0], np.array(ms_events) + interval[0], np.array(stim_events) + interval[0], np.array(stim_events) + interval[0] + 150.0]
		do1 = ArrayOperator.DeconvolutionOperator( inputObject = timeseries,
							eventObject = events, TR = tr, deconvolutionSampleDuration = tr/subsampling, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		do1.residuals()

		
		new_events = np.zeros(((event_division+1),np.shape(event_data[1])[0]))

		for index, ev in enumerate(event_data[1]):
			for d in range(event_division+1):
				if d == 0.:
					new_events[d,index] = (ev + (event_durations[1][index] * d))
				else:
					new_events[d,index] = (ev + (event_durations[1][index] * d/event_division))

		boxcar_events = []
		for index, ev in enumerate(event_data[1]):
			subsampled_boxcar = ev;
			n = 0
			while subsampled_boxcar < (ev + event_durations[1][index]):
				if n == 0.:
					subsampled_boxcar = subsampled_boxcar
				else:
					subsampled_boxcar = subsampled_boxcar + (n*(tr/subsampling))
				boxcar_events.append(subsampled_boxcar)
				n += 1


		rsq = []
		for d in range(event_division+1):
			deco_events = [new_events[d,:] + interval[0]]
			do2 = ArrayOperator.DeconvolutionOperator( inputObject = np.squeeze(np.array(do1.residuals)).T, 
								eventObject = deco_events, TR = tr/subsampling, deconvolutionSampleDuration = tr/subsampling, 
								deconvolutionInterval = interval[1] - interval[0], run = True )
			do2.residuals()

			time_points = np.linspace(interval[0], interval[1], np.squeeze(do1.deconvolvedTimeCoursesPerEventType).shape[1])

			# plotting requires some setup and labels
			event_labels = ['transition ' + str(d) + '/10']
			plot_colors = ['k'] # but get reasonable colors from a nice colormap later

			sn.set(style="ticks")
			f = pl.figure(figsize = (5,6))
			ax = f.add_subplot(111)
			pl.plot(time_points, np.squeeze(do2.deconvolvedTimeCoursesPerEventType), plot_colors[0])
			ax.set_title('%s data transition %s /10'%(roi_name,str(d)))
			pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
			pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
			ax.set_xlim(xmin=interval[0], xmax=interval[1])
			pl.legend(event_labels)
			simpleaxis(ax);		spine_shift(ax)

			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'),  self.subject.initials + '_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type + '_basic_fmri_deconvolution_' + str(d) + '_' + str(event_division)+ '.pdf'))
			
			# Convolve HRF met event en bereken R2
			data_to_explain = np.squeeze(np.array(do1.residuals)).T
			explained_data = np.dot(np.array(do2.designMatrix), np.array(do2.deconvolvedTimeCoursesPerEventType).squeeze())
			rsq.append((1.0 - np.sum(explained_data - data_to_explain**2) / np.sum(data_to_explain**2) ))

			with pd.get_store(self.hdf5_mri_filename) as h5_file:
				# Safe HRFs for plots across subjects later
				h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'residuals_2_' + str(d)), pd.Series(np.squeeze(np.array(do2.residuals, dtype = np.float32)))) # save the residuals to the deconvolution
				h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'dec_time_course_2_' + str(d)), pd.DataFrame(np.squeeze(do2.deconvolvedTimeCoursesPerEventType).T))

		# Do whole transition aka boxcar separate
		rsq_boxcar = []
		boxcar_deco_events = [np.array(boxcar_events) + interval[0]]
		do3 = ArrayOperator.DeconvolutionOperator( inputObject = np.squeeze(np.array(do1.residuals)).T, 
							eventObject = boxcar_deco_events, TR = tr/subsampling, deconvolutionSampleDuration = tr/subsampling, 
							deconvolutionInterval = interval[1] - interval[0], run = True )
		do3.residuals()

		# plotting requires some setup and labels
		event_labels = ['whole transition']
		plot_colors = ['k'] # but get reasonable colors from a nice colormap later

		sn.set(style="ticks")
		f = pl.figure(figsize = (5,6))
		ax = f.add_subplot(111)
		pl.plot(time_points, np.squeeze(do3.deconvolvedTimeCoursesPerEventType), plot_colors[0])
		ax.set_title('%s data transition %s /10'%(roi_name,str(d)))
		pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
		pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
		ax.set_xlim(xmin=interval[0], xmax=interval[1])
		pl.legend(event_labels)
		simpleaxis(ax);		spine_shift(ax)

		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'),  self.subject.initials + '_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type + '_basic_fmri_deconvolution_whole_trans_boxcar.pdf'))
		
		# Convolve HRF met event en bereken R2
		data_to_explain = np.squeeze(np.array(do1.residuals)).T
		explained_data = np.dot(np.array(do3.designMatrix), np.array(do3.deconvolvedTimeCoursesPerEventType).squeeze())
		rsq_boxcar.append((1.0 - np.sum(explained_data - data_to_explain**2) / np.sum(data_to_explain**2) ))

			# now, to save the r2 back to the hdf5 file...
		with pd.get_store(self.hdf5_mri_filename) as h5_file:
			h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'rsq_boxcar'), pd.Series(rsq_boxcar))
			h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'rsq_' + str(event_division)), pd.Series(rsq))

			# Safe HRFs for plots across subjects later
			h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'time_points'), pd.Series(time_points))	
			h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'residuals_3_boxcar'), pd.Series(np.squeeze(np.array(do3.residuals, dtype = np.float32)))) # save the residuals to the deconvolution
			h5_file.put("/%s/%s"%('basic_mri_deconvolution_' + data_type + '_' + roi_name + '_' + mask_direction + '_' + mask_type, 'dec_time_course_3_boxcar'), pd.DataFrame(np.squeeze(do3.deconvolvedTimeCoursesPerEventType).T))


		
		print rsq, rsq_boxcar


	def shift_event_get_rsquared_pupil(self, condition = 'BR', data_type = 'pupil_bp', interval = [-3.0,3.0], analysis_sample_rate = 25, event_division = 10):
		
			event_types = ['percept_one_button','transition_button','percept_two_button']

			self.logger.info('starting basic pupil deconvolution with data of type %s and sample_rate of %i Hz in the interval %s' % (data_type, analysis_sample_rate, str(interval)))
			self.collect_pupil_data_from_hdf(condition = condition, event_types = event_types, data_type = data_type)

			events = [self.blink_times + interval[0]] + [self.microsaccade_times + interval[0]] + [self.stim_times[i] + interval[0] for i in range(len(self.stim_times))] 
			input_signal = np.array(sp.signal.decimate(self.pupil_data, int(self.sample_rate / analysis_sample_rate)), dtype = np.float32)
			dxy_signal = np.array(sp.signal.decimate(self.dxy_data, int(self.sample_rate / analysis_sample_rate)), dtype = np.float32)

			# shell()
			# create regressors for eye position jitter based regression
			nr_sample_times = np.arange(interval[0], interval[1], 1.0/analysis_sample_rate).shape[0]
			added_jitter_regressors = np.zeros((nr_sample_times, dxy_signal.shape[0]))
			for i in range(nr_sample_times):
				added_jitter_regressors[i,(i+1):] = dxy_signal[:-(i+1)]

			do1 = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
								eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, 
								deconvolutionInterval = interval[1] - interval[0], run = False )
			do1.runWithConvolvedNuisanceVectors(added_jitter_regressors.T)
			do1.residuals()

			# doNN = ArrayOperator.DeconvolutionOperator( inputObject = input_signal,
			# 					eventObject = events, TR = 1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, 
			# 					deconvolutionInterval = interval[1] - interval[0], run = True )
			# doNN.residuals()

			# self.logger.info('explained variance (r^sq) %1.4f'%(1.0 -(np.sum(np.array(do1.residuals)**2) / np.sum(input_signal**2))))

			# self.logger.info('eye jitter decreases residual ssqr from %2.4f to %2.4f'%(np.sum(np.array(doNN.residuals)**2), np.sum(np.array(do1.residuals)**2)))


			[event_data, bit_events, it_events, blink_events, ms_events, stim_events, event_durations, half_trans_events] = self.grab_events_for_deco()

			new_events = np.zeros(((event_division+1),np.shape(event_data[1])[0]))

			for index, ev in enumerate(event_data[1]):
				for d in range(event_division+1):
					if d == 0.:
						new_events[d,index] = (ev + (event_durations[1][index] * d))
					else:
						new_events[d,index] = (ev + (event_durations[1][index] * d/event_division))

			rsq = []
			for d in range(event_division+1):
				deco_events = [new_events[d,:] + interval[0]]
				do2 = ArrayOperator.DeconvolutionOperator( inputObject = np.squeeze(np.array(do1.residuals)).T, 
									eventObject = deco_events, TR =  1.0/analysis_sample_rate, deconvolutionSampleDuration = 1.0/analysis_sample_rate, 
									deconvolutionInterval = interval[1] - interval[0], run = True )
				do2.residuals()

				time_points = np.linspace(interval[0], interval[1], np.squeeze(do1.deconvolvedTimeCoursesPerEventTypeNuisance).shape[1])

				# plotting requires some setup and labels
				event_labels = ['transition ' + str(d) + '/10']
				plot_colors = ['k'] # but get reasonable colors from a nice colormap later

				sn.set(style="ticks")
				f = pl.figure(figsize = (5,6))
				ax = f.add_subplot(111)
				pl.plot(time_points, np.squeeze(do2.deconvolvedTimeCoursesPerEventType), plot_colors[0])
				ax.set_title('%s data transition %s /10'%(data_type,str(d)))
				pl.axvline(0, lw=0.25, alpha=0.5, color = 'k')
				pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')
				ax.set_xlim(xmin=interval[0], xmax=interval[1])
				pl.legend(event_labels)
				simpleaxis(ax);		spine_shift(ax)

				pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'),  self.subject.initials + '_' + data_type + '_basic_pupil_deconvolution_' + str(d) + '_' + str(event_division)+ '.pdf'))
				
				# Convolve HRF met event en bereken R2
				data_to_explain = np.squeeze(np.array(do1.residuals)).T
				explained_data = np.dot(np.array(do2.designMatrix), np.array(do2.deconvolvedTimeCoursesPerEventType).squeeze())
				rsq.append((1.0 - np.sum(explained_data - data_to_explain**2) / np.sum(data_to_explain**2) ))

				# now, to save the r2 back to the hdf5 file...
			with pd.get_store(self.ho.inputObject) as h5_file:
				h5_file.put("/%s/%s"%('basic_pupil_deconvolution_' + data_type, 'rsq_' + str(event_division)), pd.Series(rsq))
			
			print rsq





		
		