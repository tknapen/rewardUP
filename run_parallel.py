#!/usr/bin/env python
# encoding: utf-8
"""
run_parallel.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.

To run this script headlessly and hopefully avoiding X server errors in the process, execute this script using 'python run_parallel.py'
"""

import os, sys, datetime
import subprocess, logging
import shutil

import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib as mpl
# mpl.use('pdf')
import matplotlib.pylab as pl

from IPython import embed as shell

this_raw_folder = '/home/raw_data/BR_trans/'
this_project_folder = '/home/shared/BR/'

sys.path.append( os.environ['ANALYSIS_HOME'] )

import Tools
from Tools.Sessions import *
from Tools.Subjects.Subject import *
# from Tools.Run import *
import Tools.Run
from Tools.Projects.Project import *

import RewardUPSession
import RewardUPGroupLevel

from subjects_import import *

#####################################################
# 	select subset of subjects here
#####################################################

subject_indices = range(len(subjects))
# subject_indices = range(3,len(subjects)) # run 3 at a time... - later 3:len(subjects)
# subject_indices = range(0,3)
# subject_indices = [[0]]	# analyse specific subjects
# subject_indices = [[s.initials for s in subjects].index('OC'), [s.initials for s in subjects].index('JB')]
# subject_indices = [[s.initials for s in subjects].index('TK')]
# subject_indices = [s for s in np.arange(len(subjects)) if subjects[s].initials != 'JB']
# subject_indices = [[s.initials for s in subjects].index('JB')]
# subject_indices = [s for s in np.arange(len(subjects)) if subjects[s].initials != 'JVS']
# subject_indices = [s for s in np.arange(len(subjects)) if subjects[s].initials not in ['JVS', 'EK', 'BM']]






#####################################################
#	actual running of analyses in this function
#####################################################

def run_subject(sj, run_array, session_date, project):

	# basic general information
	visual_roi_collection = ['V1','V2','V3','V4','MT','LO1','LO2'] # visual areas (NOTE: OC does not have a V3)
	visual_eccen = ['V1-4-1','V1-4-2','V1-4-3','V1-4-4']
	striatum_roi_collection = ['Striatum_executive', 'Striatum_parietal', 'Striatum_occipital', 'Striatum_rostral-Motor', 'Striatum_caudal-Motor', 'Striatum_limbic', 'Striatum_temporal']
	thalamus_roi_collection = ['Thalamus_Occipital', 'Thalamus_Posterior parietal', 'Thalamus_Pre-frontal', 'Thalamus_Pre-motor', 'Thalamus_Sensory', 'Thalamus_Temporal']
	frontal_roi_collection = ['combined_insula','IFC', 'G_and_S_cingul-Ant', 'S_pericallosal']
	roi_collection = visual_roi_collection + striatum_roi_collection + thalamus_roi_collection + frontal_roi_collection + visual_eccen
	
	session = RewardUPSession.RewardUPSession(ID = 'RewardUP_' + sj.initials, subject = sj, date = session_date, project = project)

	for r in run_array:
		this_run = Tools.Run.Run( **r )
		session.addRun(this_run)

	session.parcelateConditions()
	session.parallelize = True

	# ----------------------------
	# fMRI:                      -
	# ----------------------------
	
	# preprocessing:
	# --------------
	# session.setupFiles(rawBase = '', process_eyelink_file = False)
	# session.import_edf_data()
	# session.edf_prepocessing_report()
	# session.mapper_events()
	# session.collect_pupil_data_from_hdf()
	# session.button_press_analysis()

	# registration now done by hand, do NOT (!) run this again..
	# session.registerSession(prepare_register = True)

	# B0 unwarping?:
	# --------------
	# Not now, not yet...
	# session.B0_unwarping(conditions=['ori_mapper', 'lum_mapper'], wfs=14.256, etl=41.0, acceleration=3.0) # 'UP', 'P', 
	# session.grab_B0_residuals()
	# session.motionCorrectFunctionals(postFix=['B0'], further_args = ' -dof 7 ')
	# session.bet_example_func()
	session.orientation_events_from_pickle()
	# session.reward_events_from_pickle()
	
	# 

	# # create cortex / LC masks:
	# # -------------------------
	# # not all subjects have retinotopy:
	# session.fsaverage_labels_to_masks()
	# session.createMasksFromFreeSurferLabels(labelFolders = [''], annot = False, annotFile = 'aparc.a2009s', cortex=False)
	# session.createMasksFromFreeSurferLabels(labelFolders = ['V1_ecc'], annot = False, annotFile = 'aparc.a2009s', cortex=False)
	# session.createMasksFromFreeSurferLabels(labelFolders = ['retmap_PRF'], annot = False, annotFile = 'aparc.a2009s', cortex=False)
	# session.createMasksFromFreeSurferLabels(labelFolders = ['retmap_PRF', 'V1_ecc'], annot = True, annotFile = 'aparc.a2009s', cortex=False)
	# session.createMasksFromFreeSurferAseg(asegFile = 'aparc.a2009s+aseg', which_regions = ['Brain-Stem']) # 'Putamen', 'Caudate', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens', 'Cerebellum_Cortex', 'Thalamus_Proper', 'Thalamus', 'VentralDC',
	# session.createMasksFromFSLAtlases(threshold = 0.95)
	# session.remove_empty_masks()
	# session.createMasksFromFreeSurferLabels(labelFolders = ['V1_ecc'], annot = False, annotFile = 'aparc.a2009s', cortex=False)
		
	# retroicor:
	# ----------
	# session.retroicorFSL(conditions=['UP', 'P', 'ori_mapper', 'lum_mapper'], postFix=['B0', 'mcf'], threshold=1.85, nr_dummies=8, sample_rate=500, gradient_direction='y', thisFeatFile='/home/shared/Niels_UvA/Retmap_UvA/analysis/feat_retro/retroicor_design.fsf', prepare=True, run=False)
	# session.retroicorFSL(conditions=['UP', 'P', 'ori_mapper', 'lum_mapper'], postFix=['B0', 'mcf'], threshold=1.85, nr_dummies=8, sample_rate=500, gradient_direction='y', thisFeatFile='/home/shared/Niels_UvA/Retmap_UvA/analysis/feat_retro/retroicor_design.fsf', prepare=False, run=True)
	# session.grab_retroicor_residuals(conditions=['UP', 'P', 'ori_mapper', 'lum_mapper'], postFix=['B0', 'mcf'])

	#

	
	# # rescale:
	# # --------
	# session.rescaleFunctionals(operations = ['sgtf'], funcPostFix = ['B0','mcf','phys','add'], mask_file = None)
	# session.rescaleFunctionals(operations = ['percentsignalchange'], funcPostFix = ['B0','mcf','phys','add','sgtf'], mask_file = None)
	# session.rescaleFunctionals(operations = ['sgtf'], funcPostFix = ['mcf', 'phys', 'add'], mask_file = None)
	# session.rescaleFunctionals(operations = ['zscore'], funcPostFix = ['mcf', 'phys', 'add', 'sgtf'], mask_file = None)
	# session.rescaleFunctionals(operations = ['zscore'], funcPostFix = ['mcf', 'sgtf'], mask_file = None)
	# session.rescaleFunctionals(operations = ['zscore'], funcPostFix = ['mcf', 'phys'], mask_file = None)
	# session.rescaleFunctionals(operations = ['zscore'], funcPostFix = ['mcf', 'phys','sgtf'], mask_file = None)

	# session.rescaleFunctionals(operations = ['highpass'], funcPostFix = ['mcf'], mask_file = None)
	# session.rescaleFunctionals(operations = ['zscore'], funcPostFix = ['mcf', 'tf'], mask_file = None)
	# session.rescaleFunctionals(operations = ['highpass'], funcPostFix = ['mcf', 'phys'], mask_file = None)
	# session.rescaleFunctionals(operations = ['zscore'], funcPostFix = ['mcf', 'phys','tf'], mask_file = None)

	# run per-run feats
	# session.feat_analysis_rivalry(postFix = ['mcf', 'phys', 'add'], analysis_type = 'trans_feat_v2', smooth_mm = 5)
	# session.feat_analysis_rivalry(postFix = ['mcf', 'phys', 'add'], analysis_type = 'trans_feat_v2', smooth_mm = 0)

	# run per-run mapper feats
	# session.feat_analysis_mapper(['mcf', 'phys', 'add'], smooth_mm = 0)
	# session.feat_analysis_mapper(['mcf', 'phys', 'add'], smooth_mm = 5)
	
	# run per-run clean up feats
	# session.feat_analysis_rivalry(postFix = ['mcf', 'phys', 'add'], analysis_type = 'trans_feat_v2', feat_name = 'halfway_trans', smooth_mm = 5)
	# session.feat_analysis_rivalry(postFix = ['mcf', 'phys', 'add'], analysis_type = 'trans_clean_stim_blinks_feat', feat_name = 'no_blinks_stim', smooth_mm = 5)


	# apply register feats using registration to T1, etc.
	# for mm in ['5']: #'0'
		# for fn in ['w_blinks']: #''no_blinks_stim'', 
			# session.register_feats(condition = 'mapper', postFix = ['mcf', 'phys', 'add', mm, 'mapper'])
			# session.register_feats(condition = 'BR', postFix = ['mcf', 'phys', 'add', mm, fn])


			# session.gfeat_analysis_rivalry(postFix = ['mcf','phys','add'], analysis_type = 'trans_gfeat_clean', smooth_mm = int(mm), feat_name = fn)
			# session.gfeat_analysis_rivalry(postFix = ['mcf','phys','add'], analysis_type = 'trans_gfeat', smooth_mm = int(mm), feat_name = fn)
			# session.gfeat_analysis_mapper(postFix = ['mcf','phys','add','mapper'], analysis_type = 'mapper_gfeat', smooth_mm = int(mm))


	# combine feats across runs
 	# session.gfeat_analysis_rivalry(postFix = ['mcf','phys','add'], analysis_type = 'trans_gfeat_clean', smooth_mm = 5)
 	# session.gfeat_analysis_rivalry(postFix = ['mcf','phys','add'], analysis_type = 'trans_gfeat_clean', smooth_mm = 0)
	 	
  	# session.gfeat_analysis_mapper(postFix = ['mcf','phys','add'], analysis_type = 'trans_gfeat', smooth_mm = 0)
 	# session.gfeat_analysis_mapper(postFix = ['mcf','phys','add','mapper'], analysis_type = 'mapper_gfeat', smooth_mm = 0)

	# now register back to epi space...
	# session.take_stats_to_session_space(condition = 'BR', gfeat_name = 'gfeat_%imm_w_blinks', smooth_mm = 0, clear_all_stats = False)
	# session.take_stats_to_session_space(condition = 'BR', gfeat_name = 'gfeat_%imm_w_blinks', smooth_mm = 5, clear_all_stats = False) 
	# session.take_stats_to_session_space(condition = 'BR', gfeat_name = 'gfeat_%imm_halfway_trans', smooth_mm = 0)

	
	# session.take_stats_to_session_space(condition = 'BR', gfeat_name = 'gfeat_%imm_no_blinks_stim', smooth_mm = 0,  clear_all_stats = False)
	# session.take_stats_to_session_space(condition = 'BR', gfeat_name = 'gfeat_%imm_no_blinks_stim', smooth_mm = 5,  clear_all_stats = False) 
	# these latter ones only for subjects that have two mapper runs, of course (that is DE, EK and TN)
	# session.take_stats_to_session_space(condition = 'mapper', gfeat_name = 'gfeat_mapper_%imm', smooth_mm = 0, clear_all_stats = False)
	# session.take_stats_to_session_space(condition = 'BR', gfeat_name = 'gfeat_%imm_no_blinks_stim', smooth_mm = 5, clear_all_stats = False)

	# session.stats_to_surface()

	# session.create_fs_mask(hemis = ['rh','lh'], roi_name = 'IFC', roi_list = ["Lat_Fis-ant-Horizont","Lat_Fis-ant-Vertical","G_front_inf-Opercular","Lat_Fis-post","G_front_inf-Triangul"])	
	# session.create_fs_mask(hemis = ['rh','lh'], roi_name = 'combined_insula', roi_list =  ["S_circular_insula_sup","G_Ins_lg_and_S_cent_ins","G_insular_short","S_circular_insula_ant", "S_circular_insula_inf","S_circular_insula_sup"])

	# mask everything to the hdf file, after taking out all the empty masks. 
	# session.remove_empty_masks()
	# session.mask_stats_and_data_to_hdf(roi_collection = roi_collection, postFix = ['mcf','phys','add'], which_smoothing_widths = [0,5])
	# session.mask_stats_and_data_to_hdf(roi_collection = ['V1', 'V2', 'MT', 'Thalamus'], postFix = ['mcf','phys','add'], which_smoothing_widths = [0,5])

 	# session.collect_pupil_data_from_hdf()
	# session.deconvolve_pupil(data_type = 'pupil_bp', interval = [-3.0,3.0], analysis_sample_rate = 20)
	# session.deconvolve_pupil(data_type = 'pupil_bp_dt', interval = [-3.0,3.0], analysis_sample_rate = 20)

	# interval = [-5,15.0]
	# subsampling = 4.0
	# for roi in ['V1-4-1','V1-4-2','V1-4-3','V1-4-4']:
		# for t, d in zip([2.0,-1.6],['pos','neg']): #[4.5,-2.3]
			# for mask_type in ['stim_on_mapper_Z_5mm', 'stim_on_mapper_Z_0mm']: # 
			# for mask_type in ['stim_on_mapper_Z_0mm']: # ,'BR_gfeat_5mm_w_blinks_cope3_zstat','stim_on_mapper_Z_5mm'
				# session.deconvolve_roi(roi=roi, threshold = t, mask_type = mask_type, mask_direction = d, data_type = 'mcf_phys_sgtf_Z', interval = interval, subsampling = subsampling)
	# for roi in ['V1','V2']: # 'Caudate', 'Thalamus', 'combined_insula','IFC'
		# for t, d in zip([2.3,-2.3],['all']): #[2.0,-1.6]
			# for mask_type in ['']:  #'BR_gfeat_5mm_w_blinks_cope3_zstat','stim_on_mapper_Z_5mm','stim_on_mapper_Z_0mm'
				# session.deconvolve_roi(roi=roi, threshold = t, mask_type = mask_type, mask_direction = d, data_type = 'mcf_phys_sgtf_Z', interval = interval, subsampling = subsampling)


	# interval = [-5,15.0]
	# subsampling = 4.0
	# for roi in ['V1','V2','V3']:
		# for mask_type in ['BR_gfeat_0mm_halfway_trans_cope3_zstat']:
			# session.get_halfway_trans_beta_and_deco(roi = roi, threshold = [2.0,-1.5], mask_type = mask_type, data_type = 'mcf_phys_sgtf_Z', interval = interval, subsampling = subsampling)



	# session.grab_events_for_deco()
	# session.whole_brain_deconvolution()
	# session.whole_brain_deconv_combination()
	# session.collect_pupil_data_from_hdf()
	# session.blink_transition_behavior()
	# session.find_instant_trans(inst_duration = 0.5)
	# session.make_blinks_induced_trans_evfile(interval=1)
	# session.deconvolve_and_regress_trials_roi('V1', threshold = 3.5, mask_type = 'stim_on_mapper_Z_5mm', which_data =  'mri', mask_direction = 'pos', signal_type = 'mean', data_type = 'mcf_phys_sgtf_Z')
	# session.simulate_and_regress(roi = 'V1', threshold = 2.5, mask_type = 'BR_gfeat_0mm_w_blinks_cope3_zstat', mask_direction = 'pos', data_type = 'mcf_phys_sgtf_Z', interval = [-5.0,15.0], subsampling = 6.0)
	# session.make_halfway_trans_evfile()
	
	# session.simulate_and_regress(roi = 'V1', threshold = 2.5, mask_type = 'BR_gfeat_0mm_w_blinks_cope3_zstat', mask_direction = 'pos', data_type = 'mcf_phys_tf_Z', interval = [-5.0,15.0], subsampling = 6.0)

	# session.shift_event_get_rsquared_pupil(condition = 'BR', data_type = 'pupil_bp_dt', interval = [-3.0,3.0], analysis_sample_rate = 20, event_division = 10)
	# for roi in ['V1-4-1','V1-4-2','V1-4-3','V1-4-4']:
	# 	session.shift_event_get_rsquared(roi = roi, threshold = 2.5, mask_type = 'stim_on_mapper_Z_0mm', mask_direction = 'all', data_type = 'mcf_phys_sgtf_Z', interval = [-5.0,21.0], subsampling = 6.0, event_division = 10)

	return True

#####################################################
#	parallel running of analyses in this function
#####################################################

def analyze_subjects(sjs_all, run_arrays, session_dates, projects ):
	if len(sjs_all) > 1: 
		job_server = pp.Server(ppservers=())
		jobs = [(sjs_all[i].initials, job_server.submit(run_subject,(sjs_all[i], run_arrays[i], session_dates[i], projects[i]), (), ("Tools","RewardUPSession"))) for i in range(len(sjs_all))]
		for s, job in jobs:
			job()
		job_server.print_stats()
	else:
		run_subject(sjs_all[0], run_arrays[0], session_dates[0], projects[0])


#####################################################
#	group_level_analysis
#####################################################

def group_level_analysis(sjs_all, run_arrays, session_dates, projects):	
	data_folder = this_project_folder
	sessions = []
	for i in range(len(sjs_all)):
		session = RewardUPSession.RewardUPSession(ID = 'RewardUP_' + sjs_all[i].initials, subject = sjs_all[i], date = session_dates[i], project = projects[i])

		for r in run_arrays[i]:
			this_run = Tools.Run.Run( **r )
			session.addRun(this_run)

		sessions.append(session)

	#group level object
	rup_gl = RewardUPGroupLevel.RewardUPGroupLevel(sessions = sessions, data_folder = data_folder)
	
# 	rup_gl.transition_blink_histogram_across_subjects(data_type = 'pupil_bp', hdf_file = 'eye', interval = 2.0)
		
# 	rup_gl.basic_deconv_results_across_subjects(data_type = 'pupil_bp',hdf_file = 'eye', which_data = 'pupil', smooth_width = 10)
# 	rup_gl.basic_deconv_results_across_subjects(data_type = 'pupil_bp_dt', hdf_file = 'eye', which_data = 'pupil', smooth_width = 10)
	# for d in ['pos','neg']: # 'all'
 		# roi_collection = ['V1','V2','V1-4-1','V1-4-2','V1-4-3','V1-4-4'] #'V1-4-4''IFC''V3','Caudate','Pallidum', 'Thalamus','Putamen',,'V3v','V3d'
#  		roi_collection = ['Caudate', 'Thalamus', 'combined_insula','IFC']
 		# for roi in roi_collection:
			# for mask_type in ['BR_gfeat_0mm_w_blinks_cope3_zstat']: # 'BR_gfeat_0mm_w_blinks_cope3_zstat',,'BR_gfeat_5mm_w_blinks_cope3_zstat', 'stim_on_mapper_Z_5mm', 'BR_gfeat_5mm_w_blinks_cope3_zstat', 'stim_on_mapper_Z_0mm', 
				# rup_gl.basic_deconv_results_across_subjects(data_type = 'mcf_phys_sgtf_Z_%s_%s_%s'%(roi,d, mask_type), hdf_file = 'mri', which_data = 'mri', smooth_width = 7, baseline = False)

	# rup_gl.basic_deconv_results_across_subjects_multiple_V1rois(data_type = 'mcf_phys_sgtf_Z_', roi='V1', mask_type = 'stim_on_mapper_Z_0mm', hdf_file = 'mri', which_data = 'mri', smooth_width = 7)
	
	# rup_gl.plot_rsq_across_subjects(data_type = 'mcf_phys_sgtf_Z_', roi='V1', mask_type = 'stim_on_mapper_Z_0mm', mask_direction='all', hdf_file = 'mri', which_data = 'mri', event_division = 10, nr_reps_for_btstr = 1000)
	# rup_gl.plot_rsq_across_subjects_pupil(data_type = 'pupil_bp_dt', hdf_file = 'eye', which_data = 'pupil', event_division = 10, nr_reps_for_btstr = 1000)
	# pl.show()
# 	rup_gl.bar_charts_from_deconvolutions(regions = ['Thalamus', 'combined_insula', 'IFC','Caudate'], contrast = 'BR_gfeat_0mm_w_blinks_cope3_zstat', sign = 'pos', interval = [3,6])
# 	rup_gl.bar_charts_from_deconvolutions(regions = [ 'V1', 'V2', 'MT'], contrast = 'stim_on_mapper_Z_0mm', sign = 'pos', interval = [3,6])
# 	rup_gl.bar_charts_from_deconvolutions(regions = [ 'V1', 'V2', 'MT'], contrast = 'stim_on_mapper_Z_0mm', sign = 'neg', interval = [3,6])
	# rup_gl.bar_charts_from_deconvolutions(regions = [ 'V1','V2'], contrast =  'BR_gfeat_0mm_w_blinks_cope3_zstat', sign = 'pos', interval = [3,6])
# 	rup_gl.bar_charts_from_halfway_deconvolutions(regions = ['V1','V2'], contrast = 'BR_gfeat_0mm_halfway_trans_cope3_zstat', interval = [3,6])

	# rup_gl.basic_deconv_results_across_multiple_events_and_subjects_pupil(data_type = 'pupil_bp_dt',  hdf_file = 'eye', which_data = 'pupil', smooth_width=7)


#####################################################
#	main
#####################################################

def main():
	analyze_subjects([subjects[	s] for s in subject_indices], [run_arrays[s] for s in subject_indices], [session_dates[s] for s in subject_indices], [projects[s] for s in subject_indices])
	# group_level_analysis([subjects[s] for s in subject_indices], [run_arrays[s] for s in subject_indices], [session_dates[s] for s in subject_indices], [projects[s] for s in subject_indices])


if __name__ == '__main__':
	main()




