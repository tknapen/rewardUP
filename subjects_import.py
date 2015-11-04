#!/usr/bin/env python
# encoding: utf-8
"""
analyze_7T_S1.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.

To run this script headlessly and hopefully avoiding X server errors in the process, execute this script using 'xvfb-run python subjects.py'
"""

import os, sys, datetime
import subprocess, logging

import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pylab as pl

from IPython import embed as shell

this_raw_folder = '/home/raw_data/reward/res'
this_project_folder = '/home/shared/reward_fmri/'

sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
from Tools.Subjects.Subject import *
from Tools.Run import *
from Tools.Projects.Project import *

import RewardUPSession

# -----------------
# Comments:       -
# -----------------

subject_intials = ['OC']
subjects = []
run_arrays = []
projects = []
session_dates = []

for which_subject in subject_intials:	
########################################################################################################################################################################################################


	if which_subject == 'OC':
		# subject information
		initials = 'OC'
		firstName = 'Oly'
		standardFSID = 'OC_250711'
		birthdate = datetime.date( 1983, 6, 24 )
		# labelFolderOfPreference = 'visual_areas'
		labelFolderOfPreference = ''
		presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		presentProject = Project( 'RewardUP', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
		sessionID = 'RewardUP_' + presentSubject.initials
	
		sessionDate = datetime.date(2015, 11, 04)
		sj_session = 'OC_041115'
	
		presentSession = RewardUPSession.RewardUPSession(sessionID, sessionDate, presentProject, presentSubject)
	
		try:
			os.mkdir(os.path.join(this_project_folder, 'data', initials))
		except OSError:
			presentSession.logger.debug('output folders already exist')
	
		# ----------------------
		# Decision tasks:      -
		# ----------------------
	
		runBRArray = [
			# B0 session 1:
			{'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_B0_SENSE_5_1s005a001x1.nii.gz' ),},
			{'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_B0_SENSE_5_1s005a001x2.nii.gz' ),},
			# Mapper session 1:
			{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'lum_mapper', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_LUM_MAP_2x2x3_SENSE_2_1s002a001.nii.gz' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, sj_session, 'behavior', 'oc_-1_2015-11-04_13.30.44_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, sj_session, 'hr', 'SCANPHYSLOG20151104132703.log' ), 
				# 'eyeLinkFilePath': os.path.join(this_raw_folder, sj_session, 'eye', 'oc_0_2014-12-10_11.55.40.edf' ), 
				},
			{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'ori_mapper', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_ORI_MAP_2x2x3_SENSE_3_1s003a001.nii.gz' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, sj_session, 'behavior', 'oc_0_2015-11-04_13.37.03_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, sj_session, 'hr', 'SCANPHYSLOG20151104133553.log' ), 
				'eyeLinkFilePath': os.path.join(this_raw_folder, sj_session, 'eye', 'oc_0_2015-11-04_13.37.03.edf' ), 
				},
			# unpredictable association:
			{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'UP', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_REWARD_1_2x2x3_SENSE_9_1s009a001.nii.gz' ),
				'rawBehaviorFile': os.path.join(this_raw_folder, sj_session, 'behavior', 'oc_1_2015-11-04_14.17.21_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, sj_session, 'hr', 'SCANPHYSLOG20151104141608.log' ), 
				'eyeLinkFilePath': os.path.join(this_raw_folder, sj_session, 'eye', 'oc1.edf' ), 
				},
			# unpredictable association:
			{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'UP', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_REWARD_2_2x2x3_SENSE_10_1s010a001.nii.gz' ),
				'rawBehaviorFile': os.path.join(this_raw_folder, sj_session, 'behavior', 'oc_1_2015-11-04_14.29.20_outputDict.pickle' ),
				'physiologyFile': os.path.join(this_raw_folder, sj_session, 'hr', 'SCANPHYSLOG20151104142815.log' ),
				'eyeLinkFilePath': os.path.join(this_raw_folder, sj_session, 'eye', 'oc2.edf' ),
				},
			# predictable association:
			{'ID' : 7, 'scanType': 'epi_bold', 'condition': 'P', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_REWARD_3_2x2x3_SENSE_11_1s011a001.nii.gz' ),
				'rawBehaviorFile': os.path.join(this_raw_folder, sj_session, 'behavior', 'oc_3_2015-11-04_14.39.42_outputDict.pickle' ),
				'physiologyFile': os.path.join(this_raw_folder, sj_session, 'hr', 'SCANPHYSLOG20151104143829.log' ),
				'eyeLinkFilePath': os.path.join(this_raw_folder, sj_session, 'eye', 'oc3.edf' ),
				},
			# unpredictable association:				
			{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'P', 'session' : 1,
				'rawDataFilePath': os.path.join(this_raw_folder, sj_session, 'mri', 'Olympia_Colizoli_WIP_REWARD_4_2x2x3_SENSE_12_1s012a001.nii.gz' ),
				'rawBehaviorFile': os.path.join(this_raw_folder, sj_session, 'behavior', 'oc_3_2015-11-04_14.50.27_outputDict.pickle' ),
				'physiologyFile': os.path.join(this_raw_folder, sj_session, 'hr', 'SCANPHYSLOG20151104144911.log' ),
				'eyeLinkFilePath': os.path.join(this_raw_folder, sj_session, 'eye', 'oc4.edf' ),
				},
			]


		# ----------------------
		# Initialise session   -
		# ----------------------
		
	subjects.append(presentSubject)
	run_arrays.append(runBRArray)
	projects.append(presentProject)
	session_dates.append(sessionDate)






