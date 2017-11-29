#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySFC - parameters class used to set all PySFC parameters.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
import re

class Params:
    def __init__(self):

        ###############
#%%     General flow
        ###############

        self.load_corpus = True
        self.load_processed_corpus = True
        self.do_all_phrases = True
        self.good_files_only = True  # use only files specified
        self.remove_folders = False

        ###############
#%%     SFC params
        ###############

        self.database = 'french'  # french, chinese

        self.vowel_marks=[.1,.5,.9]  # original SFC
#        self.vowel_marks=[.1,.3,.5,.7,.9]  # in 5pts
        self.vowel_pts = len(self.vowel_marks)

        #% contour generators params
        self.f0_scale = .05  # originally .05 for 3 vowel marks
        self.dur_scale = 10  # 10/.05 = 200 (originally 10)
        self.learn_rate = 0.01  # learning rate - default for adam is 0.001
        self.max_iter = 500  # number of iterations at each stepping of the contour generators
        self.iterations = 20  # to run analysis by synthesis
        self.l2 = .1  # 1e-4 default
        self.hidden_units = 17  # 15 in the paper, but snns config says 17

        #################
#%%     database params
        #################
        self.home = '/home/bgerazov/'
        self.file_type='TextGrid' # from what to build the corpus

        if self.database == 'french':
            self.corpus_name = self.database+'_v6_{}_{}pts'.format(self.file_type, self.vowel_pts)

            self.processed_corpus_name = self.corpus_name+'_learn{}_maxit{}_it{}_{}n_l2{}'.format(
                    self.learn_rate, self.max_iter, self.iterations, self.hidden_units, self.l2)

            if self.do_all_phrases:
                # these are the phrase types we have
                self.phrase_types = 'DC QS EX SC EV DI'.split()
                #    DC - declaration
                #    QS - question
                #    EX - exclamation
                #    SC - suspicious irony
                #    EV - obviousness
                #    DI - incredulous question
            else:
                self.phrase_types = ['DC']  # just do it for DC

            ## and the function types for each
            self.function_types = 'DD DG XX DV EM ID IT'.split()
            #    DD - word on the right depends on the left
            #    DG - word on the left depends on the right
            #    XX - clitic on the left (les enfants) - downstepping for function words
            #    DV - like XX - downstepping for auxiliaries
            #    ID - independancy (separated by a , )
            #    IT - interdependancy - this thing links 3 segments so it's not implemented ...

            self.end_marks = 'XX DV EM'.split()  # contours with only left context

            ## good files used to train the SFC
            self.good_files = {'DC' : np.r_[np.arange(1,90), np.arange(91,220),
                                       np.arange(221,313), np.arange(314,394)],
                          'SC' : np.r_[np.arange(1,219), np.arange(220,313),
                                       np.arange(314,319), np.arange(324,329)],
                          'EV' : np.r_[np.arange(1,90), np.arange(91,313),
                                       np.arange(314,319), np.arange(324,329)],
                          'DI' : np.r_[np.arange(1,26), np.arange(27,90),
                                       np.arange(91,139), np.arange(140,220),
                                       np.arange(221,245), np.arange(246,277),
                                       np.arange(278,313), np.arange(314,319),
                                       np.arange(324,329)],
                          'QS' : np.r_[np.arange(1,219), np.arange(220, 313),
                                       np.arange(314,319), np.arange(324,329)],
                          'EX' : np.r_[np.arange(1,313), np.arange(314,319),
                                       np.arange(324,329)]}

            if self.file_type=='fpro':
                self.datafolder = self.home + 'work/data/french/_fpro/'
                self.re_folder = re.compile(r'^.*\d\.fpro$')
            elif self.file_type=='TextGrid':
                self.datafolder = self.home + 'work/data/french/_grid/'
                self.re_folder = re.compile(r'^.*\d\.TextGrid$')

        ### chinese
        elif self.database == 'chinese':

            self.corpus_name = self.database+'_v2_{}_{}pts'.format(self.file_type, self.vowel_pts)

            self.processed_corpus_name = self.corpus_name+'_learn{}_maxit{}_it{}_{}n_l2{}'.format(
                    self.learn_rate, self.max_iter, self.iterations, self.hidden_units, self.l2)

            if self.do_all_phrases:
                self.phrase_types = 'DC QS'.split()
            else:
                self.phrase_types = ['DC']  # just do it for DC

            ## and the function types
            self.function_types = 'C0 C1 C2 C3 C4 WB ID IT'.split()
            #    C0-4 - tonal accents
            #    WB - word boundary

            self.end_marks = ['WB']  # contours with only left context

            ## good files used to train the SFC
            ## without the multiple DCs
            self.good_files = np.r_[np.arange(1,101), np.arange(1001,1005),
                               4211,4212,4214, 5949]

            if self.file_type=='fpro':
                self.datafolder = self.home + 'work/data/chinese/_fpro/'
                self.re_folder = re.compile(r'^chinese_\d*\.fpro$')

            elif self.file_type=='TextGrid':
                self.datafolder = self.home + 'work/data/chinese/_grid/'
                self.re_folder = re.compile(r'^chinese_\d*\.TextGrid$')

        ##################
#%%     read data params
        ##################

        self.re_fpro = re.compile(  # fpro first line regex
        r"""^\s*
        F0_Ref=\s*(\d+).*
        CLOCK=\s*(\d*).*
        R=([01]\.\d*).*
        DISOL=\s*([\d.]*).*
        Stats=\s*([./\w]*)
        \s*$""", re.VERBOSE)

        ### read textgrids
        self.disol = 0
        self.isochrony_gravity = 0.2
        self.f0_method = 'pitch_marks'

        if self.database == 'french':
            self.use_ipcgs = True
            self.re_vowels = re.compile('[aeouiyx]')

            self.isochrony_clock = .190
            self.f0_ref = 126
            self.f0_min = 80
            self.f0_max = 300

            self.f0_ref_method='all'
            #           Method to use to accumulate stats if f0_stats is None. Can be:
            #            all - all files
            #            DC - DC files (works for french)
            #            vowels - just vowel segments
            #            vowel_marks - at the vowel marks in the vowel segments

            if self.f0_method == 'pitch_marks':
                self.re_f0 = re.compile(r'^.*.PointProcess$')
            elif self.f0_method == 'pca':
                self.re_f0 = re.compile(r'^.*.pca$')

            self.textgrid_folder = self.home + 'work/data/french/_grid/'
            self.f0_folder = self.home + 'work/data/french/_pca/'

            # levels in the textgrid - phones should always be 0
            self.syll_level = 1
            self.orthographs_level = 3
            self.phrase_level = 4
            self.tone_levels = None

            # stats
            self.f0_stats = 'french_f0_stats_all'
            self.dur_stats = 'french_phone_dur_stats'
            self.syll_stats = 'french_syll_dur_stats'

        elif self.database == 'chinese':
            self.use_ipcgs = False  # use syllables
            self.re_vowels = re.compile('.*[aeouiv].*')
            # v in yu

            self.isochrony_clock = .214

            self.f0_ref = 270
            self.f0_min = 100
            self.f0_max = 450

            self.textgrid_folder = self.home + 'work/data/chinese/_grid/'
            self.f0_folder = self.home + 'work/data/chinese/_pca/'

            if self.f0_method == 'pitch_marks':
                self.re_f0 = re.compile(r'^chinese_\d*\.PointProcess$')
            elif self.f0_method == 'pca':
                self.re_f0 = re.compile(r'^chinese_\d*\.pca$')

            # levels in the textgrid - phones should always be 0
            self.syll_level = 1
            self.tone_levels = [2, 3]
            self.orthographs_level = 4
            self.phrase_level = 7

            # stats
            self.f0_stats = 'chinese_f0_stats_all'
            self.f0_ref_method='all'
            self.dur_stats = 'chinese_phone_dur_stats'
            self.syll_stats = 'chinese_syll_dur_stats'

        ###############
#%%     corpus params
        ###############

        self.columns = 'file phrasetype contourtype unit n_unit ramp1 ramp2 ramp3 ramp4'.split()
        self.columns_in = len(self.columns)
        self.columns += ['f0{}'.format(x) for x in range(len(self.vowel_marks))] + ['dur']
        self.orig_columns = self.columns[self.columns_in:]
        self.target_columns = ['target_'+column for column in self.orig_columns]

        #######################
#%%     plotting and saving
        #######################
        self.show_plot = False  # used in all plotting - whether to close the plots immediately

        self.plot_contours = True

        # expansion plots
        self.left_max = 5
        self.right_max = 5
        self.phrase_max = 10

        # copy worst files
        self.plot_worst = False
        self.n_files=100

        # figure save path
        self.save_path = 'figures/{}'.format(self.processed_corpus_name)

        # pkl save path
        self.pkl_path = 'pkls/'