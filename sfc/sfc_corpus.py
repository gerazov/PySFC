# -*- coding: utf-8 -*-
"""
PySFC - corpus related utility functions.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
import pandas as pd
import logging
from natsort import natsorted
import os
import re
import sys
from sfc import sfc_data

#%%
def build_corpus(params):
    '''
    Build corpus from all files in datafolder and save it to corpus_name. All parameters
    passed through object params.

    Parameters
    ----------
    datafolder : str
        Folder where the input data files are located.
    file_type : str
        Whether to use fpros or TextGrids to read the data.

    phrase_types : list
        Attitude functions found in database.

    function_types : list
        Functions other than attitudes found in database.

    end_marks : list
        Functions that end the scope, i.e. they have only left context.

    database : str
        Which database are we using - sets a bunch of predifined parameters.

    vowel_marks : list
        Points where to sample vocalic nuclei.

    use_ipcgs : bool
        Use IPCGs or regular syllables as rhtythmic units. Morlec uses IPCGs,
        chinese uses syllables.

    show_plot : bool
        Whether to keep the plots open of the f0_ref, dur_stat and syll_stat if it is
        necessary to extract them when file_type is TextGrid.

    save_path : str
        Save path for the plotted figures.
    '''
    log = logging.getLogger('build_corpus')

#%% load variables from params
    datafolder = params.datafolder
    phrase_types = params.phrase_types
    function_types = params.function_types
    end_marks = params.end_marks
    database = params.database
    file_type = params.file_type
    columns = params.columns

    re_folder = params.re_folder
    re_vowels = params.re_vowels
    use_ipcgs = params.use_ipcgs
    f0_ref = params.f0_ref
    isochrony_clock = params.isochrony_clock

#%% read filenames
    filenames = natsorted([f for f in os.listdir(datafolder) if re_folder.match(f)])
#%% build a pandas corpus
#    filenames = ['DC_140.TextGrid']  # errors FF - FF
#    filenames = ['EX_265.TextGrid']  # errors FF - FF
#    filenames = ['chinese_000002.TextGrid']  #
#    filenames = ['chinese_001004.TextGrid']  # QS
#    filenames = ['chinese_005949.TextGrid']  # no tones?

    corpus = pd.DataFrame(columns=columns)
    re_fpro = params.re_fpro

    #%% main loop
    phone_set = []
    utterances = {}
    phone_duration_means=None
    for file_cnt, barename in enumerate(filenames):
        log.info('Reading file {}'.format(barename))
        filename = datafolder + barename
        bare = barename.split('.')[0]
        if file_type == 'fpro':
            try:
                fpro_lists = sfc_data.read_fpro(filename, re_fpro, database)  # read fpro
            except:
                log.error(sys.exc_info()[0])
                log.error('{} read error!'.format(barename))
                continue
            fpro_stats, f0s, enrs, units, durunits, durcoeffs, tpss, dursegs, phones, sylls, poss, \
                orthographs, phrase, levels = fpro_lists  # detuple  - TODO make pretier!
            levels = levels.T

        elif file_type == 'TextGrid':  # do it from a TextGrid
            try:
                fpro_lists, phone_duration_means, f0_ref, isochrony_clock = sfc_data.read_textgrid(barename, params,
                                                phone_duration_means=phone_duration_means, 
                                                f0_ref=f0_ref, 
                                                isochrony_clock=isochrony_clock)
            except:
                log.error(sys.exc_info()[0])
                log.error('{} read error!'.format(barename))
                continue

            fpro_stats, f0s, units, durunits, durcoeffs, dursegs, \
                    phones, orthographs, phrase, levels = fpro_lists

        phone_set = phone_set + phones.tolist()
        # get the utterance text
        utterance = orthographs[orthographs != '*'].tolist()
        utterance = [' '+s if s not in ['.',',','?','!'] else s for s in utterance]
        utterance = ''.join(utterance)[1:]
        utterances[bare]= utterance
        if fpro_stats is not None:  # might have a bad header
            f0_ref, isochrony_clock, r_clock, disol, stats = fpro_stats
        #%%  get the rhythmic units (rus) and order them
        # we can't use np.unique for this
        rus = []
        ru_f0s = []
        ru_coefs = []
        ru_map = np.zeros(phones[1:-1].size, dtype=int)
        cnt = -1  # count of unit, skip first silence
        unit = ''
        for i, phone in enumerate(phones[1:-1]):
            # run through the phones and detect vowels to accumulate the ru-s
            if len(unit) == 0:  # at start of loop and after deleting unit
                cnt += 1
                unit = units[1:-1][i]
                rus.append(unit)
                unit = unit[len(phone):]  # delete phone from unit
            else:
                assert unit.startswith(phone), \
                    log.error('{} - unit doesn''t start with phone!'.format(barename))
                unit = unit[len(phone):]  # delete phone from unit
            if re_vowels.match(phone):
                ru_f0s.append(f0s[1:-1,:][i,:])
                ru_coefs.append(durcoeffs[1:-1][i])
            ru_map[i] = cnt

        ru_f0s = np.array(ru_f0s)
        # if there are nans change them to 0s, so the learning doesn't crash:
        if np.any(np.isnan(ru_f0s)):
            log.warning('No pitch marks for unit {} in {}!'.format(
                              rus[np.where(np.any(np.isnan(ru_f0s), axis=1))[0][0]], barename))
            ru_f0s[np.isnan(ru_f0s)] = 0

        ru_coefs = np.array(ru_coefs)
        rus = np.array(rus)
        assert ru_coefs.size == rus.size, \
            log.error('{} - error ru_coefs.size != rus.size'.format(barename))

        #%% construct phrase contours
        if 'FF' not in phrase[1]:
            log.error('{} Start of phrase is not in first segment - skipping!'.format(barename))
            continue
        # this covers phr:FF too
        if phrase[-1] == '*':
            log.error('{} End of phrase is not in finall segment - skipping!'.format(barename))
            continue

        # target vector for
        phrase_type = phrase[-1].strip(':')
        phrase_targets = np.c_[ru_f0s, ru_coefs]
        # generate input ramps for phrase
        ramp_global = np.arange(rus.size)[::-1]
        ramp_local = np.c_[ramp_global,
                           np.linspace(10, 0, num=rus.size),
                           ramp_global[::-1]]
        phrase_ramps = np.c_[ramp_local, ramp_global].T


        #mask = np.arange(rus.size)  # the scope of the contour
        for unit, l in enumerate(np.c_[phrase_ramps.T, phrase_targets]):
            row = pd.Series([barename, phrase_type, phrase_type, rus[unit], unit] + l.tolist(),
                          columns)
            corpus = corpus.append([row], ignore_index=True)

        #%% do the rest of the contours
        landmarks = phrase_types + function_types
        if levels.size > 0:
            for level in levels[:,1:]:  # without the initial silence
                #%% debug
#                level = levels[0,1:]
                mask = []
                iscontour = False
                found_landmark = False
                for cnt, entry in enumerate(level.tolist()):
                    #%% debug
#                    cnt = 4
#                    entry = level[cnt]
                    entry = entry.strip(':')
                    entry = entry[:2]
                    if not iscontour:
                        if 'FF' in entry:  # contour start
                            # if it's the first IPCG or it's one with a vowel we add the mask
                            # this is to solve IPCG-Syll overlap issues:
                            # wordlevel: game DG kupe, but IPCG level: gam (e DG k) up e
                            # i.e. where is ek? it is in the left unit but not the right!
                            # this is because of the mapping function we use
                            # between segments and units (IPCGs)
                            if use_ipcgs:
                                if cnt == 0 or re_vowels.match(phones[cnt+1]):  #+1 because we skip silence
                                    mask.append(cnt)
                            else:  # use sylls - easier
                                mask.append(cnt)

                            iscontour = True

                        if entry in landmarks:
                            log.warning('{} landmark {} starts contour - skipping!'.format(barename, entry))

                    elif iscontour:
                        if entry == '*':  # normal *
                            mask.append(cnt)

                        elif (entry in end_marks and \
                                not found_landmark) or \
                             (entry in landmarks and \
                                not found_landmark and \
                                    entry not in end_marks and \
                                    cnt == len(level.tolist())-1): # end of contour,
                                 # Cs can also end contours if at the end:
                                 # :C4 :FF * :C2
                            contour_type = entry

                            if use_ipcgs:
                                if cnt < level.size - 1 and re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its the end silence
                                    # and if it is not a vowel (from the word that follows)
                                    mask.append(cnt)
                                    landmark_ind = cnt
                                else:
                                    landmark_ind = cnt - 1
                            else:
                                landmark_ind = cnt - 1

                            corpus = append_contour_to_corpus(corpus, columns, barename, phrase_type,
                                                                  contour_type, rus, mask, landmark_ind,
                                                                  ru_map, ru_f0s, ru_coefs)
                            if use_ipcgs:
                                if re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its not a vowel
                                    mask = [cnt]
                                else:
                                    mask = []

                            else:  # when using syllables it's part of the next scope
                                mask = [cnt]

                            found_landmark = False  # you cannot daisy chain XX

                        elif entry in landmarks and \
                                not found_landmark and \
                                    entry not in end_marks:   # if first landmark
                            found_landmark = True
                            contour_type = entry

                            if use_ipcgs:
                                if cnt < level.size - 1 and re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its the end silence
                                    # and if it is not a vowel (from the word that follows)
                                    landmark_ind = cnt
                                else:
                                    landmark_ind = cnt - 1
                            else:
                                landmark_ind = cnt - 1

                            mask.append(cnt)


                        elif 'FF' in entry:  # end of contour and maybe start of next one
                            if use_ipcgs:
                                if cnt < level.size - 1 and re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its the end silence
                                    # and if it is not a vowel (from the word that follows)
                                    mask.append(cnt)
#                            else:  # it's not part of the scope

                            if not found_landmark:
                                log.warning('{} no landmark in level {} - skipping!'.format(
                                        barename, level[cnt-3:cnt+4]))
#                                print(cnt, entry, level[cnt-3:cnt+3])
                                # this will also find the case where FF is not the start of the next contour
                                # e.g. FF * * * DD * * FF * * FF * * * XX * FF
                            else:
                                corpus = append_contour_to_corpus(corpus, columns, barename, phrase_type,
                                                                  contour_type, rus, mask, landmark_ind,
                                                                  ru_map, ru_f0s, ru_coefs)

                            # it might be the start of the next one like in DC_368.fpro!
                            # e.g. FF * * * DD * * * FF * * * XX * FF
                            found_landmark = False

                            if use_ipcgs:
                                if re_vowels.match(phones[cnt+1]):
                                    # the segment with FF is not part of scope if its not a vowel
                                    mask = [cnt]
                                else:
                                    mask = []

                            else:  # when using syllables it's part of the next scope
                                mask = [cnt]

                        elif entry in landmarks and found_landmark:  # end of contour
                            # if not first landmark - ie it's the start of another contour
                            # process previous contour
                            if use_ipcgs:
                                if re_vowels.match(phones[cnt+1]):  #+1 because we skip silence
                                    # also if its a vowel its from the next scope
                                    mask.append(cnt)  # well apparently it is, except the final silence
#                            else: # don't append if it's syllables

                            corpus = append_contour_to_corpus(corpus, columns, barename, phrase_type,
                                                              contour_type, rus, mask, landmark_ind,
                                                              ru_map, ru_f0s, ru_coefs)

                            # update mask to new contour
                            # keep the mask starting from previous landmark
                            # keep landmark if its on vowel
                            if use_ipcgs:
                                if re_vowels.match(phones[landmark_ind+1]):  #+1 because we skip silence
                                    mask = [i for i in mask if i >= landmark_ind]
                                else:
                                    mask = [i for i in mask if i > landmark_ind]
                            else:  # for sylls keep after the landmark
                                mask = [i for i in mask if i > landmark_ind]

                            contour_type = entry  # new contour's type
                            if use_ipcgs:
                                landmark_ind = cnt
                            else:
                                landmark_ind = cnt - 1

                            mask.append(cnt)

                        else:
                            log.warning('{} unknown landmark {} in {} - skipping!'.format(barename, entry, level))
                            break
#%% end test
    phone_set = np.array(phone_set)
    phone_set, phone_cnts = np.unique(phone_set, return_counts=True)

    return fpro_stats, corpus, utterances, phone_set, phone_cnts

def create_masks(corpus, phrase_type, contour_keys, params):
    '''
    Create masks to address the data in the corpus.

    Parameters
    ==========
    corpus : pandas data frame
        Holds all data from corpus.
    phrase_type : str
        Type of phrase type to make the mask.
    contour_keys : list
        Types of functions used for contour generators.

    params
    ======
    good_files_only : bool
        Whether to use only a subset of the files.
    good_files : list
        The subset of good files.
    database : str
        Name of database.
    '''
    #%%
    good_files_only = params.good_files_only
    good_files = params.good_files
    database = params.database
    # init masks
    mask_phrase = corpus['phrasetype'] == phrase_type
#    files = natsorted(np.unique(corpus[mask_phrase]['file'].values))
    files = natsorted(corpus[mask_phrase]['file'].unique())
    mask_file_dict = {}
    n_units_dict = {}
    mask_unit_dict = {}
    re_file_nr = re.compile('.*_(\d*).*')
    mask_contours = {}
    for contour_type in contour_keys:
        mask_contours[contour_type] = corpus['contourtype'] == contour_type
    # make a mask for all files for training the contours
    if good_files_only:
        mask_all_files = mask_phrase & False # this one we have to edit to account for the not good files
    else:
        mask_all_files = mask_phrase

    for file in files:
        file_nr =  re_file_nr.match(file).groups()[0]
        if good_files_only:
#            print(file_nr)
            if database == 'morlec':
                good_nrs = good_files[phrase_type]
            else:
                good_nrs = good_files

            if int(file_nr) in good_nrs:
                mask_file = corpus['file'] == file
                mask_file_dict[file] = mask_file
                mask_all_files = mask_all_files | mask_file
                n_units = np.max(corpus.loc[mask_file, 'n_unit'].values)
                n_units_dict[file] = n_units
                for n_unit in range(n_units+1):
                    mask_unit = corpus['n_unit'] == n_unit
                    mask_unit_dict[n_unit] = mask_unit

        else:  # all files
            mask_file = corpus['file'] == file
            mask_file_dict[file] = mask_file
            n_units = np.max(corpus.loc[mask_file, 'n_unit'].values)
            n_units_dict[file] = n_units
            for n_unit in range(n_units+1):
                mask_unit = corpus['n_unit'] == n_unit
                mask_unit_dict[n_unit] = mask_unit

    if good_files_only:
        files = list(mask_file_dict.keys())
#%%
    return files, mask_all_files, mask_file_dict, \
        mask_contours, n_units_dict, mask_unit_dict


def downcast_corpus(corpus, columns):
    '''
    Take care of dtypes and downcast.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    columns : list
        The columns of the DataFrame.
    '''
    log = logging.getLogger('down_corpus')
    log.info('Converting columns to numeric ...')  # TODO why some n_units are strings??
    start_colum = columns.index('f00')
    corpus[columns[start_colum:]] = corpus[columns[start_colum:]].apply(pd.to_numeric, downcast='float')
#    for column in ['n_unit','ramp1','ramp3','ramp4']:
#        corpus[column] = pd.to_numeric(corpus[column], downcast='unsigned')
    corpus[['n_unit','ramp1','ramp3','ramp4']] = \
            corpus[['n_unit','ramp1','ramp3','ramp4']].apply(pd.to_numeric,
                                                             downcast='unsigned')
#    for column in ['ramp2']:
#        corpus[column] = pd.to_numeric(corpus[column], downcast='float')
    corpus['ramp2'] = corpus['ramp2'].apply(pd.to_numeric, downcast='float')

    return corpus

def expand_corpus(corpus, params):
    '''
    Take care of scale and expand corpus.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all input features and all predictions by the contour generators.

    params
    ======
    columns : list
        Columns of corpus DataFrame.
    orig_columns : list
        Columns holding original f0 and dur_coeff in corpus DataFrame.
    target_columns : list
        Columns holding the f0 and dur_coeff tagets used to train the contour
        generators.
    iterations : int
        Number of iterations to run analysis-by-synthesis loop.
    f0_scale : float
        Scaling factor for the f0s to downscale them near to the dur_coeffs.
    dur_scale : float
        Scaling factor for the dur_coeffs to upscale them near to the f0s.
    '''
    log = logging.getLogger('exp_corpus')

    orig_columns = params.orig_columns
    target_columns = params.target_columns
    iterations = params.iterations
    f0_scale = params.f0_scale
    dur_scale = params.dur_scale

    log.info('Applying scaling to columns ...')
    corpus.loc[:,'dur'] = corpus.loc[:,'dur'] * dur_scale
#    corpus.loc[:,'f01':'f03'] = corpus.loc[:,'f01':'f03'] * f0_scale
    corpus.loc[:,orig_columns[0]:orig_columns[-2]] = corpus.loc[:,orig_columns[0]:
                                                               orig_columns[-2]] * f0_scale

    #% expand corpus with extra columns
    log.info('Expanding initial columns ...')
#    for column in target_columns:
#        corpus.loc[:, column] = np.NaN  # doesn't work on a list of labels
    new_columns = target_columns.copy()
    for i in range(iterations):
        pred_columns = [column + '_it{:03}'.format(i) for column in orig_columns]
        new_columns += pred_columns
#        for column in pred_columns:  # doesn't work on a list of labels
#            corpus.loc[:, column] = np.NaN

    # onliner:
    corpus = pd.concat([corpus,
                        pd.DataFrame(np.nan, index=corpus.index,
                                     columns=new_columns)],
                       axis=1)

    return corpus

#%%
def append_contour_to_corpus(corpus, columns, barename, phrase_type,
                             contour_type, rus, mask, landmark_ind,
                             ru_map, ru_f0s, ru_coefs):  #, debug=False):
    """
    Process contour and append to corpus.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    columns : list
        The columns of the DataFrame.
    barename : str
        Name of file.
    phrase_type : str
        Type of phrase component in file.
    contour_type : str
        Type of contour to add.
    rus : list
        List of Rhythmic units, e.g. IPCGs or syllables, in contour.
    mask : list
        Mask of unit number in the utterance for each unit in the contour.
    landmark_ind : int
        Position of the function type designator.
    ru_map : list
        Shows where each phone in utterance belongs to in rus.
    ru_f0s : ndarray
        f0s for each of the units in rus.
    ru_coefs : ndarray
        dur_coeffs for each unit in rus.
    """
    log = logging.getLogger('append2corpus')
    # find targets for contour
    mask_rus = np.array(np.unique(ru_map[mask]), dtype=int)  # map to the rus
                                                            # TODO cast int to int because some are string??
    contour_targets = np.c_[ru_f0s[mask_rus],
                            ru_coefs[mask_rus]]
    # generate ramps for contour
    contour_scope = contour_targets.shape[0]
    ramp_global = np.arange(contour_scope)[::-1]

    landmark_unit = ru_map[landmark_ind] - mask_rus[0]  # relative position of landmark
    unit_scope_1 = np.arange(landmark_unit+1)  # including landmark unit
    unit_scope_2 = np.arange(contour_scope - landmark_unit - 1)
    ramp_local = np.c_[np.r_[unit_scope_1[::-1], unit_scope_2[::-1]],
                       np.r_[np.linspace(0, 10, num=unit_scope_1.size)[::-1],
                             # if there is 1 unit this will favor the 0
                             np.linspace(0, 10, num=unit_scope_2.size)],
                       np.r_[unit_scope_1, unit_scope_2]]

    contour_ramps = np.c_[ramp_local, ramp_global].T

#    if debug:  # debug log.info
    log.debug('-'*42)
    log.debug(contour_type, mask, landmark_ind)
    log.debug(contour_targets)
    log.debug(contour_ramps)
    log.debug('-'*42,'\n')

    # add to corpus
    for l in np.c_[rus[mask_rus], mask_rus, contour_ramps.T, contour_targets]:
        row = pd.Series([barename, phrase_type, contour_type] + l.tolist(),
                      columns)
        corpus = corpus.append([row], ignore_index=True)
#    assert(contour_type!='DD')  # just break it : )
    return corpus
#%%
def contour_scope_count(corpus, phrase_type, max_scope=20):
    '''
    Count all the scope contexts of a contour.

    Parameters
    ==========
    corpus : pandas DataFrame
        Holds all the data.
    phrase_type : str
        Phrase contour subset within which to count.
    max_scope : int
        Maximum scope to take into account
    '''
    #%%
    log = logging.getLogger('scope_count')
#    phrase_type = 'DC'
    corpus_phrase = corpus[corpus['phrasetype'] == phrase_type]
    corpus_phrase = corpus_phrase.reset_index()
#    contour_types = np.unique(corpus_phrase['contourtype'])
    contour_types = corpus_phrase['contourtype'].unique()
    scope_counts = {}
    for contour_type in contour_types:
        scope_counts[contour_type] = np.zeros((max_scope,max_scope),dtype=int)
    contour_ends = corpus_phrase['ramp4'] == 0
    contour_ends = contour_ends.index[contour_ends == True].tolist()
    for i, contour_end in enumerate(contour_ends):
        contour_type = corpus_phrase.loc[contour_end]['contourtype']
        if i == 0:
            contour_start = 0
        else:
            contour_start = contour_ends[i-1] + 1
        scope_tot = int(corpus_phrase.loc[contour_start]['ramp4']) + 1
        scope_left = int(corpus_phrase.loc[contour_start]['ramp1']) + 1
        cnt_left = 0
        while corpus_phrase.loc[contour_start + cnt_left]['ramp1'] != 0:  # we're looking for the end
            cnt_left += 1

        if cnt_left+1 <= scope_tot-1:  # if not the end of total scope
            scope_right = int(corpus_phrase.loc[contour_start + cnt_left+1]['ramp1']) + 1
        else:
            scope_right = 0

        assert (scope_tot == scope_left + scope_right) , log.error('Scope length doesn''t match!')
        scope_counts[contour_type][scope_left, scope_right] = \
            scope_counts[contour_type][scope_left, scope_right] + 1
#%%
    return scope_counts