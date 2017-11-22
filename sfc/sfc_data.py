#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySFC - data input and analysis utility functions.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from natsort import natsorted
import tgt  # textgrid tools
from scipy.interpolate import interp1d
import logging
import sys
import seaborn as sns
import pickle
import sfc_dsp
import sfc_plot

#%%

def read_fpro(filename, re_fpro, database='french'):
    """
    Function that reads fpro files.

    Parameters
    ----------
    filename : str
        Full path and filename to read.

    re_fpro : compiled reg ex
        To parse header of fpro.

    database : str
        Which settings to use.

    Output
    ----------
    fpro_stats : tuple
        Holds:
            F0_Ref : int
                reference f0 used to normalise f0 contour
            CLOCK : float
                Isochrony clock of syllable durations
            R : float
                Coefficient of the influence of isochrony clock.
            DISOL : float
                Expected duration of "isolated consonants" associated with the fpro-object.
            Stats : str
                Path to phone duration statistics used to calculate duration coeff
    f0s : ndarray
        f0s at 10, 50 and 90% of each vowel nucleus.
    enrs : ndarray
        Energy of each unit.
    units : ndarray
        IPCG unit - between two vowel onsets (modified syllable)
    durunits : ndarray
        Duration of IPCG unit in miliseconds.
    durcoeffs : ndarray
        Stretching (+) or compression (-) coefficient of IPCG unit based on phone statistics.
    tpss : ndarray
        Time stamps of beginning of phone in miliseconds.
    dursegs : ndarray
        Duration of phones in miliseconds.
    phones : ndarray
        Phone list.
    sylls : ndarray
        syllable marks: Syll is the start of a syllable, Acc is the accented syllable.
    poss : ndarray
        part of speech tags: Aux - auxilliary, Det - determiner, Vrb - verb, Adv - adverb,
        Npr - name, Nom - noun, Num - number, Adq - adjective, Pps - preposition, Pt - phrase terminal.
    orthographs : ndarray
        Written form of utterance.
    phrase : ndarray
        Phrase scope.
    levels : ndarray
        All the other function scopes.

    """
# =============================================================================
#F0_Ref=   126 Hz, CLOCK= 190.000 ms, R=0.2, DISOL=   0.000 ms,  Stats= /research/crissp/SYNTHESE/_tts/_francais//_polysons/_yfrench/ym.STAT_DUR
#    F1    F2    F3  ENR   UNIT  DURUNIT    COEFF    TPS    DUREE  PHON  SYLL      LEX          ORTHOGRAPHE  PHRASE    NIV1    NIV2
#     *     *     *   17      *        *        *      0       92     _     *        *                    *       *       *       *
#     *     *     *   52  z^e^d      259    0.047     92      103    z^   Acc      Vrb                 J'AI     :FF     :FF       *
#    -1   -27   -16   68  z^e^d      259    0.047    195       70    e^     *        *                    *       *       *       *
#   -34   -24     4   51  z^e^d      259    0.047    265       86     d   Syl      Num                 DEUX       *     :DD     :FF
#    52   117    89   71     xf      294    0.262    351      114     x     *        *                    *       *       *       *
#     *     *     *   41     xf      294    0.262    465      180     f   Acc      Nom                 FOUS       *       *     :DG
#   -73  -138  -120   54     u_      299    0.992    645       99     u     *        *                    *       *       *       *
#     *     *     *   25     u_      299    0.992    744      200     _   Syl        *                    .     :DC     :FF     :FF
# =============================================================================
#
## chinese
# F1    F2    F3  ENR     UNIT  DURUNIT    COEFF      TPS    DUREE  PHON  SYLL    TONE1    TONE2     ORTHOGRAPHE          TRANSLATION      LEX   CLAUSE     WORD  NIV1
#% test
#    filename = '/nethome/gerazovb/ownCloud/data/_modalites/_fpro/QS_8.fpro'
    log = logging.getLogger('read_fpro')
    barename = filename.split('/')[-1]
    with open(filename) as f:
        # init arrays
        f0s = []
        enrs = []
        units = []
        durunits = []
        durcoeffs = []
        tpss = []
        dursegs = []
        phones = []
        sylls = []
        poss = []
        orthographs = []
        phrase = []
        levels = []
        for i, line in enumerate(f):  # TODO - deal with unicode!
            line = line.rstrip()  # strip trailing spaces and \n
            if i == 0:
                try:
                    fpro_stats = re_fpro.match(line).groups()
#                    f0_ref, clock, r_clock, disol, stats = fpro_stats
                except:
                    log.warning('{} - Header ill formed - skipping!'.format(barename))
                    fpro_stats = None
                    continue

            if database == 'french':
                if i == 1:  # find the number of levels (countour generators)
                    n_levels = line.count('NIV') + line.count('SG') + line.count('PROP') # deal with SG1 SG2
                    if line.count('PROP') > 0:
                        log.warning('{} - PROP column found!'.format(barename))
                    issyll = line.count('SYLL')
                    if issyll == 0:
                        log.warning('{} - No syllables in annotation!'.format(barename))
    #                log.info(n_levels)
                elif i > 1:
                    if issyll:
                        if n_levels > 0:
                            f1, f2, f3, enr, unit, durunit, coeff, tps, duree, phon, syll, lex, \
                                orthographie, phrase_seg, niv = line.split(None, 14)
                            niv = niv.split()
                            niv = niv[:n_levels]
                        else:
                            f1, f2, f3, enr, unit, durunit, coeff, tps, duree, phon, syll, lex, \
                                orthographie, phrase_seg = line.split(None, 13)
                            niv = None
                    else:
                        syll = None
                        if n_levels > 0:
                            f1, f2, f3, enr, unit, durunit, coeff, tps, duree, phon, lex, \
                                orthographie, phrase_seg, niv = line.split(None, 13)
                            niv = niv.split()
                            niv = niv[:n_levels]
                        else:
                            f1, f2, f3, enr, unit, durunit, coeff, tps, duree, phon, lex, \
                                orthographie, phrase_seg = line.split(None, 12)
                            niv = None

            elif database == 'chinese':
                if i == 1:  # find the number of levels (countour generators)
                    n_levels = line.count('NIV') + line.count('WORD')
#                                line.count('TONE') + \
                    issyll = line.count('SYLL')
                    if issyll == 0:
                        log.error('{} - No syllables in annotation - stopping!'.format(barename))
                        break

                elif i > 1:
                    if n_levels > 0:
                        f1, f2, f3, enr, unit, durunit, coeff, tps, duree, phon, syll, \
                            tone1, tone2, orthographie, translation, lex, phrase_seg, niv = line.split(None, 17)
                        niv = niv.split()
                        niv = [tone1, tone2] + niv[:n_levels]
                    else:  # this should never happen!
                        log.error('{} - No words in annotation - stopping!'.format(barename))
                        break

            if i > 1:
                # append these
                f0s.append([f1, f2, f3])
                enrs.append(int(enr))
                units.append(unit)
                durunits.append(0 if durunit is '*' else int(durunit))
                durcoeffs.append(0 if durunit is '*' else float(coeff))
                tpss.append(int(tps))
                dursegs.append(int(duree))
                phones.append(phon)
                sylls.append(syll)
                poss.append(lex)
                orthographs.append(orthographie)
                phrase.append(phrase_seg)
                if niv is not None:
                    levels.append(niv)

        # can't get rid of the silences right here because of phrase and level marks
        f0s = np.array(f0s)
        f0s[f0s == '*'] = np.nan
        f0s = f0s.astype(np.int)
        enrs = np.array(enrs)
        units = np.array(units)
        durunits = np.array(durunits)
        durcoeffs = np.array(durcoeffs)
        tpss = np.array(tpss)
        dursegs = np.array(dursegs)
        phones = np.array(phones)
        sylls = np.array(sylls)
        poss = np.array(poss)
        orthographs = np.array(orthographs)
        phrase = np.array(phrase)
        levels = np.array(levels)
        #% deal with empty levels
        for i, level in enumerate(levels.T):
            if np.alltrue(level == '*'):
#                levels = np.delete(levels,i,axis=1)  # some files are empty
                levels = levels[:,:i]  # stop at first empty level
                log.warning('{} - Empty function levels found!'.format(barename))
                break
        if levels.size == 0:
            log.warning('{} - No function levels found!'.format(barename))
#%%
        return(fpro_stats, f0s, enrs, units, durunits, durcoeffs, tpss, dursegs, phones,
               sylls, poss, orthographs, phrase, levels)

def read_textgrid(filename, params, phone_duration_means=None):
    '''
    Read textgrid and generate fpro data.

    Parameters
    ==========
    filename : str
        Filename to read.
    phone_duration_means : pandas DataFrame
        Phone duration statistics.

    params
    ------
    database : str
        This sets a lot of variables.
    f0_method : str
        Method to extract pitch, including:
            pitch_marks - PointProcess files with pitch_marks
                File type = "ooTextFile short"
                "PointProcess"

                0
                1.5808
                129
                0.1685
                0.1724
                ...

            pca - pitch tier - equivalent with pitch marks but different file format:
                PTS: MARQUE
                FE: 10000  # what to divide to get time
                296 $   # these are voiceless interpolated parts
                1632 $
                1685 ^  # these are actual pitch marks
                1764 ^

            pitch_extractor - default one is from Kaldi

    vowel_marks : list
        Sampling points for the f0s.
    f0_ref : int
        If None f0_ref is extracted from data.
    isochrony_clock : float
        If None find from data.
    show_plot : bool
        Display plots of f0_stat and dur_stat historgrams.
    save_path : str
        Where to save the plots.

    Output
    ======
    fpro_lists : tuple
        Contains: fpro_stats, f0_array, units, durunits, durcoeffs, dursegs,
                    phones, orthographs, phrase, levels

    '''
#    Here's an fpro sample:
#
#F0_Ref=   126 Hz, CLOCK= 190.000 ms, R=0.2, DISOL=   0.000 ms,  Stats= /research/crissp/SYNTHESE/_tts/_francais//_polysons/_yfrench/ym.STAT_DUR
#F1    F2    F3  ENR  UNIT  DURUNIT    COEFF    TPS    DUREE  PHON  SYLL      LEX          ORTHOGRAPHE  PHRASE    NIV1    NIV2
# *     *     *   27     *        *        *      0       28     _     *        *                    *       *       *       *
# *     *     *   40   sef      318    0.052     28      110     s   Syl      Adq                  CES     :FF     :FF     :FF
#16   -16    -2   71   sef      318    0.052    138       70     e     *        *                    *       *       *       *
# *     *     *   39   sef      318    0.052    209      137     f   Syl      Adq             FABULEUX       *       *     :XX
#164   158   153   77    ab      139   -0.229    346       67     a     *        *                    *       *       *       *
#113    97    90   59    ab      139   -0.229    412       72     b   Syl        *                    *       *       *       *
#109   120   108   70    yl      131   -0.093    485       70     y     *        *                    *       *       *       *
#93    95   114   63    yl      131   -0.093    555       60     l   Syl        *                    *       *       *       *
#121   137   140   75    xg      165   -0.159    616       96     x     *        *                    *       *       *       *
#108    78    63   57    xg      165   -0.159    712       69     g   Syl      Nom               GAMINS       *     :DG     :FF
#77    50    -1   75    am      186    0.070    781      106     a     *        *                    *       *       *       *
#-29   -77  -105   65    am      186    0.070    887       80     m   Acc        *                    *       *       *       *
#-107  -144  -161   64   e~_      312    0.736    967      112    e~     *        *                    *       *       *       *
# *     *     *   17   e~_      312    0.736   1079      200     _   Syl        *                    .     :DC     :FF       *

    #%%
    log = logging.getLogger('read_textgrid')
    disol = params.disol
    isochrony_gravity = params.isochrony_gravity
    database = params.database
    f0_method = params.f0_method
    vowel_marks = params.vowel_marks
    f0_ref = params.f0_ref
    isochrony_clock = params.isochrony_clock
    show_plot = params.show_plot
    save_path = params.save_path

    textgrid_folder = params.textgrid_folder
    f0_folder = params.f0_folder

    re_grid = params.re_folder
    re_vowels = params.re_vowels
    re_f0 = params.re_f0
    use_ipcgs = params.use_ipcgs
    syll_level = params.syll_level
    orthographs_level = params.orthographs_level
    phrase_level = params.phrase_level
    tone_levels = params.tone_levels


    # stats
    f0_stats = params.f0_stats
    if not os.path.isfile(params.pkl_path + f0_stats+'.pkl'):  # check in root folder
        f0_stats = None

    dur_stats = params.dur_stats
    if not os.path.isfile(params.pkl_path + dur_stats+'.pkl'):  # check in root folder
        dur_stats = None

    syll_stats = params.syll_stats
    if not os.path.isfile(params.pkl_path + syll_stats+'.pkl'):  # check in root folder
        syll_stats = None

    #%% read files
    files_grid = natsorted([f for f in os.listdir(textgrid_folder) if re_grid.match(f)])
    files_f0 = natsorted([f for f in os.listdir(f0_folder) if re_f0.match(f)])

    #%% f0_ref
    if f0_stats is None:
        log.info('No f0 stats file found, starting f0 analysis using method {}...'.format(f0_method))
        f0_stats, data = extract_f0_stats(files_grid, files_f0, params)
        f0_all, f0_mean, f0_median, f0_kde = data
        sfc_plot.plot_histograms(f0_all, f0_mean, f0_median, f0_kde, save_path,
                                 plot_type='f0', show_plot=show_plot)

    if f0_ref is None:
        with open(params.pkl_path + f0_stats + '.pkl', 'rb') as f:
            data = pickle.load(f)
        f0_all, f0_mean, f0_median, f0_kde = data
        log.info('F0 mean = {} Hz, median = {} Hz, kde = {} Hz'.format(
                    int(f0_mean), int(f0_median), int(f0_kde)))
        # let the user choose which f0_ref to use:
        log.info('Please enter the f0_ref in Hz:')
        f0_ref = int(input())

        log.info('f0_ref set to {}.'.format(f0_ref))

    #%% dur_stats
    if dur_stats == None:
        log.info('No durations stats file found, starting data analysis...')
        dur_stats, data = extract_dur_stats(files_grid, params)
        phone_durations, phone_duration_means, phone_counts, \
            just_phones, phone_duration_total_mean, phone_duration_total_median, \
                phone_duration_total_kde = data
        log.info('Phone duration nmean = {} ms, median = {} ms, kde = {} ms'.format(
                    int(phone_duration_total_mean*1e3), int(phone_duration_total_median*1e3),
                    int(phone_duration_total_kde*1e3)))
        sfc_plot.plot_histograms(just_phones.duration, phone_duration_total_mean,
                                 phone_duration_total_median, phone_duration_total_kde,
                                 save_path, plot_type='phone', show_plot=show_plot)

    if phone_duration_means is None:
        with open(params.pkl_path + dur_stats + '.pkl', 'rb') as f:
            data = pickle.load(f)
        phone_durations, phone_duration_means, phone_counts, \
           just_phones, phone_duration_total_mean, phone_duration_total_median, \
               phone_duration_total_kde = data

    if syll_stats == None:
        log.info('No syllable duration stats file found, starting data analysis...')
        # syllable data if we need it (for the isochrony?)
        syll_stats, syll_data = extract_syll_stats(files_grid, params)
        syll_durations, syll_duration_means, syll_counts, \
            syll_duration_mean, syll_duration_median, syll_duration_kde = syll_data
        sfc_plot.plot_histograms(syll_durations.duration, syll_duration_mean,
                                 syll_duration_median, syll_duration_kde,
                                 save_path, plot_type='syll', show_plot=show_plot)

    # let the user choose which clock to use:
    if isochrony_clock is None:
        with open(params.pkl_path + syll_stats + '.pkl', 'rb') as f:
            data = pickle.load(f)
        syll_durations, syll_duration_means, syll_counts, \
            syll_duration_mean, syll_duration_median, syll_duration_kde = data
        log.info('Syllable duration mean = {} ms, median = {} ms, kde = {} ms'.format(
                int(syll_duration_mean*1e3), int(syll_duration_median*1e3), int(syll_duration_kde*1e3)))
        log.info('Please enter the isochrony_clock in ms:')
        isochrony_clock = int(input())/1000

        log.info('isochrony clock set to {}.'.format(isochrony_clock))

    #isochrony_clock = syll_duration_median
    fpro_stats = f0_ref, isochrony_clock, isochrony_gravity, disol, dur_stats  # these are set
#    file_fpro = 'EX_265.fpro'
#    file_fpro = 'DC_140.fpro'
#    file_fpro = 'chinese_000001.fpro'

    file_bare = filename.split('.')[0]

    if f0_method == 'pitch_marks':
        file_f0 = file_bare + '.PointProcess'
    elif f0_method == 'pca':
        file_f0 = file_bare + '.pca'

    ### read and establish if they exist
    try:
        textgrid = tgt.read_textgrid(textgrid_folder+filename)
    except:
        log.error(sys.exc_info()[0])
        log.error('Can''t read file {} - skipping!'.format(filename))
        raise

    try:
        with open(f0_folder+file_f0, 'r') as f:
            lines = f.readlines()
    except:
        log.error(sys.exc_info()[0])
        log.error('Can''t read file {} - skipping!'.format(file_f0))
        raise

    # read f0
    if f0_method == 'pitch_marks':
        pitch_ts = np.array([float(x) for x in lines[6:]])

    elif f0_method == 'pca':
        divisor = int(lines[1].split(':')[1])
        # take only the ^ lines
        pitch_ts = np.array([float(x.split()[0])/divisor for x in lines[2:]
                                 if x.split()[1] == '^'])
    pitchs = np.diff(pitch_ts)
    pitchs[pitchs==0] = 0.001  # to avoid division with 0
    f0s = 1/pitchs
    f0s = np.r_[f0s[0], f0s]  # repeat first value to keep length
#    f0s = f0s[(f0s > f0_min) & (f0s < f0_max)]  # this will shorten it so no
#    f0s[(f0s > f0_min) & (f0s < f0_max)] = np.nan
    #%% some smoothing...
    f0s = sfc_dsp.f0_smooth(pitch_ts, f0s, plot=False)

    #%% init the fpro lists:
    n_phones = len(textgrid.tiers[0])
    dursegs = np.zeros(n_phones)
    phone_starts = np.zeros(n_phones)
    phone_ends = np.zeros(n_phones)

    f0_array = np.empty((n_phones, len(vowel_marks)))
    f0_array.fill(np.nan)
    durunits = np.empty(n_phones)
    durunits.fill(np.nan)
    durunits_stat = np.empty(n_phones)  # for the statistical average durations
    durunits_stat.fill(np.nan)
    durcoeffs = np.empty(n_phones)
    durcoeffs.fill(np.nan)

    phones = []
    units = []
    orthographs = []
    phrase = []
    levels = []

    # these are not used by the SFC
#    enrs = []
#    tpss = []
#    sylls = []
#    poss = []

    # go through the textgrid
    unit = ''
    unit_indexes = []  # to keep track which phones are in the units
    unit_dur = 0  # accumulate seg durations to get the unit
    unit_dur_stat = 0  # this will accumulate the average durations of the segments in a rhythmic unit
    first_ru = True  # first rhythmic unit (i.e. first vowel or first syllable)

    if not use_ipcgs:  # pre loop for speed
        syll_starts = [x.start_time for x in textgrid.tiers[syll_level]]
#        syll_ends = [x.end_time for x in textgrid.tiers[syll_level]]

    for i, seg in enumerate(textgrid.tiers[0]):  # tier 0 should always be the phones
        phone = seg.text
        phone_start = seg.start_time
        phone_end = seg.end_time
        phone_dur = seg.duration()

        phone_starts[i] = phone_start
        phone_ends[i] = phone_end
        dursegs[i] = phone_dur
        phones.append(phone)

        ## do the f0s
#        if not re_unvoiced.match(phone):
        if not '_' in phone:  # if not silence
            pitch_bool = (pitch_ts > phone_start) & (pitch_ts < phone_end)
            if np.sum(pitch_bool) > 2:  # if there are pitch marks
                f0_seg = f0s[pitch_bool]

                pitch_inds = np.where(pitch_bool)[0]
                interfunc = interp1d(np.linspace(0, 1, pitch_inds.size),
                                     f0_seg, kind='linear',
                                     bounds_error=True, fill_value=np.nan)
                f0_seg = interfunc(vowel_marks)
        #        f0_seg = f0_seg[(f0_seg > f0_min) & (f0_seg < f0_max)]
                f0_array[i,:] = f0_seg

        ## do the units with durs
        if '_' in phone:  # if silence reset counters
            if 0 < i < len(textgrid.tiers[0])-1:  # not start or end silence

                unit += phone  # add it to the unit
                unit_indexes.append(i)
                unit_dur += phone_dur
                unit_dur_stat += phone_duration_means.loc[phone,'duration']

            if unit_indexes:  # this means there is a unit at the moment
                for j in unit_indexes:
                    units.append(unit)
                    durunits[j] = unit_dur
                    durunits_stat[j] = unit_dur_stat

            if not (0 < i < len(textgrid.tiers[0])-1):
                units.append('*')  # for the start/end silence


            # reset
            unit = ''
            unit_indexes = []
            unit_dur = 0
            unit_dur_stat = 0
            first_ru = True

        elif use_ipcgs:  # vowels are the separation points
            if not re_vowels.match(phone):  # if it's not a vowel
                unit += phone  # add it to the unit
                unit_indexes.append(i)
                unit_dur += phone_dur
                unit_dur_stat += phone_duration_means.loc[phone,'duration']

            else:  # if it is a vowel it's the start of the next IPCG so wrap up
                if first_ru:  # if first vowel accumulate
                    first_ru = False
                    unit += phone  # add it to the unit
                    unit_indexes.append(i)
                    unit_dur += phone_dur
                    unit_dur_stat += phone_duration_means.loc[phone,'duration']

                else:  # end the current IPCG and restart accumulators
                    for j in unit_indexes:
                        units.append(unit)
                        durunits[j] = unit_dur
                        durunits_stat[j] = unit_dur_stat

                    # reset
                    unit = phone  # add it to the unit
                    unit_indexes = [i]
                    unit_dur = phone_dur
                    unit_dur_stat = phone_duration_means.loc[phone,'duration']

        elif not use_ipcgs:  # syllables are the separation points
            if not phone_start in syll_starts:  # if it's not start of a syllable
                unit += phone  # add it to the unit
                unit_indexes.append(i)
                unit_dur += phone_dur
                unit_dur_stat += phone_duration_means.loc[phone,'duration']

            else:  # if it is the start of the next syllable wrap up
                if unit_indexes:
                    for j in unit_indexes:
                        units.append(unit)
                        durunits[j] = unit_dur
                        durunits_stat[j] = unit_dur_stat

                # reset
                unit = phone  # add it to the unit
                unit_indexes = [i]
                unit_dur = phone_dur
                unit_dur_stat = phone_duration_means.loc[phone,'duration']


        ### do the orthographs
        orthographs.append('*')
        for interval in textgrid.tiers[orthographs_level]:
            if np.isclose(interval.start_time, phone_start, atol=1e-3):
                orthographs[-1] = interval.text
            if np.isclose(interval.end_time, phone_end, atol=1e-3):  # for speed
                break

        ### do the phrase
        phrase.append('*')
        for interval in textgrid.tiers[phrase_level]:
            if np.isclose(interval.start_time, phone_start, atol=1e-3):
                phrase[-1] = interval.text
            if np.isclose(interval.end_time, phone_end, atol=1e-3):  # for speed
                break

    ## do the levels
    n_tiers = len(textgrid.tiers)
    levels = []
    if database in ['chinese']:  # if chinese then add tones to levels
        for j in tone_levels:
            level = []
            for seg in textgrid.tiers[0]:  # tier 0 is the phones
                phone_start = seg.start_time
                phone_end = seg.end_time

                level.append('*')
                for interval in textgrid.tiers[j]:
                    if np.isclose(interval.start_time, phone_start, atol=1e-3):
                        level[-1] = interval.text
                    if np.isclose(interval.end_time, phone_end, atol=1e-3):  # for speed
                        break
            levels.append(level)

    if n_tiers-1 > phrase_level:
        for j in range(phrase_level+1, n_tiers):
            level = []
            for seg in textgrid.tiers[0]:  # tier 0 is the phones
                phone_start = seg.start_time
                phone_end = seg.end_time

                level.append('*')
                for interval in textgrid.tiers[j]:
                    if np.isclose(interval.start_time, phone_start, atol=1e-3):
                        level[-1] = interval.text
                    if np.isclose(interval.end_time, phone_end, atol=1e-3):  # for speed
                        break
            levels.append(level)

#%% finish up calculations and send back data
### f0 - calc quartertones
    f0_array = 240 * np.log(f0_array / f0_ref) / np.log(2)

### dur coeff
    denom = (1-isochrony_gravity)*durunits_stat + isochrony_gravity*isochrony_clock
    durcoeffs = np.log(durunits / denom)

    phones = np.array(phones)
    units = np.array(units)
    orthographs = np.array(orthographs)

    phrase = np.array(phrase)
    levels = np.array(levels)

# we don\t need these:
#    enrs = np.array(enrs)
#    tpss = np.array(tpss)
#    sylls = np.array(sylls)
#    poss = np.array(poss)

    fpro_lists = fpro_stats, f0_array, units, durunits, durcoeffs, dursegs, \
                    phones, orthographs, phrase, levels
#%%
    return fpro_lists, phone_duration_means
#%%
def extract_f0_stats(files_grid, files_f0, params):
    '''
    Extract the reference f0 used to convert f0 to semitones.

    Parameters
    ==========
    textgrid_folder : str
        Path to textgrid folder.
    f0_folder : str
        Path to f0 folder.

    params
    ======
    files_grid : list
        List of TextGrid files in folder.
    files_f0 : list
        List of f0 files in folder.
    f0_ref_method : str
        Maethod used to determine f0_ref: all, vowels, DC, vowel_marks.
    vowel_marks : lsit
        Sampling points for f0 in vocalic nuclei.
    f0_type : str
        Filetype to read f0 from.
    f0_min : int
        Minimum f0.
    f0_max : int
        Maximum f0.
    re_vowels : compiled reg ex
        To match for vowels.
    database : str
        Name of database used.
    '''
    log = logging.getLogger('f0_stats')

    textgrid_folder = params.textgrid_folder
    f0_folder = params.f0_folder
    f0_ref_method = params.f0_ref_method
    vowel_marks = params.vowel_marks
    f0_method = params.f0_method
    f0_min = params.f0_min
    f0_max = params.f0_max
    re_vowels = params.re_vowels
    database = params.database

    f0_all = np.array([])

    if f0_ref_method in ['all','DC']:  # accumulate f0s through all files
        for file in files_f0:
            # read file
            if f0_ref_method == 'DC' and 'DC' not in file:
                continue
            print('\rreading file: {}'.format(file), end='')
            with open(f0_folder+file, 'r') as f:
                lines = f.readlines()

            if f0_method == 'pitch_marks':
                pitch_ts = np.array([float(x) for x in lines[6:]])

            elif f0_method == 'pca':
                divisor = int(lines[1].split(':')[1])
    #            pitch_ts = np.array([float(x.split()[0])/divisor for x in lines[2:]])
                # take only the ^ lines
                pitch_ts = np.array([float(x.split()[0])/divisor for x in lines[2:]
                                         if x.split()[1] == '^'])

            pitchs = np.diff(pitch_ts)
            f0s = 1/pitchs[pitchs>0]
            f0s = f0s[(f0s > f0_min) & (f0s < f0_max)]
            f0_all = np.r_[f0_all, f0s]
        f0_mean = np.mean(f0_all)
        f0_median = np.median(f0_all)

    elif 'vowels' in f0_ref_method:  # find all vowel segments and accumulate f0s
        f0_all = np.array([])
        for file_grid in files_grid:
            print('\rreading file: {}'.format(file_grid), end='')
            if f0_method == 'pitch_marks':
                file_f0 = file_grid.split('.')[0] + '.PointProcess'
            elif f0_method == 'pca':
                file_f0 = file_grid.split('.')[0] + '.pca'

            try:
                textgrid = tgt.read_textgrid(textgrid_folder+file_grid)
            except:
                log.error(sys.exc_info()[0])
                log.error('Can''t read file {}.'.format(file_grid))
                continue

            try:
                with open(f0_folder+file_f0, 'r') as f:
                    lines = f.readlines()
            except:
                log.error(sys.exc_info()[0])
                log.error('Can''t read file {}.'.format(file_f0))
                continue

            if f0_method == 'pitch_marks':
                pitch_ts = np.array([float(x) for x in lines[6:]])

            elif f0_method == 'pca':
                divisor = int(lines[1].split(':')[1])
                # take only the ^ lines
                pitch_ts = np.array([float(x.split()[0])/divisor for x in lines[2:]
                                         if x.split()[1] == '^'])
            pitchs = np.diff(pitch_ts)
            pitchs[pitchs==0] = 0.001
            f0s = np.r_[0, 1/pitchs]

            for segment in textgrid.tiers[0]:  # tier 0 is the phones
                phone = segment.text
                if re_vowels.match(phone):  # if it is a vowel accumulate
                    # take whole vowel:
                    pitch_bool = (pitch_ts > segment.start_time) & \
                                         (pitch_ts < segment.end_time)
                    f0_seg = f0s[pitch_bool]

                    if f0_ref_method == 'vowel_marks':
                        pitch_inds = np.where(pitch_bool)[0]
                        interfunc = interp1d(np.linspace(0, 1, pitch_inds.size),
                                             f0_seg, kind='linear',
                                             bounds_error=True, fill_value=np.nan)
                        f0_seg = interfunc(vowel_marks)

                    f0_seg = f0_seg[(f0_seg > f0_min) & (f0_seg < f0_max)]
                    f0_all = np.r_[f0_all, f0_seg]
    print()  # new line
    #% set f0_ref
    f0_mean = np.mean(f0_all)
    f0_median = np.median(f0_all)

    #% plot histogram - this is now in sfc_plot
    fig = plt.figure()
    sns.set(color_codes=True, style='ticks')
    ax = sns.distplot(f0_all)

    kde_x, kde_y = ax.get_lines()[0].get_data()
    f0_kde = kde_x[np.argmax(kde_y)]
    plt.close(fig)

    #%% save

    pkl_name = database+'_f0_stats_'+f0_ref_method
    data = f0_all, f0_mean, f0_median, f0_kde

    with open(params.pkl_path + pkl_name +'.pkl', 'wb') as f:
        pickle.dump(data, f, -1)

    log = log.info('f0_stats pkl saved.')
    return pkl_name, data

def extract_dur_stats(files_grid, params):
    '''
    Extract phone durations and calculate statistics.

    Parameters
    ==========
    files_grid : list
        List of TextGrid files in folder.

    params
    ======
    textgrid_folder : str
        Path to textgrid folder.
    database : str
        Name of database used.
    '''
    textgrid_folder = params.textgrid_folder
    database = params.database

    log = logging.getLogger('dur_stats')
    phone_durations = pd.DataFrame(index=np.arange(0, 3e4, dtype=int),
                                   columns=['phone','duration'], dtype=float)
    index = -1
    for file_grid in files_grid:
        print('\rreading file: {}'.format(file_grid), end='')
        ### read and establish if it's readable
        try:
            textgrid = tgt.read_textgrid(textgrid_folder+file_grid)
        except:
            log.error(sys.exc_info()[0])
            log.error('Can''t read file {}.'.format(file_grid))
            continue

        for phone in textgrid.tiers[0]:  # tier 0 is the phones
            index += 1
            phone_durations.loc[index] = [phone.text, phone.duration()]
    print('')  # new line
    phone_durations = phone_durations.loc[:index]  # inclusive indexing in pandas
    phone_counts = phone_durations['phone'].value_counts()
    phone_duration_means = phone_durations.groupby('phone').mean()
    just_phones = phone_durations[~phone_durations.phone.isin(['_','__'])]
    phone_duration_total_mean = just_phones.duration.mean()
    phone_duration_total_median = just_phones.duration.median()

    #%% plot
    fig = plt.figure()
    sns.set(color_codes=False, style="white", context='paper')
    ax = sns.distplot(just_phones.duration)

    kde_x, kde_y = ax.get_lines()[0].get_data()
    phone_duration_total_kde = kde_x[np.argmax(kde_y)]
    plt.close(fig)

    #% save file
    phone_durations.duration = phone_durations.duration.apply(pd.to_numeric,
                                                              downcast='float')
    pkl_name = database+'_phone_dur_stats'

    with open(params.pkl_path + pkl_name +'.pkl', 'wb') as f:
        data = phone_durations, phone_duration_means, phone_counts, \
                just_phones, phone_duration_total_mean, phone_duration_total_median, \
                phone_duration_total_kde
        pickle.dump(data, f, -1)
    log.info('Phone duration stats pkl saved.')
    return pkl_name, data
#%%
def extract_syll_stats(files_grid, params):
    '''
    Extract syllable durations and calculate statistics.

    Parameters
    ==========
    files_grid : list
        List of TextGrid files in folder.

    params
    ======
    textgrid_folder : str
        Path to textgrid folder.
    database : str
        Name of database used.
    syll_tier : int
        Syllable tier in TextGrid annotations.

    '''
    log = logging.getLogger('syll_stats')
    textgrid_folder = params.textgrid_folder
    database = params.database
    syll_level = params.syll_level

    # preallocate for speed
    syll_durations = pd.DataFrame(index=np.arange(0, 3e4, dtype=int),
                                   columns=['syllable','duration'], dtype=float)
    index = -1
    for file_grid in files_grid:
        print('\rreading file: {}'.format(file_grid), end='\n')
        #%
#        file_grid = 'chinese_000001.TextGrid'
        ### read and establish if it's readable
        try:
            textgrid = tgt.read_textgrid(textgrid_folder+file_grid)
        except:
            log.error(sys.exc_info()[0])
            log.error('Can''t read file {}.'.format(file_grid))
#            continue

        for syll in textgrid.tiers[syll_level]:  # syllables tier
            syll_dur = 0  # accumulate through phones (because of sylls with silence)
            syll_text = ''

            for phone in textgrid.tiers[0]:  # tier 0 is the phones
                # check if phone is in syllable
                if (np.isclose(phone.start_time, syll.start_time, atol=1e-3) or
                    phone.start_time > syll.start_time) and \
                        (np.isclose(phone.end_time, syll.end_time, atol=1e-3) or
                         phone.end_time < syll.end_time):

                    syll_text += phone.text

                    # don't count silence in syllable duration
                    if '_' not in phone.text:
                        syll_dur += phone.duration()

                    if np.isclose(phone.end_time, syll.end_time, atol=1e-3):  # cut it short
                        break

            if syll_dur > 0:  # if not all silence
                index += 1
                syll_durations.loc[index, 'syllable':'duration'] = [syll_text,
                                                                    syll_dur]

            # if not all silence and not equal duration
            if syll_dur != 0 and not np.isclose(syll.duration(), syll_dur, atol=1e-3):
#                print('')  # new line
                log.warning('syll dur difference in {} for {}'.format(
                            file_grid, syll_text))
#                raise()

    print('')  # new line
    syll_durations = syll_durations.loc[:index]  # inclusive indexing in pandas

    syll_counts = syll_durations['syllable'].value_counts()
    syll_duration_means = syll_durations.groupby('syllable').mean()
    syll_duration_mean = syll_durations.duration.mean()
    syll_duration_median = syll_durations.duration.median()

    #%% plot and get KDE
    fig = plt.figure()
    sns.set(color_codes=True, style='ticks')
    ax = sns.distplot(syll_durations.duration)

    kde_x, kde_y = ax.get_lines()[0].get_data()
    syll_duration_kde = kde_x[np.argmax(kde_y)]

    plt.close(fig)

    #% save file
    syll_durations.duration = syll_durations.duration.apply(pd.to_numeric,
                                                              downcast='float')
    pkl_name = database+'_syll_dur_stats'

    with open(params.pkl_path + pkl_name +'.pkl', 'wb') as f:
        data = syll_durations, syll_duration_means, syll_counts, \
                syll_duration_mean, syll_duration_median, syll_duration_kde
        pickle.dump(data, f, -1)

    log.info('Syllable duration stats pkl saved.')

    return pkl_name, data