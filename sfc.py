#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySFC - Python implementation of the Superposition of Functional Contours (SFC)
prosody model [1].

[1] Bailly, Gérard, and Bleicke Holm. "SFC: a trainable prosodic model."
    Speech communication 46, no. 3 (2005): 348-364.

@authors:
    Branislav Gerazov Oct 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import pandas as pd
from sfc import sfc_params, sfc_corpus, sfc_learn, sfc_plot
import pickle
from datetime import datetime
import logging
import os
import shutil

start_time = datetime.now()  # start stopwatch

#%% logger setup
logging.basicConfig(filename='sfc.log', filemode='w',
                    format='%(asctime)s %(name)-12s: %(levelname)-8s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s: %(message)s')
console.setFormatter(formatter)  # tell the handler to use this format
logging.getLogger('').addHandler(console)  # add the handler to the root logger

#%% init params
params = sfc_params.Params()

#%% mkdirs
if os.path.isdir(params.save_path):  # delete them
    if params.remove_folders:
        shutil.rmtree(params.save_path, ignore_errors=False)
        os.mkdir(params.save_path)
        for phrase_type in params.phrase_types:
            os.mkdir(params.save_path+'/'+phrase_type+'_f0')
            os.mkdir(params.save_path+'/'+phrase_type+'_dur')
            os.mkdir(params.save_path+'/'+phrase_type+'_exp')
else:
    os.mkdir(params.save_path)
    for phrase_type in params.phrase_types:
        os.mkdir(params.save_path+'/'+phrase_type+'_f0')
        os.mkdir(params.save_path+'/'+phrase_type+'_dur')
        os.mkdir(params.save_path+'/'+phrase_type+'_exp')

#%% load processed corpus or redo
if not params.load_processed_corpus:
    #% read or rebuild corpus
    if params.load_corpus:
        logging.info('Loading corpus ...')
        with open(params.pkl_path + params.corpus_name + '.pkl', 'rb') as f:
            data = pickle.load(f)
            fpro_stats, corpus, utterances, phone_set, phone_cnts = data
            f0_ref, isochrony_clock, isochrony_gravity, disol, stats = fpro_stats

    else:  # rebuild corpus
        logging.info('Rebuilding corpus ...')
        fpro_stats, corpus, utterances, phone_set, phone_cnts = \
            sfc_corpus.build_corpus(params)

        f0_ref, isochrony_clock, isochrony_gravity, disol, stats = fpro_stats
        corpus = sfc_corpus.downcast_corpus(corpus, params.columns)
        with open(params.pkl_path + params.corpus_name + '.pkl', 'wb') as f:
            fpro_stats = f0_ref, isochrony_clock, isochrony_gravity, disol, stats  # to avoid bad headers
            data = (fpro_stats, corpus, utterances, phone_set, phone_cnts)
            pickle.dump(data, f, -1)  # last version

#%% do analysis-by-synthesis
    # fix and add columns to the corpus
    corpus = sfc_corpus.expand_corpus(corpus,params)

#%% get the scope counts
    dict_scope_counts ={}
    for phrase_type in params.phrase_types:
        # this doesn't take care of good phrases!!!
        dict_scope_counts[phrase_type] = sfc_corpus.contour_scope_count(corpus,
                                                                     phrase_type,
                                                                     max_scope=40)

    #%% phrase loop
    # init dictionaries per phrase type
    dict_contour_generators = {}
    dict_losses = {}
    dict_files = {}
    for phrase_type in params.phrase_types:
        logging.info('='*42)
        logging.info('='*42)
        logging.info('Analysis-by-synthesis for phrase type {} from {} ...'.format(
                      phrase_type, params.phrase_types))

        # init contour generators
        logging.info('Initialising contour generators and masks ...')
        contour_generators = {}
        contour_keys = [phrase_type]
        for function_type in params.function_types:
#            if dict_scope_counts[phrase_type][function_type].sum() > 0:
            if function_type in dict_scope_counts[phrase_type].keys():
                contour_keys.append(function_type)

        for contour_type in contour_keys:
            contour_generators[contour_type] = \
                sfc_learn.construct_contour_generator(params)
        # save them in dictonary
        dict_contour_generators[phrase_type] = contour_generators

        # create masks
        files , mask_all_files, mask_file_dict, \
        mask_contours, n_units_dict, mask_unit_dict = \
                sfc_corpus.create_masks(corpus, phrase_type, contour_keys, params)
        dict_files[phrase_type] = files

        corpus, dict_contour_generators[phrase_type], dict_losses[phrase_type] = \
            sfc_learn.analysis_by_synthesis(corpus, mask_all_files, mask_file_dict, mask_contours,
                              n_units_dict, mask_unit_dict, contour_keys,
                              contour_generators, params)

    ## save results
    # delete not good files
    corpus = corpus[corpus.notnull().all(1)]

    # downcast
    corpus.loc[:, 'f01':] = corpus.loc[:, 'f01':].apply(pd.to_numeric, downcast='float')

    with open(params.pkl_path + params.processed_corpus_name+'.pkl', 'wb') as f:
        data = (corpus, fpro_stats, utterances, dict_files, dict_contour_generators,
                dict_losses, dict_scope_counts)
        pickle.dump(data, f, -1)  # last version
else:
#%%  if load processed data
    with open(params.pkl_path + params.processed_corpus_name+'.pkl', 'rb') as f:
        data = pickle.load(f)
        corpus, fpro_stats, utterances, dict_files, dict_contour_generators, \
                dict_losses, dict_scope_counts = data

#%% make a DataFrame from utterances
db_utterances = pd.DataFrame(data=list(utterances.values()), index=utterances.keys(), columns=['utterance'])
db_utterances["length"] = db_utterances.utterance.apply(lambda x: len(x.split()))

#%% get colors
colors = sfc_plot.init_colors(params)

#%% plot last iteration for every file
if params.plot_contours:
    logging.info('='*42)
    logging.info('='*42)
    logging.info('Plotting final iterations ...')
    for phrase_type, files in dict_files.items():
        for file in files:
#            #%% plot one file
##            file = 'DC_393.fpro'
#            file = 'yanpin_000003.TextGrid'
            logging.info('Plotting f0 and dur for file {} ...'.format(file))
            sfc_plot.plot_contours(params.save_path+'/'+phrase_type+'_f0/', file, utterances,
                                   corpus, colors, params, plot_contour='f0', show_plot=True)

            sfc_plot.plot_contours(params.save_path+'/'+phrase_type+'_dur/', file, utterances,
                                   corpus, colors, params, plot_contour='dur', show_plot=False)

#%% plot losses
logging.info('='*42)
logging.info('Plotting losses ...')
for phrase_type, losses in dict_losses.items():
    #%% plot single phrase type
    phrase_type = 'DC'
    losses = dict_losses[phrase_type]
    sfc_plot.plot_losses(params.save_path, phrase_type, losses, show_plot=True)

#%% plot expansion
logging.info('Plotting expansions ...')

for phrase_type, contour_generators in dict_contour_generators.items():
#    #%% plot single phrase type
#    phrase_type = 'DC'
#    contour_generators = dict_contour_generators[phrase_type]
    scope_counts = dict_scope_counts[phrase_type]
    sfc_plot.plot_expansion(params.save_path+'/'+phrase_type+'_exp/', contour_generators,
                           colors, scope_counts, phrase_type, params, show_plot=False)

#%% final losses
logging.info('Plotting final losses ...')
sfc_plot.plot_final_losses(dict_losses, params, show_plot=False)


#%% now copy figures in new folder sorted by loss
if params.copy_worst:
    ## 100 worst files
    logging.info('Copying 100 worst files ...')
    sfc_plot.plot_worst(corpus, params, n_files=100)
    #
    ## all DC files
    logging.info('Copying all DC from worst to best ...')
    sfc_plot.plot_worst(corpus, params, phrase='DC')

#%% wrap up
end_time = datetime.now()
dif_time = end_time - start_time
logging.info('='*42)
logging.info('Finished in {}'.format(dif_time))


