#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning utils used for the SFC.

Created on Fri Oct 20 15:38:03 2017

@author: gerazovb
"""
#import logging
from sklearn.neural_network import MLPRegressor
import logging
import numpy as np

#%% analysis-by-synthesis
def analysis_by_synthesis(corpus, mask_all_files, mask_file_dict, mask_contours,
                          n_units_dict, mask_unit_dict, contour_keys,
                          contour_generators, params):
    '''
    Runs SFC analysis by synthesis.

    Parameters
    ===========

    corpus : pandas data frame
        Holds all data from corpus.
    orig_columns : list of str
        The original prosodic parameters f0, f1, f2 and coeff_dur,
    target_columns : list of str
        The targets for training the contour generators.
    iterations : int
        Number of iterations of analysis-by-synthesis loop.
    mask_all_files : dataseries bool
        Mask for all good files in the corpus DataFrame.
    mask_file_dict : dict
        Dictionary with masks for each of the files in corpus.
    mask_contours : dict
        Dictinary with masks for each of the contours.
    n_units_dict : dict
        Dictionary with the number of units in each file in corpus.
    mask_unit_dict : dict
        Dictionary of masks for each unit number in corpus.
    contour_keys : list
        Types of functions covered by the contour generators.
    contour_generators : dict
        Dictionary of contour generators.
    '''

    log = logging.getLogger('an-by-syn')
    orig_columns = params.orig_columns
    target_columns = params.target_columns
    iterations = params.iterations
#%%  set initial targets
    log.info('='*42)
    log.info('Setting initial targets....')
    for file, mask_file in mask_file_dict.items():
        n_units = n_units_dict[file]
        for n_unit in range(n_units+1):
            mask_unit = mask_unit_dict[n_unit]
            mask_row = mask_file & mask_unit
            n_contours = corpus[mask_row].shape[0]  # coeff to divide the error
            targets = corpus.loc[mask_row, orig_columns].values/n_contours
            corpus.loc[mask_row, target_columns] = targets

#%% now do the training iterations updating the targets
    losses = {key : np.empty(iterations) for key in contour_keys}
    for i in range(iterations):
        log.info('='*42)
        log.info('Analysis-by-synthesis iteration {} ...'.format(i))
        log.info('='*42)
        pred_columns = [column + '_it{:03}'.format(i) for column in orig_columns]

        for contour_type in contour_keys:
            log.info('Training for contour type : {}'.format(contour_type))
            contour_generator = contour_generators[contour_type]
            mask_row = mask_all_files & mask_contours[contour_type]
            X = corpus.loc[mask_row,'ramp1':'ramp4']
            y_target = corpus.loc[mask_row, target_columns]
            contour_generator.fit(X, y_target)
            losses[contour_type][i] = contour_generator.loss_
            log.info('mean squared error : {}'.format(contour_generator.loss_))
            y_pred = contour_generator.predict(X)
            corpus.loc[mask_row, pred_columns] = y_pred

        #% now sum the predictions for each unit, calculate the error and new targets
        log.info('Summing predictions, calculate the error and new targets ...')
        for file, mask_file in mask_file_dict.items():
            n_units = n_units_dict[file]
            for n_unit in range(n_units+1):
                mask_unit = mask_unit_dict[n_unit]
                mask_row = mask_file & mask_unit
                n_contours = corpus[mask_row].shape[0]  # coeff to divide the error
                y_pred = corpus.loc[mask_row, pred_columns].values
                y_pred_sum = np.sum(y_pred, axis=0)
                y_orig = corpus.loc[mask_row, orig_columns].values  # every row should be the same
                y_error = y_orig - y_pred_sum  # this will automatically tile row to matrix
                targets = y_pred + y_error/n_contours
                corpus.loc[mask_row, target_columns] = targets  # write new targets

    return corpus, contour_generators, losses

def construct_contour_generator(params):
    '''
    Construct Neural Network based contour generator.

    Parameters
    ==========
    learn_rate : float
        Learning rate of NN optimizer.
    max_iter : int
        Maximum of training iterations.
    l2 : float
        L2 regulizer value.
    hidden_units : int
        Number of units in the hidden layer.

    '''
    learn_rate = params.learn_rate
    max_iter = params.max_iter
    l2 = params.l2
    hidden_units = params.hidden_units
    contour_generator = MLPRegressor(hidden_layer_sizes=(hidden_units, ), # 15 in 2004 paper but Config says 17
                                     activation='logistic',  # relu default, logistic in snns
                       batch_size='auto',  # auto batch_size=min(200, n_samples)
                       max_iter=max_iter,  # is this 50 in the original??
                       alpha=l2,  # L2 penalty 1e-4 default - config says should be 0.1??
                       shuffle=True,  # Whether to shuffle samples in each iteration. Only solver=’sgd’ or ‘adam’.
                       random_state=42,
                       verbose=False,
                       warm_start=True,  # When set to True, reuse the solution of the
                                          # previous call to fit as initialization, otherwise,
                                          # just erase the previous solution.
                       early_stopping=False, validation_fraction=0.01,
                           # Whether to use early stopping to terminate training when
                           # validation score is not improving. If set to true, it will
                           # automatically set aside 10% of training data as validation
                           # and terminate training when validation score is not improving
                           # by at least tol for two consecutive epochs. Only solver=’sgd’ or ‘adam’
                       solver='adam',  # adam is newer, I don't think you can use rprop
                       learning_rate_init=learn_rate,  # default 0.001, in Config it's 0.1
                       beta_1=0.9, beta_2=0.999, epsilon=1e-08,
#                       learning_rate='constant',  # Only used when solver='sgd'
    ##    'constant' is a constant learning rate given by ‘learning_rate_init’.
    ##    'invscaling' gradually decreases the learning rate learning_rate_ at each
    ##       time step 't' using an inverse scaling exponent of 'power_t'.
    ##       effective_learning_rate = learning_rate_init / pow(t, power_t)
    ##    'adaptive' keeps the learning rate constant to 'learning_rate_init' as long
    ##       as training loss keeps decreasing. Each time two consecutive epochs fail
    ##       to decrease training loss by at least tol, or fail to increase validation
    ##       score by at least tol if 'early_stopping' is on, the current learning rate
    ##       is divided by 5.
                       power_t=0.5, tol=0.0001,
                       momentum=0.9,  # Momentum for gradient descent update. Only solver=’sgd’.
                       nesterovs_momentum=True) # Only used when solver=’sgd’
    return contour_generator