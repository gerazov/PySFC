#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySFC - functions for signal processing.

@authors:
    Branislav Gerazov Nov 2017

Copyright 2017 by GIPSA-lab, Grenoble INP, Grenoble, France.

See the file LICENSE for the licence associated with this software.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal as sig

def f0_smooth(pitch_ts, f0s, plot=False):
    '''
    Smooth the f0.

    Parameters
    ==========
    pitch_ts : ndarray
        Pitch marks timepoints.
    f0s : ndarray
        f0s at those timepoints.
    plot : bool
        Plot smoothing results.
    '''
    fs = 200
    t = np.arange(0, pitch_ts[-1]+.01, 1/fs)
    interfunc = interp1d(pitch_ts, f0s, kind='linear', bounds_error=False, fill_value=0)
    f0s_t = interfunc(t)
    fl = 30  # Hz
    order= 8

#    b_iir, a_iir = sig.iirfilter(order, np.array(fl/(fs/2)), btype='lowpass', ftype='butter')
    b_fir = sig.firwin(order, fl, window='hamming', pass_zero=True, nyq=fs/2)
#    f0s_t_lp = sig.lfilter(b_iir, a_iir, f0s_t)
#    f0s_t_lp = sig.filtfilt(b_iir, a_iir, f0s_t)
    f0s_t_lp = sig.filtfilt(b_fir, [1], f0s_t)

    # now find the points you need:
    interfunc_back = interp1d(t, f0s_t_lp, kind='linear', bounds_error=True, fill_value=None)
    f0s_smooth = interfunc_back(pitch_ts)

    if plot:
        plt.figure()
        plt.subplot(2,1,1)
    #    w, h_spec = sig.freqz(b_iir, a_iir)
        w, h_spec = sig.freqz(b_fir, 1)
        plt.plot(w/np.pi*fs/2,
                 20*np.log10(np.abs(h_spec)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid('on')
        plt.subplot(2,1,2)
        plt.plot(t, f0s_t)
        plt.plot(t, f0s_t_lp)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.grid('on')

    return f0s_smooth

