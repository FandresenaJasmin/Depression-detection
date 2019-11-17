#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""
Created on Mon Apr 29 13:20:21 2019

@author: A. Komaty
@email: akomaty@gmail.com

# How to use this code:

This code will read a .wav file and compute the MFCC features. It consists of two parts:

Part 1: Preprocessing: Voice Activity Detection (VAD)
=====================================================
This part uses one of three VADs:

1- Energy Threshold (Energy_Thr)
2- The 4 Hz Modulation (Mod_4Hz)
3- Tow-Gaussian Energy comparison (Energy_2Gauss)

You can use the one that you want by only modifying the variable "vad" in the code!
This preprocessing will give you back an array of labels. This is an array containing one for the frame that corresponds to voice and zero for the frame that corresponds to silence.
If you want, you can apply these labels on the final mfcc to make sure that you are only computing the features for meaningful data.

Part 2: Feature Extraction: MFCC
=====================================================
A set of coefficients are predefined for you, these are the most used coefficients in the literature.
Feel free to change them according to your needs.

For these particular parameters, the resulting feature will have a dimension of (nb_of_frames * 60).
The number of frames is determined by the window shift overlap parameter (win_shift_ms), it is measured in milliseconds.
The number 60 is determined by the following parameters:

1. "n_ceps" which is the number of cepstral coefficients (in our case 19 coefficients)
2. "with_energy=True" adds the energy parameter (1 coefficient)
3. "with_delta=True" adds the first derivative (20 coefficients)
4. "with_delta_delta=True" adds the second derivative (20 coefficients)

If you want features with different dimensions, you should work on these aforementioned parameters!

Example:

For the file  Audio_files/301_P/split/Participant/301_AUDIO_0.wav 
You'll get 32 frames, so your MFCC will have dimensions of (32*60)
"""

from bob.bio.spear.preprocessor import Energy_2Gauss, Mod_4Hz, Energy_Thr, External
from bob.bio.spear.database import AudioBioFile
import bob.ap
import numpy as np

def extract_coef(directory, FileName):
    """ choose the VAD you want from the following options: Energy_2Gauss, Mod_4Hz, Energy_Thr, External, as told by the name External is a predefined class ready to host your preferable VAD algorithm if the available option does not meet your needs """
    """ PAY ATTENTION to the default parameter values of the VAD you're using, and make sure they are compatible with the parameters of the feature extractor"""
    vad = Energy_Thr()
    # specify the directory of the audio file, the extension type and the File name
    #directory = '/home/emna/Desktop/DAIC WOZ # EMNA/done/303_P/split/Participant' # this should be only the directory path and not the filename
    """directory = '/home/akomaty/Documents/Projets/Co-encadrement_Emna_Rejaibi/Audio_files/301_P/split/Participant'"""
    extension = '.wav'
    #FileName = '303_AUDIO_0' # Put your filename here
    # put your file in a format compatible with bob
    myfile = AudioBioFile('', FileName, FileName)
    # read the data from your audio file: fs is the sampling frequency and audio signal
    fs, audio_signal = vad.read_original_data(myfile, directory, extension)
    # apply the VAD algorithm to the audio signal, the result will be a set of labels per frame (labels/frame)
    rate, data, labels = vad((fs, audio_signal))
    
    #print(rate) # rate is Hz
    
    """ The LFCC and MFCC coefficients can be extracted from a audio signal by using bob.ap.Ceps. To do so, several parameters can be chosen by the user. Typically, these are chosen in a configuration file. The following values are the default ones:"""
    win_length_ms = 60 # The window length of the cepstral analysis in milliseconds
    win_shift_ms = 40 # The window shift of the cepstral analysis in milliseconds
    n_filters = 24 # The number of filter bands
    n_ceps = 19 # The number of cepstral coefficients
    f_min = 0. # The minimal frequency of the filter bank
    f_max = 8000. # The maximal frequency of the filter bank
    delta_win = 2 # The integer delta value used for computing the first and second order derivatives
    pre_emphasis_coef = 1.0 # The coefficient used for the pre-emphasis
    dct_norm = True # A factor by which the cepstral coefficients are multiplied
    mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale
    
    """ Once the parameters are chosen, bob.ap.Ceps can be called as follows: """
    ceps_mfcc = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    ceps_mfcc.dct_norm = True
    ceps_mfcc.mel_scale = True
    ceps_mfcc.with_energy = True
    ceps_mfcc.with_delta = True
    ceps_mfcc.with_delta_delta = True
    
    data_mfcc = np.cast['float'](data) # vector should be in **float**
    mfcc = ceps_mfcc(data_mfcc)
    """
    print("number of MFCC coef in the first frame: ")
    print(len(mfcc[0])) # print number of MFCC coef in the first frame
    print("number of frames: ")
    print(len(mfcc)) # print number of frames
    print("the mfcc matrix of the first frame: ")
    print(mfcc[0]) # print the matrix of the first frame
    """
    ceps_lfcc = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    ceps_lfcc.dct_norm = True
    ceps_lfcc.mel_scale = False
    ceps_lfcc.with_energy = True
    ceps_lfcc.with_delta = True
    ceps_lfcc.with_delta_delta = True
    
    data_lfcc = np.cast['float'](data) # vector should be in **float**
    lfcc = ceps_lfcc(data_lfcc)
    """
    print("number of LFCC coef in the first frame: ")
    print(len(lfcc[0])) # print number of LFCC coef in the first frame
    print("number of frames: ")
    print(len(lfcc)) # print number of frames
    print("the lfcc matrix of the first frame: ")
    print(lfcc[0]) # print the matrix of the first frame
    """
    matrix_coef = np.concatenate((mfcc, lfcc), axis=1) # concatenate both matrix: mfcc and lfcc
    
    #np.savetxt("matrice_coef.csv", matrice_coef, delimiter=",")
    
    """
    print("number of coef in the first frame: ")
    print(len(matrice_coef[0])) # print number of coef in the first frame
    print("number of frames: ")
    print(len(matrice_coef)) # print number of frames
    print("the final matrix of the first frame: ")
    print(matrice_coef[0]) # print the matrix of the first frame
    """
    """
    print("number of MFCC coef in the first frame: ")
    print(len(mfcc[0])) # print number of MFCC coef in the first frame
    print("number of frames: ")
    print(len(mfcc)) # print number of frames
    print("total number of MFCC coef in all the frames")
    print(mfcc.size) # print total number of MFCC coef in all the frames
    print("the MFCC coef: ")
    print(mfcc)
    print("the matrix of the first frame: ")
    print(mfcc[0]) # print the matrix of the first frame
    print("the second mfcc coef of the matrix of the first frame: ")
    print(mfcc[0][1]) # print the second mfcc coef of the matrix of the first frame
    print("the forth mfcc coef of the matrix of the first frame: ")
    print(mfcc[0][3]) # print the forth mfcc coef of the matrix of the first frame 
    """
    
    return matrix_coef
            