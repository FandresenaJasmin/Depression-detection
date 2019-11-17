#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:19:15 2019

@author: emna

# How to use this code:

This code will generate the MFCC & LFCC matrix of each frame for each Participant and will save 
it as a csv file.

* The execution steps are as follows: 
    1- Parse the database directory
    2- Within each Participant ID sub-directory: get the wav file for each frame: 
        example: ParticipantID_AUDIO_0.wav
    3- Call the function extract_coef(directory, FileName) from the program vad_mfcc.py 
        to extract the MFCC & LFCC matrix for the wav file in the current loop. 
        Give it the parameters: the directory of the wav file and its name
    4- Save the matrix as a csv file ParticipantID_AUDIO_0.csv in the same folder as the framed 
        wav file ParticipantID_AUDIO_0.wav.

* Run the script under the conda environment: (bob_py3):
    $ python dataset_coef_matrix.py -d YOUR_DATABASE_DIRECTORY/ 


"""

import csv
from pathlib import Path
from vad_mfcc import extract_coef

def main(database_path):
    
    parent_dir = str(database_path)+'/'
    
    for wav_file in database_path.glob("*/split/Participant/*.wav"):
       # Extract the .wav filename   
       wav_file = wav_file.stem
       print("## \n Extracting matrix from file: ", wav_file)
            
       # Extract the Participant ID
       participantID = wav_file[0:3]+'_P'
       
       # Get the Matrix
       directory = parent_dir+participantID+'/split/Participant'
       matrix = extract_coef(directory, wav_file)
       
       # Write the matrix in the csv file
       my_csv_file = directory+'/'+wav_file+'.csv'
       print("## \n Saving matrix as a csv file: ", wav_file+'.csv')
       with open(my_csv_file, 'a') as csvFile:
           writer = csv.writer(csvFile)
           writer.writerows(matrix)
       csvFile.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='database_root', type=str, help = 'PATH to the Database directory')
    args = parser.parse_args()
    main(Path(args.database_root))