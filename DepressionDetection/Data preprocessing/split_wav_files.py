#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""
Created on Sat Apr 20 15:58:06 2019

@author: A. Komaty
@email: akomaty@gmail.com
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
To run the script:
1. Create a virtual environment:
    $ python3 -m venv /path/to/new/virtual/env

* Install the dependencies using:
    $ pip install -r requirements.txt

* Run the script:
    $ python split_wav_files.py -d YOUR_DATABASE_DIRECTORY/Audio_files/

"""

import csv, re, os
from pydub import AudioSegment
from glob import glob
from pathlib import Path

def main(database_path):
    
    for wav_file in database_path.glob("*/*.wav"):
        current_dir = wav_file.parent
        # Check if the split directory and subdirectories exist, if not, create them
        if not current_dir.joinpath('split/Ellie').exists():
            current_dir.joinpath('split/Ellie').mkdir(parents=True)
        if not current_dir.joinpath('split/Participant').exists():
            current_dir.joinpath('split/Participant').mkdir(parents=True)
    
        # Extract the .wav filename   
        wav_filename = wav_file.stem
        print("## \n Splitting the file: ", wav_filename)
        # Read the transcript csv file
        csv_path = current_dir.joinpath(re.split('_AUDIO', wav_filename)[0]+'_TRANSCRIPT.csv')
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            c_Ellie = 0; c_Participant = 0 # Create counters for Ellie and for the participant
            for row in reader:
                try:
                    t1 = float(re.split('\t', row[0])[0])*1000 # in milliseconds
                    t2 = float(re.split('\t', row[0])[1])*1000 # in milliseconds
                    # Read the audio file
                    audio_file = AudioSegment.from_wav(wav_file)
                    # Cut the audio file using t1 and t2
                    audio_file = audio_file[t1:t2]
                    if (re.split('\t', row[0])[2]=='Ellie'):
                        audio_file.export(current_dir.joinpath('split/Ellie/'+wav_filename+'_'+str(c_Ellie)+'.wav'), format="wav") #Exports to a wav file in the split directory.
                        c_Ellie+=1
                    elif (re.split('\t', row[0])[2]=='Participant'):
                        audio_file.export(current_dir.joinpath('split/Participant/'+wav_filename+'_'+str(c_Participant)+'.wav'), format="wav") #Exports to a wav file in the split directory.
                        c_Participant+=1
                except ValueError:
                    print("Skipping the following line: ", row[0])
        csvFile.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='database_root', type=str, help = 'PATH to the Database directory')
    args = parser.parse_args()
    main(Path(args.database_root))