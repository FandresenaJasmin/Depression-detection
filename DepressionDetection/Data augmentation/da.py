"""
@author: Zaineb Abdelli

objective: aiming to augment all data and save it in a hierarchical structure similar to the original one
"""




import librosa
from nlpaug.util.visual.wave import VisualWave
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
from pathlib import Path
import csv, re, os

target_data_csv = Path('D:/DAIC WOZ # EMNA/full_dataset.csv')
i=0
with open(target_data_csv, 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        try:
            i=i+1
            print(i)
            participant_ID = str(re.split('\t', row[0])[0])              
            csv_file_path = 'D:/Emna/DAIC WOZ # EMNA/DAIC WOZ # EMNA/'+participant_ID+'_P/split/Participant/'
            path='C:/Users/yobitrust/Desktop/DAIC_augmented/'+participant_ID+'_P/split/Participant/'
            try:  
                os.makedirs(path)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:
                print("Creation of the directory %s success" % path)
            if os.path.exists(csv_file_path):
                input_directory = Path(csv_file_path)
                for my_filename in input_directory.glob("*_AUDIO_*.wav"):
                    audio, sampling_rate = librosa.load(my_filename)
                    VisualWave.visual('Original', audio, sampling_rate)
                        
                    flow = naf.Sequential([
                        naa.NoiseAug(),
                        naa.PitchAug(sampling_rate=sampling_rate, pitch_factor=1.5),
                        naa.ShiftAug(sampling_rate=sampling_rate, shift_max=2),
                        naa.SpeedAug(speed_factor=1.5)
                    ])
                    augmented_audio = flow.augment(audio)
                    VisualWave.visual('augment', augmented_audio, sampling_rate)
                    my_filename=my_filename.stem
                    librosa.output.write_wav(path+my_filename+'.wav',augmented_audio, sampling_rate, norm=False)
        except ValueError:
            print("Skipping the following line: ", row[0])
csvFile.close()                        
