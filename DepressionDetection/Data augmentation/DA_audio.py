"""
@Ali Komati

augment data for one audio recording
"""



import librosa
from nlpaug.util.visual.wave import VisualWave
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf



path = '/home/akomaty/Documents/Projets/Co-encadrement_Emna_Rejaibi/Audio_files/300_P/split/Participant/300_AUDIO_fixed_5.wav'
audio, sampling_rate = librosa.load(path)
VisualWave.visual('Original', audio, sampling_rate)

"""
Noise injection
It simply add some random value into data.
"""
aug = naa.NoiseAug(nosie_factor=0.05)

augmented_audio = aug.substitute(audio)
VisualWave.visual('Noise', augmented_audio, sampling_rate)

"""
Pitch Augmenter
This augmentation is a wrapper of librosa function. It change pitch randomly
"""
aug = naa.PitchAug(sampling_rate=sampling_rate, pitch_factor=0.5)

augmented_audio = aug.substitute(audio)
VisualWave.visual('Pitch', augmented_audio, sampling_rate)

"""
Shift Augmenter
The idea of shifting time is very simple.
It just shift audio to left/right with a random second. 
If shifting audio to left (fast forward) with x seconds, first x seconds will mark as 0 (i.e. silence). 
If shifting audio to right (back forward) with x seconds, last x seconds will mark as 0 (i.e. silence).
"""
aug = naa.ShiftAug(sampling_rate=sampling_rate, shift_max=0.2)

augmented_audio = aug.substitute(audio)
VisualWave.visual('Shift', augmented_audio, sampling_rate)

"""
Speed Augmenter
Same as changing pitch, this augmentation is performed by librosa function. It stretches times series by a fixed rate.
"""
aug = naa.SpeedAug(speed_factor=1.5)

augmented_audio = aug.substitute(audio)
VisualWave.visual('Speed', augmented_audio, sampling_rate)

# Frequency Masking and Time Masking

flow = naf.Sequential([
    naa.NoiseAug(),
    naa.PitchAug(sampling_rate=sampling_rate, pitch_factor=1.5),
    naa.ShiftAug(sampling_rate=sampling_rate, shift_max=2),
    naa.SpeedAug(speed_factor=1.5)
])
augmented_audio = flow.augment(audio)
VisualWave.visual('Speed', augmented_audio, sampling_rate)