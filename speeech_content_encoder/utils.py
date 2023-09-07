import torch
import torchaudio
import torchaudio.transforms as T
import librosa

import glob
import os
import numpy as np
import argparse
import json
from scipy.io.wavfile import write

def specAug(Speech_File,sample_rate):
    my_audio_as_np_array, my_sample_rate= librosa.load(Speech_File,sr=sample_rate)

    #transform to melSpectrogram
    spec = librosa.feature.melspectrogram(y=my_audio_as_np_array,
                                        sr=my_sample_rate, 
                                            n_fft=1024, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                     n_mels=201)
    print(spec.shape)
    input()
    #define time stretch, time masking and FrequencyMasking
    stretch = T.TimeStretch()
    TimeMasking = T.TimeMasking(time_mask_param=80)
    FreMasking = T.FrequencyMasking(freq_mask_param=80)
    
    rate = 1.2
    spec = stretch(torch.from_numpy(spec), rate).real

    spec = TimeMasking(spec.unsqueeze(0))

    spec = FreMasking(spec)

    # transform augmented spectrogram to wavform
    wavform = librosa.feature.inverse.mel_to_audio(spec.float().squeeze().numpy(), 
                                           sr=my_sample_rate, 
                                           n_fft=2048,
                                           hop_length=512, 
                                           win_length=None, 
                                           window='hann', 
                                           center=True, 
                                           pad_mode='reflect', 
                                           power=2.0, 
                                           n_iter=32)
    
    return torch.from_numpy(wavform)

    

