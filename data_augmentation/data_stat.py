import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import scipy.signal
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import os

audio_data_dir = 'data/train_audio/'
csv_file = 'data/train_metadata.csv'

data = pd.read_csv(csv_file)
labels_arr = data['primary_label']
audio_data_filename_arr = data['filename']

waveforms = []
sample_rates = []
num_of_not_exist = 0

for file_name in audio_data_filename_arr:
    audio_file = os.path.join(audio_data_dir, file_name)

    if os.path.exists(audio_file):
        waveform, sample_rate = librosa.load(audio_file, sr=None)

        waveforms.append(librosa.get_duration(y=waveform, sr=sample_rate))
        sample_rates.append(sample_rate)
    else:
        num_of_not_exist += 1

print('min waveform ', min(waveforms))
print('max waveform  ', max(waveforms))

print('min sample rate ', min(sample_rates))
print('max sample rate ', max(sample_rates))

print('num of not exist ', num_of_not_exist)