import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Dataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

##TODO Define Modell for Classification

def transformAudio(waveform,sample_rate):
    # Compute the Mel spectrogram
    n_fft = 1024  # Size of the FFT window
    hop_length = 256  # Hop size for spectrogram frames
    n_mels = 128  # Number of Mel filterbanks
    #Compute MelSpectogram

    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels, fmin=20)
    # Convert to decibels (log-scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Visualize the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_length,
                             cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()
    exit()

    return mel_spec_db

class AudioClassificationDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.audio_data_filename_arr = []
        self.labels = []
        self.data = self._load_data()

    #Itt tortenik a labelek es a hozzajuk tartozo fileok kiszedese a csv segitsegevel
    def _load_data(self):
        data = pd.read_csv(self.csv_file)
        #labels_arr = data['primary_label']
        self.audio_data_filename_arr = data['filename']

        return self.audio_data_filename_arr

    def __len__(self):
        return len(self.audio_data_filename_arr)

    def __getitem__(self, idx):
        audio_file = os.path.join(audio_data_dir, self.audio_data_filename_arr[idx])
        print('audiofile',audio_file)
        self.labels = self.data.iloc[idx].split('/')[0]
        #print('label',self.labels)

        waveform, sample_rate = librosa.load(audio_file, sr=None)

        melSpectogram = transformAudio(waveform,sample_rate)

        # plt.figure(figsize=(10, 4))
        # plt.plot(waveform.t().numpy())
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Audio Waveform')
        # plt.grid()
        # plt.show()

        #print('here'+self.labels)
        #exit()

        return melSpectogram,self.labels


#contents of train_audio
audio_data_dir = 'data/train_audio/'
#train_metadata.csv
csv_file = 'data/train_metadata.csv'

# Create an instance of your custom dataset
dataset = AudioClassificationDataset(audio_data_dir, csv_file)

# Define train, validation, and test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the number of samples for each split
num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

# Randomly shuffle the dataset
indices = list(range(num_samples))
random.shuffle(indices)

# Split the dataset
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train + num_val]
test_indices = indices[-num_test:]

# Create data loaders for train, validation, and test sets
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

batch_size = 32  # Adjust as needed
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


#print('dataset0',dataset[0])

# Create a DataLoader to batch and shuffle the data
batch_size = 32
learning_rate = 0.001
num_epochs = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
exit()

# Training loop
for epoch in range(num_epochs):
    # Training loop
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()