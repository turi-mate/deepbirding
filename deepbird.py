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

#contents of train_audio
audio_data_dir = 'data/train_audio/'
#train_metadata.csv
csv_file = 'data/train_metadata.csv'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Create a dictionary that maps class labels to numerical labels
data = pd.read_csv(csv_file)
labels_arr = data['primary_label']
class_to_index = {class_label: index for index, class_label in enumerate(labels_arr)}


##TODO Define Modell for Classification
##TODO Make the MelSpectograms the same length
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))

        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 500)  # Adjust the input size based on your spectrogram size
        self.fc2 = nn.Linear(500, num_classes)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor before fully connected layers
        x = x.view(-1, 256 * 4 * 4)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x




def transformAudio(audio,sample_rate):
    # Compute the Mel spectrogram
    n_fft = 1024  # Size of the FFT window
    hop_length = 256  # Hop size for spectrogram frames
    n_mels = 128  # Number of Mel filterbanks

    audio = librosa.resample(audio, orig_sr=32000, target_sr=20000)
    #Compute MelSpectogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=20000, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels, fmin=20)
    # Convert to decibels (log-scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Visualize the Mel spectrogram
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_length,
    #                          cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.show()
    #exit()

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
        # Convert the label to a tensor (assuming it's an integer label)
        # Retrieve the class label for this sample from your dataset's data source
        class_label = self.data.iloc[idx].split('/')[0]

        # Convert the class label to a numerical label
        converted_labels = class_to_index[class_label]
        label_tensor = torch.tensor(converted_labels, dtype=torch.long)  # Use torch.float32 for regression tasks
        # Define the desired shape (e.g., 128 rows and 940 columns)
        desired_length = 1000

        if melSpectogram.shape[1] < desired_length:
            # Pad with zeros
            pad_width = desired_length - melSpectogram.shape[1]
            melSpectogram = np.pad(melSpectogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif melSpectogram.shape[1] > desired_length:
            # Trim to desired length
            melSpectogram = melSpectogram[:, :desired_length]
        print('Len of trimmed or padded melspec:', melSpectogram.shape)
        #exit()
        ##TODO Find the minimum size of arr padding

        return melSpectogram,label_tensor




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
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the AudioClassifier model
num_classes = 264  # Set the number of classes
model = AudioClassifier(num_classes)

# Print the model architecture
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(train_loader)
#exit()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for mel_spectrogram, labels in train_loader:
        # Transfer data to the selected device
        mel_spectrogram = mel_spectrogram.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(mel_spectrogram)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()

    # Print average loss for the epoch
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {average_loss:.4f}")

# Save the trained model (optional)
torch.save(model.state_dict(), 'audio_classifier_model.pth')