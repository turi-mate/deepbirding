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
from PIL import Image

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
    segment_duration = 5.0  # egy segment hossza
    mel_start_freq = 40.0
    mel_end_freq = 15000
    # durationFile = 20 # 0.5min
    durationFile = librosa.get_duration(y=audio, sr=32000)
    audio_segments_arr = []

    durationFileInSamples = int(durationFile * 32000)
    segmentDurationInSamples = int(segment_duration * 32000)

    for startSample in range(0, durationFileInSamples, segmentDurationInSamples):
        endSample = startSample + segmentDurationInSamples

        sampleVec = audio[startSample:endSample]

        # Get mel spectrogram
        melSpec = librosa.feature.melspectrogram(y=sampleVec, sr=32000, n_fft=n_fft,
                                                 hop_length=hop_length, n_mels=n_mels, fmin=mel_start_freq,
                                                 fmax=mel_end_freq, power=2.0)

        # Convert to decibels (log-scale)
        mel_spec_db = librosa.power_to_db(melSpec, ref=np.max, top_db=100)
        nLowFreqsInPixelToCut = int(2 / 2.0)
        nHighFreqsInPixelToCut = int(4 / 2.0)

        # if nHighFreqsInPixelToCut:
        #     mel_spec_db = mel_spec_db[nLowFreqsInPixelToCut:-nHighFreqsInPixelToCut]
        # else:
        #     mel_spec_db = mel_spec_db[nLowFreqsInPixelToCut:]

        # Normalize values between 0 and 1 (& prevent divide by zero)
        mel_spec_db -= mel_spec_db.min()
        melSpecMax = mel_spec_db.max()
        if melSpecMax:
            melSpec /= melSpecMax

        maxVal = 255.9
        mel_spec_db *= maxVal
        mel_spec_db = maxVal - mel_spec_db
        audio_segments_arr.append(mel_spec_db)

        # # Resize
        # specImagePil = Image.fromarray(melSpec.astype(np.uint8))
        # specImagePil = specImagePil.resize(imageSize, interpolationMethod)
        #
        # # Expand to 3 channels
        # specImage = specImagePil.convert('RGB')
        # plt.imshow(specImage)
        # plt.axis('off')  # Turn off axis labels and ticks
        # plt.show()

        # Visualize the Mel spectrogram
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_length,
        #                          cmap='viridis')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel Spectrogram')
        # plt.show()

    # exit()

    # [x,y] 1D n_mels = 128 2D audio_length
    return audio_segments_arr


# Define a function to resize a Mel spectrogram to a target shape
def resize_mel_spectrogram(mel_spec, target_shape):
    return scipy.ndimage.zoom(mel_spec, (1, target_shape / mel_spec.shape[1]))

class AudioClassificationDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, shuffle=False,batch_size = 10):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.audio_data_filename_arr = []
        self.labels = []
        self.data = self._load_data()
        self.shuffle = shuffle
        self.batch_size = batch_size

    #Itt tortenik a labelek es a hozzajuk tartozo fileok kiszedese a csv segitsegevel
    def _load_data(self):
        data = pd.read_csv(self.csv_file)
        #labels_arr = data['primary_label']
        self.audio_data_filename_arr = data['filename']

        return self.audio_data_filename_arr

    def __len__(self):
        return len(self.audio_data_filename_arr)

    def __getitem__(self, idx):
        fixed_size = 30
        ##TODO file does not exist check
        audio_file = os.path.join(audio_data_dir, self.audio_data_filename_arr[idx])
        print('audiofile',audio_file)
        self.labels = self.data.iloc[idx].split('/')[0]
        #print('label',self.labels)

        waveform, sample_rate = librosa.load(audio_file, sr=None)

        melSpectogram = transformAudio(waveform,sample_rate)

        # Apply fixed size padding or trimming to each array in the list
        for i in range(len(melSpectogram)):
            # Ensure that each segment has the default shape (128, 626)
            if melSpectogram[i].shape != 626:
                melSpectogram[i] = resize_mel_spectrogram(melSpectogram[i], 626)

        # Apply fixed size padding or trimming
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
        print('label:', class_label)

        # Convert the class label to a numerical label
        converted_labels = class_to_index[class_label]
        label_tensor = torch.tensor(converted_labels, dtype=torch.long)  # Use torch.float32 for regression tasks
        print('Len of trimmed or padded melspec:', melSpectogram[0].shape)

        return melSpectogram,label_tensor




# Create an instance of your custom dataset
batch_size = 32
dataset = AudioClassificationDataset(audio_data_dir, csv_file, shuffle=True,batch_size=batch_size)

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

  # Adjust as needed
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
print('here', train_loader)
# exit()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for mel_spectrogram, labels in dataset:

        mel_spectrogram = np.array(mel_spectrogram)
        for index,lab in enumerate(mel_spectrogram):
            #Visualize the Mel spectrogram
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(mel_spectrogram[index], x_axis='time', y_axis='mel', sr=32000, hop_length=256,
            #                          cmap='viridis')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Mel Spectrogram')
            # plt.show()
            # Convert the NumPy array to a PyTorch tensor
            mel_spectrogram_tensor = torch.tensor(mel_spectrogram[index], dtype=torch.float32)
            mel_spectrogram_tensor.to(device)
            labels.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(mel_spectrogram)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            running_loss += loss.item()

        exit()
        # Transfer data to the selected device


    # Print average loss for the epoch
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {average_loss:.4f}")

# Save the trained model (optional)
torch.save(model.state_dict(), 'audio_classifier_model.pth')