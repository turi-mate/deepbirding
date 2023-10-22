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

