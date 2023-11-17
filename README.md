## Deepbirding
### Team members: 
- Bence Bihari (IVXWF8), Lehel Hegyi (GSZLZ7), Máté Turi (OPVP7J) left team after Milestone 1)
### Project description: 
This is a university project teamwork for the Deep Learning course. Our task is to take part in the Kaggle BirdCLEF 2023 competition, where we are creating and training a deep learning model that could identify and classify different kinds of bird's chirping. The initial dataset can be found in the official BirdCLEF description.

### Resources used for data preparation:
- Feeding audio datas to CNN: https://www.mdpi.com/2076-3417/11/13/5796?fbclid=IwAR2IWzKZQIj5DcqTbg4VubWJs8CSr0RKhLYVCPOj95zc9-YkkW0WgR8iKsQ
- Converting audio files to Mel Spectograms: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

### Files and functions:
- `data_preprocessing.ipynb` - in this Jupyter notebook file we are acquiring, preprocessing and visualizing the BirdCLEF 2023 competition dataset
- `training.ipynb` - in this Jupyter notebook file we are preparing the dataset to fit the models that will be trained, we also made an initial training to check whether the dataset is well prepared
- `Dockerfile` - we created an initial Dockerfile that will later be used for continerization
- `requirements.txt` - requirements file that will be used to specify the dependencies and required packages
- `data` - directory, where the dataset will be loaded
- `instructions` - directory, images that are used in the description
  
### Dataset Acquistion steps:
- To use the competition's dataset you will need to create (if you do not already have) a Kaggle account.
- In your account Setting tab generate a new token:
  ![image](https://github.com/turi-mate/deepbirding/blob/main/instructions/creating_token.png)
- The token file will be automatically downloaded as `kaggle.json` in your local machine
- Enter the BirdCLEF 2023 competition from your account and accept the terms and rules of the competition to use the dataset at the Rules tab (https://www.kaggle.com/competitions/birdclef-2023/rules)
- Supposed you have already downloaded our proprosed Deepbirding repository, add the recieved `kaggle.json` file to your project root directory
- Now with the Kaggle API key the code is able to retrieve the dataset from Kaggle
- We already wrote the code for the rest of the data acquisition part in the `data_preprocessing.ipynb` file
- After these steps the BirdCLEF 2023 comptetition dataset will be ready to use

### Dataset Visualization and Preprocessing steps:
- For the audio data visualization part we made a diagram that shows the waveform and the mel spectogram of an audio instance 
- For the data preprocessing part we applied some data preparing methods and techniques to balance and fit the provided dataset for the model training:
  - We analyzed the acquired dataset from two views and applied some changes to the dataset:
  - We inspected the length of the provided datafiles, and we decided that we should filter out those audio data instances that are too short (less than 2 seconds) and that are excessively long (more than 1 minute) for a training sample
- We applied these changes to a new metadata file, so the original provided metadata file is also remained intact
- Both of the visualization and preprocessing steps were done at the `data_preprocessing.ipynb` notebook file  

### Data Preparation and Initial Training:
- We made the data preparations for the CNN model in the `training.ipynb` so we can directly fit the dataset into the model that will be later trained
- We also made an initial model training using the Pytorch Lightning platform
