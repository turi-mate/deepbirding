## Deepbirding
### Team members: 
- Bence Bihari (IVXWF8), Lehel Hegyi (GSZLZ7), Máté Turi (OPVP7J)
### Project description: 
This is a university project teamwork for the Deep Learning course. Our task is to take part in the Kaggle BirdCLEF 2023 competition, where we are creating and training a deep learning model that could identify and classify different kinds of bird's chirping. The initial dataset can be found in the official BirdCLEF description.

### Resources used for data preparation:
- Feeding audio datas to CNN: https://www.mdpi.com/2076-3417/11/13/5796?fbclid=IwAR2IWzKZQIj5DcqTbg4VubWJs8CSr0RKhLYVCPOj95zc9-YkkW0WgR8iKsQ
- Converting audio files to Mel Spectograms: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

### Dataset acquistion steps:
- To use the competition's dataset you will need to create (if you do not already have) a Kaggle account.
- In your account Setting tab generate a new token:
- ![image](https://github.com/turi-mate/deepbirding/blob/main/instructions/creating_token.png)
- The token file will be automatically downloaded as 'kaggle.json' in your local machine
- Enter the BirdCLEF 2023 competition from your account and accept the terms and rules of the competition to use the dataset at the Rules tab (https://www.kaggle.com/competitions/birdclef-2023/rules)
- In your downloaded project file add the recieved kaggle.json file to your root project directory
- Now with the kaggle.json API key file the code is able to retrieve the dataset from Kaggle
- We already wrote the code for the rest of the data acquisition part in the data_preprocessing.ipynb file
- After these steps the BirdCLEF 2023 comptetition dataset will be ready to use

### Dataset preprocessing:
- For the data preprocessing part we applied some data preparing methods and techniques to balance and fit the provided dataset for the model training:
- Firstly, we analyzed the acquired dataset from two views and applied some changes to the dataset:
* We inspected the length of the provided datafiles, and we decided that we should filter out those audio data instances that are too short (less than 2 seconds) and that are excessively long (more than 1 minute) for a training sample
* We applied these changes to a new metadata file, so the original provided metadata file is also remained intact 

