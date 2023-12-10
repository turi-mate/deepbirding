## Deepbirding: BirdCLEF2023 
### Team members: 
- Bence Bihari (IVXWF8), Lehel Hegyi (GSZLZ7), (Máté Turi (OPVP7J) left team after Milestone 1)
### Project description: 
This is a university project teamwork for the Deep Learning course. Our task is to take part in the Kaggle BirdCLEF 2023 competition, where we are creating and training a deep learning model that could identify and classify different kinds of east-african bird's calls. The initial dataset can be found in the official BirdCLEF description.

### Project final documentation (in Hungarian):
- Can be read in the `final_documentation.pdf` file

### How to run the pipeline:
- To run the pipeline the `data_preprocessing.ipynb` for the dataset preprocessing,for the dataloading and model training the `training.ipynb`,
  and for the evaluation the `evaluation.ipynb`. The Jupyter notebooks should be run separately.
   
### How to train the models: 
- To train the model we firstly need to make sure that we have the `filtered_metadata.csv` and the `BirdCLEF2023` dataset in our root project folder.
- After that we can run the `training.ipynb` Jupyter notebook to train our proposed custom model.
  
### How to evaluate the models: 
- To evaluate the models you need to have the `filtered_metadata.csv` and the `BirdCLEF2023` dataset in our root project folder, and you will also need the trained model's checkpoint. 
- After that we can run the `evaluation.ipynb` Jupyter notebook to test the custom model.
 
### Files and functions:
- `data_preprocessing.ipynb` - in this Jupyter notebook file we are acquiring, preprocessing and visualizing the BirdCLEF 2023 competition dataset
- `training.ipynb` - in this Jupyter notebook file we are preparing the dataset to fit the models that will be trained, we also made an initial training to check whether the dataset is well prepared
- `evaluation.ipynb` - in this Jupyter notebook file we test the model that we train earlier in the training phase
- `Dockerfile` - we created an initial Dockerfile that will later be used for continerization
- `requirements.txt` - requirements file that will be used to specify the dependencies and required packages
- `data/` - directory, where the dataset will be loaded
- `instructions/` - directory, images that are used in the description
  
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

### Dataset Visualization and Preparation steps:
- The dataset isualization and preprocessing steps were done at the `data_preprocessing.ipynb` notebook file  
- For the audio data visualization part we made a diagram that shows the waveform and the mel spectogram of an audio instance 
- For the data preprocessing part we applied some data preparing methods and techniques to balance and fit the provided dataset for the model training:
  - We analyzed the acquired dataset from two views and applied some changes to the dataset:
  - We inspected the length of the provided datafiles, and we decided that we should filter out those audio data instances that are too short (less than 2 seconds) and that are excessively long (more than 1 minute) for a training sample
- We applied these changes to a new metadata file, so the original provided metadata file is also remained intact
- The filtered metadatas can be found in the `datafiltering_results/filtered_metadata.csv` file

### Data Loading and Training:
- We made the data loading for the CNN model in the `training.ipynb` so we can directly fit the dataset into the model that will be trained
- We also made an initial model training using the Pytorch Lightning platform

### Weights and Biases Access for the best modell file
- The trained modell wheights are located in our team Wandb project as artifacts. Due to size issues we could not upload the modell weights to GitHub.
- We recommend for the best experience to contact us for access to the Wandb project and this way both the Dashboards and the Artifacts can be accessed.
- The modell weights can be found in this link: https://drive.google.com/file/d/1FYlRX_3Qsv4ZG5Q0KDyixZS_sx3ABviC/view?usp=sharing

### Resources used:
- Feeding audio datas to CNN: https://www.mdpi.com/2076-3417/11/13/5796?fbclid=IwAR2IWzKZQIj5DcqTbg4VubWJs8CSr0RKhLYVCPOj95zc9-YkkW0WgR8iKsQ
- Converting audio files to Mel Spectograms: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
