## Deepbirding
### Project members: Bence Bihari (IVXWF8), Lehel Hegyi (GSZLZ7), Máté Turi (OPVP7J)
### Project description: 
This is a university project teamwork for the Deep Learning course. We are taking part in the BirdCLEF competition, where we are creating a deep learning model that can identify and classify different kinds of bird's chirping. The initial dataset can be found in the official BirdCLEF description

### Resources used for data preparation:
- Feeding audio datas to CNN: https://www.mdpi.com/2076-3417/11/13/5796?fbclid=IwAR2IWzKZQIj5DcqTbg4VubWJs8CSr0RKhLYVCPOj95zc9-YkkW0WgR8iKsQ
- Melspectogram: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

### Dataset acquisition/Environment Setup (Google Colab):
- One possible solution is to create a user on Kaggle and enter the BirdCLEF 2023 competition (https://www.kaggle.com/competitions/birdclef-2023)
- ![image](https://github.com/turi-mate/deepbirding/assets/78791711/56e2b216-2996-46e4-aaff-5234b393237e)
  
- Add the recieved kaggle.json file upload the file to Google Colab:
- ![image](https://github.com/turi-mate/deepbirding/assets/78791711/a1d35741-522b-4642-b149-de3df4f6b5f7)
  
- The following few lines give the user the neccesary permissions to successfully retrieve the dataset from Kaggle:
- ![image](https://github.com/turi-mate/deepbirding/assets/78791711/ab07ec24-35ae-486f-92ce-60357bdd5ebb)
  
- With this key the code is able to retrieve the dataset from Kaggle:
- ![image](https://github.com/turi-mate/deepbirding/assets/78791711/e299d2ad-88b9-489b-93f2-1eb206ae8097)
  
- After this the data will be rady for use
- In our case we have attached Google Drive to Colab downloaded the dataset there, this way there is no need for downloading the dataset on every runtime modification of Colab

