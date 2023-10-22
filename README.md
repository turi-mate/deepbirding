## Deepbirding
### Project members: Bence Bihari (IVXWF8), Lehel Hegyi (GSZLZ7), Máté Turi (...)
### Project description: 
This is a university project teamwork for the Deep Learning course. We are taking part in the BirdCLEF competition, where we are creating a deep learning model that can identify and classify different kinds of bird's chirping. The initial dataset can be found in the official BirdCLEF description

### Resources used for data preparation:
- https://www.mdpi.com/2076-3417/11/13/5796?fbclid=IwAR2IWzKZQIj5DcqTbg4VubWJs8CSr0RKhLYVCPOj95zc9-YkkW0WgR8iKsQ
- https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

### Dataset acquisition:
- One possible solution is to create a user on Kaggle and enter the BirdCLEF 2023 competition
- ![image](https://github.com/turi-mate/deepbirding/assets/78791711/56e2b216-2996-46e4-aaff-5234b393237e)
- Request a Kaggle.json file key that authenticates the user in Google Colab in the following way:
- ![image](https://github.com/turi-mate/deepbirding/assets/78791711/e299d2ad-88b9-489b-93f2-1eb206ae8097)
- After this the data will be rady for use
- With this key and with using the Kaggle api the user can fetch the dataset and use it for the model
- In our case we have attached Google Drive to Colab downloaded the dataset there, this way there is no need for downloading the dataset on every runtime modification of Colab

