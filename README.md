## Deepbirding
### Team members: 
- Bence Bihari (IVXWF8), Lehel Hegyi (GSZLZ7), Máté Turi (OPVP7J)
### Project description: 
This is a university project teamwork for the Deep Learning course. We are taking part in the BirdCLEF competition, where we are creating a deep learning model that can identify and classify different kinds of bird's chirping. The initial dataset can be found in the official BirdCLEF description.

### Resources used for data preparation:
- Feeding audio datas to CNN: https://www.mdpi.com/2076-3417/11/13/5796?fbclid=IwAR2IWzKZQIj5DcqTbg4VubWJs8CSr0RKhLYVCPOj95zc9-YkkW0WgR8iKsQ
- Melspectogram: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

### Setting up the enviromnent:
- Download the deepbirding Github repository to your local machine
- To use the competition's dataset you will need to create (or use the existing if you already have) a Kaggle user, and in your account Setting tab generate a new token
![image](https://github.com/turi-mate/deepbirding/blob/main/instructions/creating_token.png)
- The token file will be automatically downloaded as 'kaggle.json' in your local machine
- Enter the BirdCLEF 2023 competition from your account and accept the terms and rules of the competition to use the dataset at the Rules tab
- (https://www.kaggle.com/competitions/birdclef-2023/rules)
- In your downloaded project file add the recieved kaggle.json file to your root project directory
- With the kaggle.json key file the code is able to retrieve the dataset from Kaggle
- After these steps the BirdCLEF2023 comptetition dataset will be ready for use

