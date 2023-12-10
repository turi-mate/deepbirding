### This is the dataset dictionary that will be populated with the dataset after running the pipeline
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
