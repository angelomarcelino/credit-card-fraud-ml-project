# PROBLEM

Credit card fraud occurs when someone steals or uses a credit card's information without the cardholder's permission. To combat and prevent fraudulent transactions, credit card companies and financial institutions have implemented various measures. Most modern solutions leverage artificial intelligence (AI) and machine learning (ML).

The purpose of this project is to emulate a service used by one of these institutions to predict whether a purchase is fraudulent. The service receives as input all the information about a purchase made with a credit card by a client and returns as output the probability that the purchase is fraudulent, as well as a recommendation on whether it should be flagged as fraud. The response from this service can be used to prevent customers from being charged for items they did not purchase


# PROYECT FILES AND THEIR FUNCTIONS

Root-level files:

- Dockerfile: Contains instructions to build a Docker image for the project.
- Pipfile: Specifies the packages required for the project.
- Pipfile.lock: Locks the specific versions of dependencies used to ensure reproducibility.

Files inside folders:

- dataset/: This folder contains the training datasets split into multiple CSV files (creditcard_part_1.csv, creditcard_part_2.csv, creditcard_part_3.csv, and creditcard_part_4.csv). This structure allows for upload of all the data in smaller files (the original csv was downloaded from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- model/: This directory holds the trained XGBoost model, saved as "xgboost_eta=0.3_depth=3_minchild=1_round=100.bin". The filename includes hyperparameters used during training for clarity.

- notebooks/: This folder contains several Jupyter notebooks used for various purposes:
 - notebook.ipynb: The primary notebook for data exploration and initial analysis.
 - split-data.ipynb: A notebook for preprocessing and splitting the original CSV file into multiple parts.
 - train.ipynb: A notebook that serves as a base for generating the script to train the model.
 - predict.ipynb: A notebook that serves as a base for generating predictions using the trained model.
 - predict-test.ipynb: A notebook used to test the prediction web service after it has been deployed locally.
 
- scripts/: This directory contains Python scripts for model training and prediction:
  - predict.py: A script that implements the prediction logic using the trained model.
  - train.py: A script dedicated to training the model using the available dataset.
  
# TRAINING AND USING THE MODEL FOR PREDICTIONS

In order to train the model and make predictions, after cloning this repository to your local machine, you can set up a virtual environment as explained below.cd te

## SETTING UP A VIRTUAL ENVIRONMENT

To set up a virtual environment for running the Python scripts, follow these steps:

1. Install Pipenv
First, install Pipenv on your system by running the following command:

pip install pipenv

Notice that sometimes you can get a warning if pipenv is installed in a folder that is not in your path. Add that folder to your path after receiving this warning.

2. Navigate to the Project Root Folder
Change your directory to the project root folder (i.e., credit_card_fraud_predictor):

3. Install Dependencies
Allow Pipenv to install all the necessary dependencies by executing:

pipenv install

4. Activate the Virtual Environment:
Once the installation is complete, you can activate the virtual environment

pipenv shell

5. Navigate to the Scripts Folder
Change to the scripts folder inside the root folder, so you can execute the scripts from there



## TRAINING THE MODEL

From the scripts folder, you can train the model by running:

python train.py

The model will be trained with the dataset provided by using the Gradient boosting model and its state will be serialized to and output file in the /model folder

## USING PREDICTIONS

1. Deploy the model
From the scripts folder, you can deploy the model with the following command:

gunicorn --bind=0.0.0.0:5000 predict:app

This will make the endpoint available for predictions at: http://localhost:5000/predict

2. Getting predictions
You can get predictions from this service using the predict-test.ipynb notebook located in the notebooks folder or you can use curl like this:

curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "time": 0,
    "v1": -1.3598071336738,
    "v2": -0.07278117330985,
    "v3": 2.53634673796914,
    "v4": 1.37815522427443,
    "v5": -0.338320769942518,
    "v6": 0.462387777762292,
    "v7": 0.239598554061257,
    "v8": 0.098697901261051,
    "v9": 0.363786969611213,
    "v10": 0.090794171978932,
    "v11": -0.551599533260813,
    "v12": -0.617800855762348,
    "v13": -0.991389847235408,
    "v14": -0.311169353699879,
    "v15": 1.46817697209427,
    "v16": -0.470400525259478,
    "v17": 0.207971241929242,
    "v18": 0.025790580198559,
    "v19": 0.403992960255733,
    "v20": 0.251412098239705,
    "v21": -0.018306777944153,
    "v22": 0.277837575558899,
    "v23": -0.110473910188767,
    "v24": 0.066928074914673,
    "v25": 0.128539358273528,
    "v26": -0.189114843888824,
    "v27": 0.133558376740387,
    "v28": -0.021053053453822,
    "amount": 149.62
}'

(You can exit the virtual environment once you have run the scripts by typing "exit")

# CREATING AND RUNNING A DOCKER CONTAINER

To create and run a Docker container for the prediction service from the root directory of your project, follow these steps:


1. Ensure Docker is Installed:
Make sure that you have Docker installed on your machine. You can verify your installation by running:

docker --version

2. Navigate to the Project Root Directory
Change your directory to the root folder of the project (i.e., credit_card_fraud_predictor). Notice that you don't need to be inside the virtual environment to run the following commands

3. Build the Docker Image:
Use the following command to build the Docker image using the Dockerfile in the current directory (notice the dot at the end of the command)

docker build -t creditcardfraud .

If you get a permission error while trying to build the image, use sudo to overcome the problem or add your user to the docker group
In case the container can't resolves names, you can modify the configuration file, setting there the dns to be used. For example:

{
  "dns": ["8.8.4.4", "8.8.8.8"],
  "ipv6": false
}


4. Run the Docker Container:
Once the image is built, you can run the Docker container with the following command. This will bind the container’s port 5000 to the host’s port 5000:


docker run -p 5000:5000 creditcardfraud

5. Access the Prediction Service:
After the container is running, the prediction service will be available at the following endpoint:
http://localhost:5000/predict

You can use the aforementioned notebook predict-test.ipynb to test the prediction service.


# DEPLOYMENT TO AWS ELASTIC BEANSTALK


1. 1. Install the AWS CLI
The instructions will depend on your system, for Debian-based linux use:
sudo apt-get install awscli

2. Verify AWS CLI Installation

aws --version

3. Configure AWS CLI
Configure the AWS CLI with your AWS Access Key, Secret Key, and default region

4. Install the AWS Elastic Beanstalk Command Line Interface (EB CLI)
If you haven't installed the EB CLI, you can do so using pip:

pip install awsebcli

5. Create a New Elastic Beanstalk Application
Navigate to the project root directory
Initialize a new Elastic Beanstalk application:

eb init

Follow the prompts, and when asked, "It appears you are using Docker. Is this correct?", answer "yes".

6. Create an Environment and Deploy Your Application

eb create

Follow the prompt and when prompted with "Would you like to enable Spot Fleet requests for this environment?", answer "yes"

This command will build your Docker image, create necessary resources, and deploy the application.

7. Get the URL for the Application
After deployment, retrieve the URL for the application by running:

eb open

This command will open a web browser with the base URL of the deployed application. Append /predict to the base URL to access the endpoint, which you can use in the predict-test.ipynb notebook to test the service.



  
  


