# PROBLEM

Credit card fraud occurs when someone steals or uses a credit card's information without the cardholder's permission. Losses related to credit card fraud continue to grow, and the impact of these rising costs is felt by all parties within the payment lifecycle: from banks and credit card companies that absorb the costs of such fraud, to consumers who face higher fees or lower credit scores, and to merchants and small businesses that incur chargeback fees.

To combat and prevent fraudulent transactions, credit card companies and financial institutions have implemented various measures. Most modern solutions leverage artificial intelligence (AI) and machine learning (ML).

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
  

# SETTING UP A VIRTUAL ENVIRONMENT

To set up a virtual environment for running the Python scripts, follow these steps:

1. Install Pipenv
First, install Pipenv on your system by running the following command:

pip install pipenv

2. Navigate to the Project Root Folder
Change your directory to the project root folder (i.e., credit_card_fraud_predictor):

3. Install Dependencies
Allow Pipenv to install all the necessary dependencies by executing:

pipenv install

4. Activate the Virtual Environment:
Once the installation is complete, you can activate the virtual environment

pipenv shell

5. Navigate to the Scripts Folder
Change to the scripts folder inside the root folder

6. Train the Model
From the scripts folder, you can train the model by running:

python train.py

The model will be trained with the dataset provided by using the Gradient boosting model and its state will be serialized to and output file in the /model folder

7. Deploy the Model for Predictions:
To deploy the model, use Gunicorn with the following command:

$gunicorn --bind=0.0.0.0:5000 predict:app

This will make the endpoint available for predictions at: http://localhost:5000/predict

8. Testing the Service:
You can test this service using the predict-test.ipynb notebook located in the notebooks folder.


# CREATING AND RUNNING A DOCKER CONTAINER

To create and run a Docker container for the prediction service from the root directory of your project, follow these steps:


1. Ensure Docker is Installed:
Make sure that you have Docker installed on your machine. You can verify your installation by running:

docker --version

2. Navigate to the Project Root Directory
Change your directory to the root folder of the project (i.e., credit_card_fraud_predictor)

3. Build the Docker Image:
Use the following command to build the Docker image using the Dockerfile in the current directory. 

docker build -t creditcardfraud .

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



  
  


