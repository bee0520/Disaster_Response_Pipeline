# Disaster Response Pipeline

### Project Overview
In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The project includes a web app where anyone can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
I use the data set contains real messages that were sent during disaster events and created a machine learning pipeline to categorize these events.  

Below are a few screenshots of the web app.

![image](https://user-images.githubusercontent.com/92233197/144969681-8680a27a-2d08-4b08-a660-92be642793af.png)
![image](https://user-images.githubusercontent.com/92233197/144970058-df81aad8-74c1-4ce1-a047-9640f91c20e0.png)
![image](https://user-images.githubusercontent.com/92233197/144970003-29919e2e-6956-43d8-aa08-1e7944039e30.png)


### Project Components
There are three components in this project.

**ETL Pipeline:** `process_data.py` has a data cleaning pipeline:
1. Loads the messages and categories datasets.
2. Merge the two datasets and cleans the data 
3. Stores the dataset in the SQLite database

**ML Pipeline:** `train_classifier.py` has a machine learning pipeline:
1. Load data from the SQLite database 
2. Build text processing 
3. Build the machine learning pipeline with text processing
4. Tune the model using GridSearchCV 
5. Split the dataset to create training and testing sets and evluate the model
6. Exports the final model as a pickle file

**Flask Web App:** The web app enables users to enter disaster messages and view the categories of the message. It has three plotly visualization

### Instructions:
Run the following commands in the project's root directory to set up database and model.

To run ETL pipeline `data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

To run ML pipeline  `models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run your web app `python run.py`

Go to http://0.0.0.0:3001/
