# Project2: Disaster Response Pipeline 

# Steps:
1. Run the commands in the root directory to set up the database and model

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the directory of the app to run your web app
    `python run.py`
