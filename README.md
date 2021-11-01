# Capstone
# NewsArticalClassifier

#### IIITH Final project *( By Swati Surampudi and Oggu Rachana)*

### Problem Statement:
Classify News Articles into categories - With information overload today users are inundated with news articles of all topics, even the ones which may not be relevant to users. 
Design a system which can classify incoming news articles and appropriately tag the corresponding category. Develop a data pipeline which includes the all the 
following stages of Machine Learning Project Life Cycle –
1. Data Ingestion
2. Data Preparation
3. Data segregation & Model Training
4. Model Deployment
5. Model Prediction

### Requirements:
- MongoDB for database
- Apache Zookeeper + Kafka for message streams
- Pyspark for stream processing
- POSTMAN for testing Flask API’s
- MLFLOW for model versioning +hyper-parameters versioning
- Dockers for containarizing 

### 1.Data Ingestion

- Installation of kafka in local system:https://github.com/swathi14-5/capstone/blob/master/week_1/kafka-win-install.pdf
- Commands to run for data Ingestion:
- Used free new API keys from https://mediastack.com/ and https://rapidapi.com/newscatcher-api-newscatcher-api-default/api/free-news/

### 2&3.Data Preparation and Model training:

- Run following command: python data.py
- This will create a pkl file(newsclassifier_model.pkl) and topics.csv file

### 4.Model Deployment:

- Run the docker file for web app view and predection: docker-compose up
### 5.Application output:
![Screenshot (159)](https://user-images.githubusercontent.com/77978493/139678018-1d98d01e-a25e-4cf5-b28a-65b52037633f.png)


All the commands are present in commands file
