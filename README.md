# Sentiment Analysis API with Flask and Swagger

This is a Flask-based API for performing sentiment analysis using pre-trained deep-learning models. It includes endpoints for both text and file input, and it stores the analyzed data in a SQLite database. Swagger is integrated for easy API documentation.

This repo contains:
- the flask app .py and .ymls
- .py files for the LSTM and RNN models
- .csv files used for training
- neural network calculations report
- overall presentation

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Database](#database)
- [Swagger Documentation](#swagger-documentation)
- [License](#license)

## Installation

1. Clone this repository to your local machine:

```shell
git clone <https://github.com/donmaruko/Sentiment-Analysis-API.git>
```

2. Install the required Python packages by running the following command in your project directory:

```shell
pip install -r requirements.txt
```

## Usage

1. Make sure you have completed the installation steps. Before running app.py, please run all the model.py files to generate the pickle files

2. Start the Flask application:

```shell
python app.py
```

This will start the Flask development server, and your API will be accessible at `http://localhost:8000/docs`.

3. You can use the API endpoints to perform sentiment analysis on text input or file input.

## Endpoints

### Text Input Endpoints

- `/lstm` (POST): Perform sentiment analysis using an LSTM model with text input.
- `/rnn` (POST): Perform sentiment analysis using an RNN model with text input.
- `/nn` (POST): Perform sentiment analysis using an MLPClassifier model with text input.

### File Input Endpoints

- `/lstmfile` (POST): Perform sentiment analysis using a LSTM model with file input.
- `/rnnfile` (POST): Perform sentiment analysis using a RNN model with file input.
- `/nnfile` (POST): Perform sentiment analysis using a MLPClassifier model with file input.

### View and Clear Database

- `/view_database` (GET): View the data stored in the SQLite database.
- `/clear_database` (POST): Clear the data stored in the SQLite database.

## Database

The API uses an SQLite database (`data.db`) to store the analyzed data. You can view the stored data using the `/view_database` endpoint and clear the database using the `/clear_database` endpoint.

## Swagger Documentation

Swagger is integrated to provide API documentation. You can access the Swagger UI by visiting `http://localhost:8000/docs/` in your web browser. The Swagger documentation provides detailed information about each API endpoint, including request and response examples.
