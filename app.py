from flask import Flask, jsonify, request
from flasgger import Swagger, swag_from, LazyString, LazyJSONEncoder
import numpy as np
import pickle, re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info={
        'title': "API Documentation for Deep Learning",
        'version': "1.0.0",
        'description': "Neural Network and LSTM"
    },
    host = request.host
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# parameters for feature extraction
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
# sentiment labels
sentiment = ['negative', 'neutral', 'positive']

# cleansing
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

file = open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/LSTM_files/x_pad_sequences.pickle", "rb")
feature_file_from_lstm = pickle.load(file)
file.close()
model_file_from_lstm = load_model("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/LSTM_files/model_lstm.h5")

file = open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/x_pad_sequences.pickle", "rb")
feature_file_from_rnn = pickle.load(file)
file.close()
model_file_from_rnn = load_model("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/model_rnn.h5")

# lstm text input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis using LSTM",
        "data": {
            "text": original_text,
            "sentiment": get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# lstm file input (change .yml)
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/lstm.yml", methods=['POST'])
@app.route('/lstmfile', methods=['POST'])
def lstmfile():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    pretext = file.read().decode('utf-8')
    text = [cleansing(pretext)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis using LSTM",
        "data": {
            "text": pretext,
            "sentiment": get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# rnn text input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/rnn.yml", methods=['POST'])
@app.route('/rnn', methods=['POST'])
def rnn():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis using RNN",
        "data": {
            "text": original_text,
            "sentiment": get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# rnn file input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/rnn.yml", methods=['POST'])
@app.route('/rnnfile', methods=['POST'])
def rnnfile():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    pretext = file.read().decode('utf-8')
    text = [cleansing(pretext)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis using RNN",
        "data": {
            "text": pretext,
            "sentiment": get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()