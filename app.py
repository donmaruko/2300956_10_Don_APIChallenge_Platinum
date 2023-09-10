from flask import Flask, jsonify, request
from flasgger import Swagger, swag_from, LazyString, LazyJSONEncoder
import numpy as np
import pickle, re
import sqlite3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

DATABASE = 'C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/data.db'

def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_table()

swagger_template = dict(
    info={
        'title': "API Documentation for Deep Learning",
        'version': "1.0.0",
        'description': "Neural Network and LSTM"
    }
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

file = open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/LSTM_files/lstm_x_pad_sequences.pickle", "rb")
feature_file_from_lstm = pickle.load(file)
file.close()
model_file_from_lstm = load_model("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/LSTM_files/model_lstm.h5")

file = open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/rnn_x_pad_sequences.pickle", "rb")
feature_file_from_rnn = pickle.load(file)
file.close()
model_file_from_rnn = load_model("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/model_rnn.h5")

count_vect = pickle.load(open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/NN_files/feature.p", "rb"))
model = pickle.load(open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/NN_files/model.p", "rb"))

# lstm text input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    try:
        original_text = request.form.get('text')
        text = cleansing(original_text)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sentiment_data (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
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
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

# lstm file input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/lstmfile.yml", methods=['POST'])
@app.route('/lstmfile', methods=['POST'])
def lstmfile():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        pretext = file.read().decode('utf-8')
        text = cleansing(pretext)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sentiment_data (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
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
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

# rnn text input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/rnn.yml", methods=['POST'])
@app.route('/rnn', methods=['POST'])
def rnn():
    try:
        original_text = request.form.get('text')
        text = cleansing(original_text)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_file_from_rnn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sentiment_data (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
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
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500
    
# rnn file input
@swag_from("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/rnnfile.yml", methods=['POST'])
@app.route('/rnnfile', methods=['POST'])
def rnnfile():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        pretext = file.read().decode('utf-8')
        text = cleansing(pretext)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_file_from_rnn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sentiment_data (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
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
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

@swag_from('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/nn.yml', methods=['POST'])
@app.route('/nn', methods=['POST'])
def nn():
    try:
        # Get the original text from the request
        original_text = request.form.get('text')
        # Clean and preprocess the text data
        text = cleansing(original_text)
        # Transform the text data using the CountVectorizer and convert to dense array
        text_features = count_vect.transform([text]).toarray()
        # Make predictions using the pre-trained neural network model
        prediction = model.predict(text_features)[0]
        get_sentiment = sentiment[np.argmax(prediction)]
        # Store the result in the database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sentiment_data (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
        # Prepare the response JSON
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using Neural Network",
            "data": {
                "text": original_text,
                "sentiment": get_sentiment
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

@swag_from('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/nnfile.yml', methods=['POST'])
@app.route('/nnfile', methods=['POST'])
def nnfile():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        pretext = file.read().decode('utf-8')
        text = cleansing(pretext)
        text = count_vect.transform([text]).toarray()
        prediction = model.predict(text)[0]
        get_sentiment = sentiment[np.argmax(prediction)]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sentiment_data (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
        json_response = {
                "status_code": 200,
                "description": "Result of Sentiment Analysis using Neural Network",
                "data": {
                    "text": pretext,
                    "sentiment": get_sentiment
                },
            }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

@swag_from('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/view_database.yml', methods=['GET'])
@app.route('/view_database', methods=['GET'])
def view_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT text, sentiment FROM sentiment_data")
    data = cursor.fetchall()
    conn.close()
    # convert data to list of dictionaries
    database_data = [{'text': row[0], 'sentiment': row[1]} for row in data]
    return jsonify({'data': database_data})

@swag_from('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/clear_database.yml', methods=['POST'])
@app.route('/clear_database', methods=['POST'])
def clear_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sentiment_data")
    conn.commit()
    conn.close()
    return jsonify({'message': 'Database cleared successfully'})

if __name__ == '__main__':
    app.run()