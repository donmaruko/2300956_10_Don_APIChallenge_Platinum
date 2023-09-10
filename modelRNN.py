# CHAPTER 9 TOPIC 2 - RNN

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, RNN, SpatialDropout1D, SimpleRNN, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from keras.models import load_model
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn import metrics

# Data Preparation
# Load and concatenate data
df_train = pd.read_csv('C:/Users/realt/Documents/Binar/For Platinum Project/Data NusaX - Chapter 9/train.csv')
df_valid = pd.read_csv('C:/Users/realt/Documents/Binar/For Platinum Project/Data NusaX - Chapter 9/valid.csv')

df = pd.concat([df_train, df_valid], ignore_index=True)
df_test = pd.read_csv('C:/Users/realt/Documents/Binar/For Platinum Project/Data NusaX - Chapter 9/test.csv')

df = pd.concat([df, df_test], ignore_index=True)

# Data Exploration
print("head:",df.head())
print("shape:",df.shape)
print("df.label.value_counts()",df.label.value_counts())

# Text Normalization and Cleaning
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

df['text_clean'] = df.text.apply(cleansing)
print("new head:",df.head())

# Separating data by sentiment
neg = df.loc[df['label'] == 'negative'].text_clean.tolist()
neu = df.loc[df['label'] == 'neutral'].text_clean.tolist()
pos = df.loc[df['label'] == 'positive'].text_clean.tolist()
neg_label = df.loc[df['label'] == 'negative'].label.tolist()
neu_label = df.loc[df['label'] == 'neutral'].label.tolist()
pos_label = df.loc[df['label'] == 'positive'].label.tolist()

# Checking the number of data for each sentiment
total_data = pos + neu + neg
labels = pos_label + neu_label + neg_label
print("Pos: %s, Neu: %s, Neg: %s" % (len(pos), len(neu), len(neg)))
print("Total data: %s" % len(total_data))

# Feature Extraction
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(total_data)
with open('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/rnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("rnn_tokenizer.pickle has been created!")

# Tokenize text data
X = tokenizer.texts_to_sequences(total_data)
vocab_size = len(tokenizer.word_index)
maxlen = max(len(x) for x in X)
X = pad_sequences(X)
with open('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/rnn_x_pad_sequences.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("rnn_x_pad_sequences.pickle has been created!")

# Input data labels
Y = pd.get_dummies(labels)
Y = Y.values
with open('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/rnn_y_labels.pickle', 'wb') as handle:
    pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("rnn_y_labels.pickle has created!")

# Data Splitting
file = open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/rnn_x_pad_sequences.pickle",'rb')
X = pickle.load(file)
file.close()
file = open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/rnn_y_labels.pickle",'rb')
Y = pickle.load(file)
file.close()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# adjustments
embed_dim = 100 # 100
units = 18 # 16 going to 20
dropout_rate = 0.8 # 0.7 maybe go up
learning_rate = 0.06 # 0.60 
batch_size = 30 # 30
# standard test : 0.688

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SimpleRNN(units, dropout=dropout_rate, kernel_regularizer=l2(0.0001)))
model.add(Dense(3, activation='softmax'))
sgd = optimizers.Adam(learning_rate = learning_rate)
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

adam = optimizers.Adam(learning_rate = learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

# epochs 10 default
es = EarlyStopping(monitor='val_l oss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, callbacks=[es])

# evaluation with confusion matrix, accuracy, f1, recall, precision
predictions = model.predict(X_test)
y_pred = predictions
matrix_test = metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Testing finished")
print(matrix_test)

# Cross-Validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
accuracies = []
y = Y
embed_dim = 100 # also try 200
units = 32

for iteration, data in enumerate(kf.split(X), start=1):

    data_train   = X[data[0]]
    target_train = y[data[0]]
    data_test    = X[data[1]]
    target_test  = y[data[1]]

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(SimpleRNN(units, dropout=dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    history = model.fit(data_train, target_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0, callbacks=[es])

    predictions = model.predict(data_test)
    y_pred = predictions

    # For the current fold only
    accuracy = accuracy_score(target_test.argmax(axis=1), y_pred.argmax(axis=1))

    print("Training ke-", iteration)
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print("======================================================")
    accuracies.append(accuracy)

# Average accuracy over all folds
average_accuracy = np.mean(accuracies)
print()
print()
print()
print("Rata-rata Accuracy: ", average_accuracy)

# Visualization for Checking Overfitting, Underfitting, or Good Fit
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Save the model
plot_history(history)
model.save('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/model_rnn.h5')
print("Model has been created!")

input_text = '''
Saya rasa bahagia.
'''
sentiment = ['negative', 'neutral', 'positive']

text = [cleansing(input_text)]
predicted = tokenizer.texts_to_sequences(text)
guess = pad_sequences(predicted, maxlen=X.shape[1])

model = load_model('C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/RNN_files/model_rnn.h5')
prediction = model.predict(guess)
threshold = 0.5
polarity = np.argmax(prediction[0])

# Check if the highest probability is below the threshold
if np.max(prediction) < threshold:
    polarity = 1  # Classify as "neutral"
else:
    polarity = np.argmax(prediction[0])  # Otherwise, choose the highest probability class

print("Text: ", text[0])
print("Sentiment: ", sentiment[polarity])