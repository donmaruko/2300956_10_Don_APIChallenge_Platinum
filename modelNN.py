# CHAPTER 8 TOPIC 2

# Training Data in NN
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df_train = pd.read_csv('C:/Users/realt/Documents/Binar/For Platinum Project/Data NusaX - Chapter 9/train.csv')
df_valid = pd.read_csv('C:/Users/realt/Documents/Binar/For Platinum Project/Data NusaX - Chapter 9/valid.csv')

df = pd.concat([df_train, df_valid], ignore_index=True)
df_test = pd.read_csv('C:/Users/realt/Documents/Binar/For Platinum Project/Data NusaX - Chapter 9/test.csv')

df = pd.concat([df, df_test], ignore_index=True)

print("head:",df.head())
print("shape:",df.shape)
print("df.label.value_counts()",df.label.value_counts())

def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string
df['text_clean'] = df.text.apply(cleansing)
print(df.head())

# before we perform feature extraction
data_preprocessed = df.text_clean.tolist()
# print(data_preprocessed)

# FEATURE EXTRACTION, pickle is to store the results
count_vect = CountVectorizer()
count_vect.fit(data_preprocessed)
X = count_vect.transform(data_preprocessed)
print("Feature Extraction Completed")
pickle.dump(count_vect, open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/NN_files/feature.p", "wb"))

# split into 80% training data and 20% testing data
classes = df.label
# print(classes)
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2)

# training and storing
model = MLPClassifier(early_stopping=True, validation_fraction=0.1)  # Enable early stopping
model.fit(X_train, y_train)
print("Training Completed")
pickle.dump(model, open("C:/Users/realt/Documents/Python/PYTHON/Binar/Platinum/Chapter9/API/NN_files/model.p", "wb"))

# evaluation with Accuracy, Precision, Recall, and F-1 Score
test = model.predict(X_test)
print("Testing Completed")
print(classification_report(y_test, test))

# cross-validation
kf = KFold(n_splits=5, random_state=42,shuffle=True)
accuracies = []
y = classes

for iteration, data in enumerate(kf.split(X), start=1):
    data_train   = X[data[0]]
    target_train = y[data[0]]
    data_test    = X[data[1]]
    target_test  = y[data[1]]
    
    clf = MLPClassifier()
    clf.fit(data_train, target_train)
    preds = clf.predict(data_test)

    accuracy = accuracy_score(target_test, preds)
    print("Training number", iteration)
    print(classification_report(target_test, preds))
    print("====================================================================")
    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
print()
print()
print()
print("Average accuracy:",average_accuracy)

# prediction
original_text = '''
Aku suka kamu
'''

text = count_vect.transform([cleansing(original_text)])
result = model.predict(text)[0]
print("Sentiment:",result)