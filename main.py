import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv(r'E:\my project\spam detection\emails.csv')

# Check for null values
print(df.isnull().sum())

# Prepare features and labels
x = df['text']
y = df['spam']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Train the model
model = LogisticRegression()
model.fit(x_train_features, y_train)

# Evaluate the model
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print('Training accuracy:', accuracy_on_training_data)

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print('Test accuracy:', accuracy_on_test_data)

# Print classification report
print(classification_report(y_test, prediction_on_test_data))

# Predict on new input
input_mail = ['best picks hey, best picks are zigo and smtx steve']
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

print('Prediction:', 'Spam' if prediction[0] == 1 else 'Not Spam')

# Save the trained model and vectorizer
with open('spam_detection_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)
