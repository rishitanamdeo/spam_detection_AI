# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Display first rows
print(df.head())

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Check dataset info
print(df.info())

# Check distribution
print(df['label'].value_counts())

# Convert text labels to numeric if needed
df['label_num'] = df['label'].map({'ham':0, 'spam':1})

# Features and Target
X = df['text']
y = df['label_num']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Test with custom message
def predict_spam(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"

# Example
msg = "Congratulations! You have won a free iPhone. Click here to claim."
print("Message:", msg)
print("Prediction:", predict_spam(msg))