import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📧 AI Spam Detection App")
st.write("This app detects whether a message is Spam or Not Spam using Machine Learning.")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spam_ham_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df['label_num'] = df['label'].map({'ham':0, 'spam':1})
    return df

df = load_data()

# Features and Labels
X = df['text']
y = df['label_num']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# User Input
message = st.text_area("Enter a message to check if it is Spam:")

# Prediction Button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT Spam.")