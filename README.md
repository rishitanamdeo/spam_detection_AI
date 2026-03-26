# 📧 AI/ML Spam Detection System

An AI-powered **Spam Detection System** built using **Machine Learning and Natural Language Processing (NLP)**.
This project classifies email or text messages as **Spam** or **Not Spam (Ham)** using a trained ML model and provides an interactive **Streamlit web interface** for testing messages.

---

## 🚀 Project Overview

Spam emails and messages are a major issue in digital communication.
This project uses **Machine Learning techniques** to automatically detect spam messages by analyzing their text content.

The system converts text messages into numerical features using **TF-IDF Vectorization** and trains a **Naive Bayes classifier** to identify spam patterns.

---

## 🧠 Technologies Used

* Python
* Machine Learning
* Natural Language Processing (NLP)
* Streamlit
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn

---

## 📂 Project Structure

```
spam-detection-ml
│
├── spam_ham_dataset.csv
├── spam_detector.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How the Model Works

1. Load the spam dataset
2. Clean and preprocess the text
3. Convert text into numerical features using **TF-IDF**
4. Train the **Multinomial Naive Bayes model**
5. Evaluate the model using accuracy and confusion matrix
6. Use the trained model to classify new messages

---

## 💻 Installation

Clone the repository:

```
git clone https://github.com/yourusername/spam-detection-ml.git
```

Navigate to the project folder:

```
cd spam-detection-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit App

Run the following command:

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 Example Test Messages

### Spam Example

```
Congratulations! You have won a FREE iPhone.
Click here to claim your prize.
```

### Not Spam Example

```
Hi, the meeting has been scheduled for tomorrow at 10 AM.
```

---

## 📊 Model Performance

Typical accuracy using this dataset:

```
Accuracy: ~97%
```

Evaluation metrics include:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 📈 Future Improvements

* Add deep learning models (LSTM / BERT)
* Deploy model online
* Add email integration
* Improve UI dashboard

---

## 👩‍💻 Author

Project developed for **AI & ML BYOP Capstone Project**.

---

