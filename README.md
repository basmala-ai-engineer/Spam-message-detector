# 📩 Spam Message Detection

An AI-powered Natural Language Processing (NLP) application that detects whether an SMS message is **Spam** or **Not Spam**.

## 🎯 Project Objective

The goal of this project is to automatically classify SMS messages and help users identify potentially unwanted or fraudulent messages.

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* NLP
* TF-IDF Vectorization
* Multinomial Naive Bayes
* Streamlit
* Joblib

## 📊 Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

The dataset contains **5,572 SMS messages** classified into two categories:

* Ham — legitimate messages
* Spam — unwanted messages

## ⚙️ Machine Learning Approach

The project follows these main steps:

1. Load and explore the dataset.
2. Clean and preprocess the text.
3. Convert text into numerical features using **TF-IDF**.
4. Train a baseline Multinomial Naive Bayes classifier.
5. Improve the TF-IDF representation using **unigrams and bigrams**.
6. Evaluate and compare the models.
7. Save the trained model and vectorizer.
8. Deploy the model using Streamlit.

## 📈 Model Performance

The final model achieved:

| Metric         |      Score |
| -------------- | ---------: |
| Accuracy       | **96.05%** |
| Spam Precision |   **100%** |
| Spam Recall    |    **70%** |
| Spam F1-Score  |    **83%** |

The improved model reduced missed spam messages from **53 to 44** compared with the baseline model.

## 🖥️ Application

The Streamlit application allows users to:

* Enter an SMS message.
* Analyze the message using the trained NLP model.
* Predict whether the message is Spam or Not Spam.
* Display the model's prediction confidence.

## 📁 Project Structure

```text
Spam_Message_Detection/
│
├── data/
│   └── smsspamcollection/
│       ├── SMSSpamCollection
│       └── readme
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── spam_detection.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app.py
```

### 3. Open the application

Streamlit will provide a local URL where you can access the application.

## ⚠️ Note

This project is intended for educational and portfolio purposes. The prediction represents the model's classification and should not be considered a guarantee that a message is safe or malicious.
