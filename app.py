import streamlit as st
import joblib
import re

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f8fc;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e6e6e6;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        background-color: white;
        border: 1px solid #e6e6e6;
    }

    .result-title {
        font-size: 25px;
        font-weight: 700;
    }

    .confidence {
        font-size: 18px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">📩 Spam Message Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered NLP application for detecting spam messages'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Information Card
# -----------------------------
st.markdown(
    """
    <div class="info-card">
        <b>How it works</b><br><br>
        Enter an SMS message and the AI model will analyze its
        text patterns and predict whether it is Spam or Not Spam.
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Message Input
# -----------------------------
st.markdown("### 💬 Enter your message")

message = st.text_area(
    "",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=160
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        cleaned_message = clean_text(message)

        message_vector = vectorizer.transform([cleaned_message])

        prediction = model.predict(message_vector)[0]

        probabilities = model.predict_proba(message_vector)[0]

        classes = model.classes_

        spam_probability = probabilities[
            list(classes).index("spam")
        ]

        confidence = spam_probability if prediction == "spam" else 1 - spam_probability

        confidence_percentage = confidence * 100


        # -----------------------------
        # Result
        # -----------------------------
        if prediction == "spam":

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        🚨 Spam Message
                    </div>

                    <div class="confidence">
                        Confidence: <b>{confidence_percentage:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

            st.write(
                "This message is likely to be spam."
            )

        else:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        ✅ Not Spam
                    </div>

                    <div class="confidence">
                        Confidence: <b>{confidence_percentage:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

            st.write(
                "This message appears to be legitimate."
            )


# -----------------------------
# Model Information
# -----------------------------
st.divider()

st.caption(
    "Built with Python • NLP • TF-IDF • Multinomial Naive Bayes • Streamlit"
)