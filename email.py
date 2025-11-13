import streamlit as st
import joblib
import numpy as np

# Load models and vectorizer
model1 = joblib.load('naive_bayes_model.joblib')
model2 = joblib.load('logistic_regression_model.joblib')
model3 = joblib.load('svm.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Prediction functions
def spam_checking(new_email):
    return vectorizer.transform([new_email])

def naive_bayes(new_email):
    x_new = spam_checking(new_email)
    prediction = model1.predict(x_new)[0]
    confidence = model1.predict_proba(x_new)[0][1]
    return prediction, confidence

def logistic_regression(new_email):
    x_new = spam_checking(new_email)
    confidence = model2.predict_proba(x_new)[0][1]
    prediction = 1 if confidence >= 0.56 else 0
    return prediction, confidence

def svm(new_email):
    x_new = spam_checking(new_email)
    prediction = model3.predict(x_new)[0]
    decision_score = model3.decision_function(x_new)[0]
    confidence = 1 / (1 + np.exp(-decision_score))  # Sigmoid
    return prediction, confidence

# App title
st.title("📧 Spam Email Detector")
st.markdown("Check if an email is **Spam** or **Not Spam** using 3 different ML models!")

# Input box
new_email = st.text_area("✍️ Enter your email content below:")

if st.button("Check Spam"):
    if new_email.strip() == "":
        st.warning("Please enter some email text to check.")
    else:
        nb_pred, nb_conf = naive_bayes(new_email)
        lr_pred, lr_conf = logistic_regression(new_email)
        svm_pred, svm_conf = svm(new_email)

        spam_votes = nb_pred + lr_pred + svm_pred
        avg_conf = (nb_conf + lr_conf + svm_conf) / 3

        st.subheader("🔍 Results from Each Model:")
        st.write(f"🧠 Naive Bayes: {'Spam' if nb_pred else 'Not Spam'} (Confidence: {round(nb_conf*100, 2)}%)")
        st.write(f"📈 Logistic Regression: {'Spam' if lr_pred else 'Not Spam'} (Confidence: {round(lr_conf*100, 2)}%)")
        st.write(f"📊 SVM: {'Spam' if svm_pred else 'Not Spam'} (Confidence: {round(svm_conf*100, 2)}%)")

        st.markdown("---")
        if spam_votes >= 2:
            st.success(f"🚨 Final Decision: **SPAM** (Confidence: {round(avg_conf*100, 2)}%)")
        else:
            st.info(f"✅ Final Decision: **NOT SPAM** (Confidence: {round((1 - avg_conf)*100, 2)}%)")
