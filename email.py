# import streamlit as st
# import joblib
# import numpy as np

# # Load models and vectorizer
# model1 = joblib.load('naive_bayes_model.joblib')
# model2 = joblib.load('logistic_regression_model.joblib')
# model3 = joblib.load('svm.joblib')
# vectorizer = joblib.load('vectorizer.joblib')

# # Prediction functions
# def spam_checking(new_email):
#     return vectorizer.transform([new_email])

# def naive_bayes(new_email):
#     x_new = spam_checking(new_email)
#     prediction = model1.predict(x_new)[0]
#     confidence = model1.predict_proba(x_new)[0][1]
#     return prediction, confidence

# def logistic_regression(new_email):
#     x_new = spam_checking(new_email)
#     confidence = model2.predict_proba(x_new)[0][1]
#     prediction = 1 if confidence >= 0.56 else 0
#     return prediction, confidence

# def svm(new_email):
#     x_new = spam_checking(new_email)
#     prediction = model3.predict(x_new)[0]
#     decision_score = model3.decision_function(x_new)[0]
#     confidence = 1 / (1 + np.exp(-decision_score))  # Sigmoid
#     return prediction, confidence

# # App title
# st.title("📧 Spam Email Detector")
# st.markdown("Check if an email is **Spam** or **Not Spam** using 3 different ML models!")

# # Input box
# new_email = st.text_area("✍️ Enter your email content below:")

# if st.button("Check Spam"):
#     if new_email.strip() == "":
#         st.warning("Please enter some email text to check.")
#     else:
#         nb_pred, nb_conf = naive_bayes(new_email)
#         lr_pred, lr_conf = logistic_regression(new_email)
#         svm_pred, svm_conf = svm(new_email)

#         spam_votes = nb_pred + lr_pred + svm_pred
#         avg_conf = (nb_conf + lr_conf + svm_conf) / 3

#         st.subheader("🔍 Results from Each Model:")
#         st.write(f"🧠 Naive Bayes: {'Spam' if nb_pred else 'Not Spam'} (Confidence: {round(nb_conf*100, 2)}%)")
#         st.write(f"📈 Logistic Regression: {'Spam' if lr_pred else 'Not Spam'} (Confidence: {round(lr_conf*100, 2)}%)")
#         st.write(f"📊 SVM: {'Spam' if svm_pred else 'Not Spam'} (Confidence: {round(svm_conf*100, 2)}%)")

#         st.markdown("---")
#         if spam_votes >= 2:
#             st.success(f"🚨 Final Decision: **SPAM** (Confidence: {round(avg_conf*100, 2)}%)")
#         else:
#             st.info(f"✅ Final Decision: **NOT SPAM** (Confidence: {round((1 - avg_conf)*100, 2)}%)")

import streamlit as st
import joblib
import numpy as np

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

st.markdown("""
<style>
    body { background-color: #f4f7fb; }
    .title {
        font-size: 36px !important;
        font-weight: 800 !important;
        color: #002b80 !important;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        font-size: 16px !important;
        color: #444;
        margin-bottom: 20px;
    }
    .box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 3px 15px rgba(0,0,0,0.07);
        margin-top: 10px;
    }
    .spam {
        background-color: #ffe6e6;
        color: #b30000;
        font-weight: 700;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .ham {
        background-color: #e6ffe6;
        color: #006600;
        font-weight: 700;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Spam Detection"])


# ---------------- Load Models ----------------
model1 = joblib.load('naive_bayes_model.joblib')
model2 = joblib.load('logistic_regression_model.joblib')
model3 = joblib.load('svm.joblib')
vectorizer = joblib.load('vectorizer.joblib')


# ------------- Backend functions ---------------
def spam_checking(new_email):
    return vectorizer.transform([new_email])

def naive_bayes(new_email):
    x_new = spam_checking(new_email)
    pred = model1.predict(x_new)[0]
    conf = model1.predict_proba(x_new)[0][1]
    return pred, conf

def logistic_regression(new_email):
    x_new = spam_checking(new_email)
    conf = model2.predict_proba(x_new)[0][1]
    pred = 1 if conf >= 0.56 else 0
    return pred, conf

def svm(new_email):
    x_new = spam_checking(new_email)
    pred = model3.predict(x_new)[0]
    score = model3.decision_function(x_new)[0]
    conf = 1 / (1 + np.exp(-score))
    return pred, conf


# ---------------- Home Page ----------------
if page == "Home":
    st.markdown("<h1 class='title'>📩 Email Spam Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Detect spam messages using Naive Bayes, Logistic Regression & SVM combined!</p>", 
                unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/646/646094.png", width=250)
    st.info("➡ Switch to **Spam Detection** tab from sidebar to test emails!")


# ------------- Spam Detection Page -------------
if page == "Spam Detection":

    st.markdown("<h2 class='title'>✍️ Email Spam Detector</h2>", unsafe_allow_html=True)

    email = st.text_area("Write email content here:", height=180)

    if st.button("🔍 Check Spam"):
        if email.strip() == "":
            st.warning("Please enter some email content!")
        else:
            with st.spinner("Analyzing email... 🔄"):

                nb_pred, nb_conf = naive_bayes(email)
                lr_pred, lr_conf = logistic_regression(email)
                svm_pred, svm_conf = svm(email)

                votes = nb_pred + lr_pred + svm_pred
                avg_conf = (nb_conf + lr_conf + svm_conf) / 3

            # Display model results in 3 columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("<div class='box'>🧠 Naive Bayes</div>", unsafe_allow_html=True)
                st.write("Prediction:", "Spam 🚨" if nb_pred else "Not Spam ✅")
                st.write("Confidence:", f"{round(nb_conf*100, 2)}%")

            with col2:
                st.markdown("<div class='box'>📈 Logistic Regression</div>", unsafe_allow_html=True)
                st.write("Prediction:", "Spam 🚨" if lr_pred else "Not Spam ✅")
                st.write("Confidence:", f"{round(lr_conf*100, 2)}%")

            with col3:
                st.markdown("<div class='box'>📊 SVM</div>", unsafe_allow_html=True)
                st.write("Prediction:", "Spam 🚨" if svm_pred else "Not Spam ✅")
                st.write("Confidence:", f"{round(svm_conf*100, 2)}%")

            st.markdown("---")

            # Final Voting result
            if votes >= 2:
                st.markdown(f"<div class='spam'>🚨 FINAL RESULT: SPAM ({round(avg_conf*100, 2)}% Confidence)</div>", 
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ham'>✅ FINAL RESULT: NOT SPAM ({round((1-avg_conf)*100, 2)}% Confidence)</div>", 
                            unsafe_allow_html=True)
