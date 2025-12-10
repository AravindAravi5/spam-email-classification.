import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
)

# Custom CSS Styling
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 45px !important;
            color: #4A90E2;
            font-weight: bold;
        }
        .sub-header {
            color: #0a66c2;
        }
        .card {
            padding: 15px 20px;
            background-color: #ffffff;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        }
        .history-box {
            background: #dff7ff;
            padding: 10px;
            border-radius: 12px;
            font-size: 15px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            padding-top: 20px;
            opacity: 0.7;
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------- SESSION HISTORY -----------------------
if 'history' not in st.session_state:
    st.session_state.history = []


# ----------------------- LOAD MODELS -----------------------
model1 = joblib.load('naive_bayes_model.joblib')
model2 = joblib.load('logistic_regression_model.joblib')
model3 = joblib.load('svm.joblib')
vectorizer = joblib.load('vectorizer.joblib')


def spam_checking(email):
    return vectorizer.transform([email])


def naive_bayes(email):
    x = spam_checking(email)
    pred = model1.predict(x)[0]
    conf = model1.predict_proba(x)[0][1]
    return pred, conf


def logistic_regression(email):
    x = spam_checking(email)
    conf = model2.predict_proba(x)[0][1]
    pred = 1 if conf >= 0.56 else 0
    return pred, conf


def svm_predict(email):
    x = spam_checking(email)
    pred = model3.predict(x)[0]
    score = model3.decision_function(x)[0]
    conf = 1 / (1 + np.exp(-score))
    return pred, conf


# ----------------------- SIDEBAR MENU -----------------------
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "History", "About"],
    index=0
)


# ----------------------- HOME PAGE -----------------------
if menu == "Home":
    st.markdown("<h1 class='main-title'>📧 Spam Email Detection Portal</h1>", unsafe_allow_html=True)
    st.write("Enter the email content below and get prediction from **3 AI Models**.")

    email_input = st.text_area("✍️ Enter Email Content")

    if st.button("🔍 Analyze"):
        if email_input.strip() == "":
            st.warning("Please enter valid email content!")
        else:
            nb_pred, nb_conf = naive_bayes(email_input)
            lr_pred, lr_conf = logistic_regression(email_input)
            svm_pred, svm_conf = svm_predict(email_input)

            spam_votes = nb_pred + lr_pred + svm_pred
            avg_spam_conf = (nb_conf + lr_conf + svm_conf) / 3
            avg_not_conf = (1 - nb_conf + 1 - lr_conf + 1 - svm_conf) / 3

            # RESULT
            if spam_votes >= 2:
                result = "SPAM"
                confidence = avg_spam_conf
                st.error(f"🚨 Final Decision: **SPAM** ({round(confidence*100,2)}%)")
            else:
                result = "NOT SPAM"
                confidence = avg_not_conf
                st.success(f"✅ Final Decision: **NOT SPAM** ({round(confidence*100,2)}%)")

            st.markdown("---")
            st.subheader("📊 Model Predictions")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="card">
                <h4 class="sub-header">🧠 Naive Bayes</h4>
                <b>{'Spam' if nb_pred else 'Not Spam'}</b><br>
                Confidence: {round(nb_conf*100,2)} %
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card">
                <h4 class="sub-header">📈 Log. Regression</h4>
                <b>{'Spam' if lr_pred else 'Not Spam'}</b><br>
                Confidence: {round(lr_conf*100,2)} %
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="card">
                <h4 class="sub-header">📊 SVM</h4>
                <b>{'Spam' if svm_pred else 'Not Spam'}</b><br>
                Confidence: {round(svm_conf*100,2)} %
                </div>
                """, unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "email": email_input,
                "result": result,
                "confidence": round(confidence * 100, 2),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })


# ----------------------- HISTORY PAGE -----------------------
elif menu == "History":
    st.header("📜 Analysis History")
    if len(st.session_state.history) == 0:
        st.info("No history available yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"""
                <div class="history-box">
                <b>{i}. {item['result']} ({item['confidence']}%)</b><br>
                <i>{item['email'][:150]}...</i><br>
                <small>Time: {item['time']}</small>
                </div><br>
            """, unsafe_allow_html=True)


# ----------------------- ABOUT PAGE -----------------------
else:
    st.header("ℹ️ About the Project")
    st.write("""
    This Spam Detection system uses **Machine Learning models**:
    - 🧠 Naive Bayes Classifier  
    - 📈 Logistic Regression  
    - 📊 Support Vector Machine  

    The system performs:
    ✔ Majority Voting  
    ✔ Confidence Score Calculation  
    ✔ Secure Local Inference  
    """)


# ----------------------- FOOTER -----------------------
st.markdown("<div class='footer'>👨‍💻 Developed by <b>Aravind</b> | ML Spam Detection Tool</div>", unsafe_allow_html=True)
