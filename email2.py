import streamlit as st
import joblib
import numpy as np

# Initialize history
if 'history' not in st.session_state:
    st.session_state.history = []

# Load models
model1 = joblib.load('naive_bayes_model.joblib')
model2 = joblib.load('logistic_regression_model.joblib')
model3 = joblib.load('svm.joblib')
vectorizer = joblib.load('vectorizer.joblib')


# Vectorize input email
def spam_checking(new_email):
    return vectorizer.transform([new_email])


# Naive Bayes Model
def naive_bayes(new_email):
    x_new = spam_checking(new_email)
    prediction = model1.predict(x_new)[0]
    confidence = model1.predict_proba(x_new)[0][1]   # Spam probability
    return prediction, confidence


# Logistic Regression Model
def logistic_regression(new_email):
    x_new = spam_checking(new_email)
    spam_conf = model2.predict_proba(x_new)[0][1]
    prediction = 1 if spam_conf >= 0.56 else 0
    return prediction, spam_conf


# SVM Model
def svm(new_email):
    x_new = spam_checking(new_email)
    prediction = model3.predict(x_new)[0]
    
    # Convert decision score to probability using sigmoid
    decision_score = model3.decision_function(x_new)[0]
    spam_conf = 1 / (1 + np.exp(-decision_score))
    
    return prediction, spam_conf


# UI Header
st.title("📧 Spam Email Detector")
st.markdown("Check if an email is **Spam** or **Not Spam** using 3 ML models!")


# Email Input Box
new_email = st.text_area("✍️ Enter your email content below:")


# Main Button
if st.button("Check Spam"):
    if new_email.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Get predictions
        nb_pred, nb_conf = naive_bayes(new_email)
        lr_pred, lr_conf = logistic_regression(new_email)
        svm_pred, svm_conf = svm(new_email)

        # Majority vote
        spam_votes = nb_pred + lr_pred + svm_pred

        # Average spam confidence
        avg_spam_conf = (nb_conf + lr_conf + svm_conf) / 3

        # Display model-wise results
        st.subheader("🔍 Model-wise Results:")
        st.write(f"🧠 Naive Bayes: {'Spam' if nb_pred else 'Not Spam'} (Confidence: {round(nb_conf*100,2)}%)")
        st.write(f"📈 Logistic Regression: {'Spam' if lr_pred else 'Not Spam'} (Confidence: {round(lr_conf*100,2)}%)")
        st.write(f"📊 SVM: {'Spam' if svm_pred else 'Not Spam'} (Confidence: {round(svm_conf*100,2)}%)")

        st.markdown("---")

        # Compute correct confidence for NOT SPAM
        avg_not_spam_conf = ( (1-nb_conf) + (1-lr_conf) + (1-svm_conf) ) / 3

        # Final decision
        if spam_votes >= 2:
            final_result = "SPAM"
            final_conf = avg_spam_conf
            st.error(f"🚨 Final Decision: **SPAM** (Confidence: {round(final_conf*100,2)}%)")
        else:
            final_result = "NOT SPAM"
            final_conf = avg_not_spam_conf
            st.success(f"✅ Final Decision: **NOT SPAM** (Confidence: {round(final_conf*100,2)}%)")

        # Save history
        st.session_state.history.append({
            "email": new_email,
            "result": final_result,
            "confidence": round(final_conf * 100, 2)
        })


# History Button
if st.button("📜 View Spam History"):
    st.subheader("📜 Previous Checks:")
    
    if len(st.session_state.history) == 0:
        st.info("No history yet. Try checking some emails.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**{idx}.** _{item['email']}_ → **{item['result']}** ({item['confidence']}%)")


# Footer
st.markdown("---")
st.markdown("👨‍💻 Created by **Aravind**")
