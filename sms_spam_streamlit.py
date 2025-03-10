import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# List of spam keywords categorized
SPAM_KEYWORDS = {
    "Financial & Money-Related": [
        "free money", "make money fast", "earn extra cash", "get rich quick", "investment opportunity",
        "no credit check", "financial freedom", "lowest price", "double your income", "risk-free",
        "cash bonus", "work from home", "income opportunity", "credit card offer", "unlimited income", "congratulations you won"
    ],
    "Urgency & Pressure Tactics": [
        "act now", "urgent", "limited time offer", "don’t miss out", "only for today", "this won’t last",
        "final notice", "exclusive deal"
    ],
    "Loan & Debt-Related": [
        "debt consolidation", "no hidden fees", "mortgage rates", "instant approval", "pre-approved",
        "lower your bills", "eliminate debt", "no down payment", "reduce your mortgage"
    ],
    "Free & Giveaway": [
        "free trial", "free access", "win big", "free consultation", "get it now", "you have won",
        "claim your prize", "gift card", "complimentary", "100% free"
    ],
    "Health & Medicine Spam": [
        "miracle cure", "no prescription required", "lose weight fast", "guaranteed results", "anti-aging",
        "burn fat", "fast metabolism booster", "viagra", "male enhancement"
    ],
    "Email & Phishing Scams": [
        "verify your account", "click here", "password reset", "urgent update", "security alert",
        "suspicious login attempt", "dear customer", "your account has been suspended", "login now"
    ],
    "Marketing & Promotional": [
        "buy now", "increase sales", "best price", "special promotion", "order now", "satisfaction guaranteed",
        "call now", "act immediately"
    ],
    "Gambling & Betting": [
        "online casino", "big jackpot", "win cash", "betting tips", "get lucky"
    ],
    "Adult Content & Dating": [
        "xxx", "hot singles", "sexy", "adult entertainment", "sex", "nudity", "nude", "meet now", "no strings attached"
    ],
    "IT & Tech-Related Spam": [
        "software update required", "tech support alert", "virus detected", "your device is infected", "access now"
    ]
}

# Function to check for spam keywords
def contains_spam_keywords(text):
    text = text.lower()
    for category, keywords in SPAM_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return True
    return False

# Function to preprocess text
def preprocess_text(text):
    """Cleans and tokenizes text for model prediction."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = text.split()
    return ' '.join(words)

# Load the model
db_path = "SMS_Spam_Detection/spam_classifier.pkl"
with open(db_path, "rb") as f:
    model_data = pickle.load(f)
    vectorizer = model_data[0]
    classifiers = model_data[1]

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message to classify it as Spam or Ham.")

message = st.text_area("Message", "")
selected_model = st.selectbox("Choose a model", list(classifiers.keys()))

if st.button("Classify"):
    if message.strip():
        # Check for spam keywords
        if contains_spam_keywords(message):
            result = "🚨 Spam (Keyword Matched)"
        else:
            processed_message = preprocess_text(message)
            transformed_message = vectorizer.transform([processed_message])
            model = classifiers[selected_model]
            prediction = model.predict(transformed_message)[0]
            result = "🚨 Spam" if prediction == 1 else "✅ Ham"
        
        st.subheader(f"**Prediction: {result}**")
    else:
        st.warning("⚠️ Please enter a message before classifying.")