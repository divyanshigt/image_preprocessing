import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#second import sklearn dependencies
import joblib

import string
def preprocess_text(sentences):
    english_stopwords = stopwords.words("english")
    punctuations = string.punctuation
    cleaned_documents = []

    for doc in sentences:
        raw_text = doc.lower()
        print("After lowercase:", raw_text)

        tokens = word_tokenize(raw_text)
        print("Tokens:", tokens)

        filtered_tokens = []
        for word in tokens:
            if word not in english_stopwords:
                filtered_tokens.append(word)

        print("Filtered Tokens:", filtered_tokens)

        clean_tokens = [word for word in filtered_tokens if word not in punctuations]
        print("After removing punctuations:", clean_tokens)

        wnet = WordNetLemmatizer()
        lemmatized_words = []
        for word in clean_tokens:
            lemmatized_words.append(wnet.lemmatize(word, "v"))

        print("After Lemmatization:", lemmatized_words)

        final_tokens = []
        for word in lemmatized_words:
            if word.isalpha():
                final_tokens.append(word)

        print("Final Tokens:", final_tokens)

        cleaned_text = " ".join(final_tokens)
        print("Cleaned Text:", cleaned_text)

        cleaned_documents.append(cleaned_text)
        print("=" * 50)

    return cleaned_documents

def predict_intent(user_input):
    vectorizer = joblib.load("tfidf.pkl")
    logistic = joblib.load("intent_clf_model.pkl")
    processed = preprocess_text([user_input])
    user_vector = vectorizer.transform(processed)
    prediction = logistic.predict(user_vector)
    return prediction[0]