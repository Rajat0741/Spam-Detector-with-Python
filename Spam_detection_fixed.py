import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # Changed from LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# **Download Necessary NLTK Data**
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load Dataset
df = pd.read_csv(r"spam_dataset.csv", encoding='latin-1')

# Rename columns for consistency
df = df.rename(columns={"Message": "message", "Category": "label"})

# Preprocess text messages
df['message'] = df['message'].apply(preprocess_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Map labels to binary

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluating the accuracy of Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model and Vectorizer
with open('model.pkl', 'wb') as model_file, open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(model, model_file)
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")

# Loading Model and Vectorizer
with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)
print("Model and vectorizer have been loaded successfully.")


# Test Cases
test_message = input("Enter a message for classification: ")
# Preprocessing Input
preprocessed_input = preprocess_text(test_message)
# Transforming Input
input_vector = vectorizer.transform([preprocessed_input])
# Predicting the classification
prediction = model.predict(input_vector)
output = "Spam" if prediction[0] == 1 else "Not Spam"
print("\nMessage:", test_message)
print(f"The input Message is Classified as: {output}")

