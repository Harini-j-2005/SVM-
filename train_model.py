import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Load real dataset
dataset = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

X = dataset.data
y = dataset.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)

# Train Linear SVM
svm = LinearSVC(C=1.0)
svm.fit(X_train_tfidf, y_train)

# Save model
with open("model/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

# Save vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model and TF-IDF Vectorizer saved successfully")
