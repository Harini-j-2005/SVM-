from flask import Flask, render_template, request
import pickle
import os

# ---------- Flask App Initialization ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ---------- Load Model & Vectorizer ----------
with open(os.path.join(BASE_DIR, "model", "svm_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# ---------- Class Labels ----------
categories = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
    'comp.windows.x', 'misc.forsale', 'rec.autos',
    'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
    'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
    'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc',
    'talk.religion.misc'
]

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        text_vector = vectorizer.transform([text])
        pred = model.predict(text_vector)[0]
        prediction = categories[pred]

    return render_template("index.html", prediction=prediction)

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)
