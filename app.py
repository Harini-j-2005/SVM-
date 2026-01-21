from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Class labels (20 Newsgroups)
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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        text_vector = vectorizer.transform([text])
        pred = model.predict(text_vector)[0]
        prediction = categories[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
