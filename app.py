from flask import Flask,render_template,request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib,re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Vectorizer type:", type(vectorizer))


def textCleaning(text):
    text = text.lower()
    text = re.sub(r'\S*[\\/]\S*',"",text)
    text = re.sub(r'[^a-z\s]',"",text)
    text = re.sub(r'https\S+',"",text)
    text =  re.sub(r'\s+'," ",text)
    return text

def tokenize(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokenize_text = [word for word in tokens if word not in stop_words ]
    return tokenize_text

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    text = request.form['newstext']
    cleanText = textCleaning(text)
    tokenize_text = tokenize(cleanText)
    join_tokenize_text = " ".join(tokenize_text)
    matrix = vectorizer.transform([join_tokenize_text])
    prediction = model.predict(matrix)
    if (prediction == 1):
        result = "World"
    elif (prediction == 2):
        result = "Sport"
    elif (prediction == 3):
        result = "Business"
    elif (prediction == 4):
        result = "Science/Tech"
    
    return render_template("index.html", prediction=result)
 
if __name__ == "__main__":
    app.run(debug=True)