import os
from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidfvect = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

def predict(text):
    t = re.sub('[^a-zA-Z]', ' ', text)
    t = t.lower()
    t = t.split()
    t = [ps.stem(word) for word in t if not word in stopwords.words('english')]
    t = ' '.join(t)
    t = tfidfvect.transform([t]).toarray()
    prediction = 'HAM' if model.predict(t) == 0 else 'SPAM'
    return prediction

@app.route('/predict/', methods=['post'])
def predictions():
    text = request.form['text']
    prediction = predict(text)
    return render_template('predict.html', text=text, result=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))