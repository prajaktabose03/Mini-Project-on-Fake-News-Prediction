import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import os
import joblib
import pickle
from flask import Flask
import os
import newspaper
from newspaper import Article
import urllib
import nltk
nltk.download('punkt')

#loading flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=Flask(__name__,template_folder='template')
with open(r"C:\Users\Prajakta Bose\OneDrive\Documents\Prajakta Clg\fakenewspredictor\model.pkl" , 'rb') as handle:
    model = pickle.load(handle)
    
@app.route('/')
def main():
    return render_template('index.html')

#receiving the input url from the user and using web scrapping to extract the news content
@app.route('/predict',methods=['GET', 'POST'])
def predictr():
    url=request.get_data(as_text=True)[5:]
    url=urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    #passing the news article to the model and returning whether it is fake or Real
    pred = model.predict([news])
    return render_template('index.html', prediction_text='The News is "{}"'.format(pred[0]))

if __name__=="__main__":
    port=int(os.environ.get('PORT' ,5000))
    app.run(port=port,debug=True,use_reloader=False)