from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model1=pickle.load(open('model.pkl','rb'))

cv=pickle.load(open('vec.pkl','rb'))

@app.route('/')
def home():
        return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data =[message]
		vect=cv.transform(data).toarray()
		my_prediction = model1.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	#app.run(host='0.0.0.0',port=8080)
	app.run(debug=True)