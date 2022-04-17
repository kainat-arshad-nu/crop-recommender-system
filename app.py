import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")



q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
    
    inputQuery1 = request.form['n']
    inputQuery2 = request.form['p']
    inputQuery3 = request.form['k']
    inputQuery4 = request.form['temp']
    inputQuery5 = request.form['hum']
    inputQuery6 = request.form['ph']

    model = pickle.load(open("trained_model.sav", "rb"))
    
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6]]
    new_df = pd.DataFrame(data, columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph'])
    
    single = model.predict(new_df)
    probablity = model.predict_proba(new_df)[:,1]
    
    o1 = single
    o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'], query6 = request.form['query6'])
    
app.run()