import pandas as pd
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

    single = model.predict(data)
        
    return render_template('index.html', output1=single[0], query1 = inputQuery1, query2 = inputQuery2,query3 = inputQuery3 ,query4 = inputQuery4, query5 = inputQuery5, query6 = inputQuery6)

if __name__ == "__main__":
    app.debug = True
    app.run()