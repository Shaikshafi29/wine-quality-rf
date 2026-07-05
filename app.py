import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
# Load the model
rfmodel=pickle.load(open('rfmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    
    new_data = pd.DataFrame([data])

    prediction = rfmodel.predict(new_data)
    return jsonify({'prediction':prediction.tolist()})

if __name__=="__main__":
    app.run(debug=True)
    
