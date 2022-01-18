import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = joblib.load("covid19_model.sav")
model2 = pickle.load(open('heartmodel.pkl', 'rb'))
model3=joblib.load("calories_model.sav")


dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/',methods=['GET','POST'])
def first():
    return render_template('index.html')


@app.route('/home',methods=['GET','POST'])
def home():
    return render_template('diabatics.html')

@app.route('/predict1',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('diabatics.html', prediction_text='{}'.format(output))

@app.route('/covid',methods=['GET','POST'])
def home1():
    return render_template("covid.html")

@app.route('/predict', methods=['POST'])
def result():
    s_length = request.form['sepal_length']
    s_width = request.form['sepal_width']

    
    pred = model1.predict([[s_length, s_width]])
    
    
    return render_template("covid.html", result = pred[0])

@app.route('/caronary',methods=['GET','POST'])
def home2():
    return render_template('caronary.html')

@app.route('/results',methods=['POST'])
def prediction():
    age=    request.form['age']
    tchol = request.form['tchol']
    sysbp=  request.form['sysbp']
    diabp=  request.form['diabp']
    bmi=    request.form['bmi']
    hrate=  request.form['hrate']
    glucose=request.form['glucose']
    prediction = model2.predict(np.array([[age,tchol,sysbp,diabp,bmi,hrate,glucose]]))
    if prediction == 1:
        pred = "You have coronary heart disease, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have coronary heart disease"
    output = pred
    return render_template('caronary.html',result = output)

@app.route('/calories',methods=['GET','POST'])
def home3():
    return render_template('calories.html')

@app.route('/predict2',methods=['POST'])
def prediction1():
    gen1=    request.form['gen1']
    age1 = request.form['age1']
    hei1=  request.form['hei1']
    wei1=  request.form['wei1']
    dur1=    request.form['dur1']
    hra1=  request.form['hra1']
    temp1= request.form['temp1']
    h1=float(hei1)
    w1=float(wei1)
    d1=float(dur1)
    h1=float(hra1)
    t1=float(temp1)
    
    predi = model3.predict(np.array([[gen1,age1,h1,w1,d1,h1,t1]]))
    return render_template('calories.html',result = predi[0])




app.run()
