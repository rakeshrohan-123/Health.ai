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
model4=joblib.load('cancer_model')


dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/',methods=['GET','POST'])
def first():
    return render_template('homepage.html')


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
    gen1=   request.form['gen1']
    age1 = request.form['age1']
    hei1=  float(request.form['hei1'])
    wei1= float( request.form['wei1'])
    dur1=   float(request.form['dur1'])
    hra1=  float(request.form['hra1'])
    temp1= float(request.form['temp1'])
    
    predi = model3.predict(np.array([[gen1,age1,hei1,wei1,dur1,hra1,temp1]]))
    return render_template('calories.html',result = predi[0])


@app.route('/cancer',methods=['GET','POST'])
def can():
    return render_template("cancer.html")

@app.route('/result', methods=['POST'])
def canresult():
    a = float(request.form['Radius_mean'])
    b = float(request.form['Texture_mean'])
    c = float(request.form['Perimeter_mean'])
    d = float(request.form['Area_mean'])
    e =float(request.form['Smoothness_mean'])
    f = float(request.form['Compactness_mean'])
    g = float(request.form['Concavity_mean'])
    h = float(request.form['concave points_mean'])
    i =float(request.form['symmetry_mean'])
    j = float(request.form['fractal_dimension_mean'])
    k = float(request.form['radius_se'])
    l = float(request.form['texture_se'])
    m = float(request.form['perimeter_se'])
    n = float(request.form['area_se'])
    o = float(request.form['smoothness_se'])
    p = float(request.form['compactness_se'])
    q =float(request.form['concavity_se'])
    r = float(request.form['concave points_se'])
    s = float(request.form['symmetry_se'])
    t = float(request.form['fractal_dimension_se'])
    u = float(request.form['radius_worst'])
    v = float(request.form['texture_worst'])
    w = float(request.form['perimeter_worst'])
    x = float(request.form['area_worst'])
    y = float(request.form['smoothness_worst'])
    z = float(request.form['compactness_worst'])
    A =float(request.form['concavity_worst'])
    B =float(request.form['concave points_worst'])
    C =float(request.form['symmetry_worst'])
    D =float(request.form['fractal_dimension_worst'])
    pred = model4.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D]])
    if pred[0] == 1:
        predict = "You have Maligant cancer, please consult a Doctor."
    elif predi[0] == 0:
        predict = "You don't have cancer and you are Benign"
    output=predict
    
    return render_template("cancer.html", result = output)



app.run()
