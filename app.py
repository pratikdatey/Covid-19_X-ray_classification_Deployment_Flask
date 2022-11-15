from flask import Flask,render_template,request,url_for
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt


app=Flask(__name__)

model=tf.keras.models.load_model('COVID-19-XRAY.h5')

@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=="POST":
        submit = request.files['xray']
        if submit.filename != '':
            submit.save(submit.filename)

        img=cv2.imread(submit.filename, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(500,500))
        plt.imshow(img)
        img=np.expand_dims(img,0)
        output=model.predict(img)

    if output==0:
        return render_template('index1.html',image=submit.filename)
    else:
        return render_template('index2.html')


if __name__=='__main__':
    app.run(debug=True)
