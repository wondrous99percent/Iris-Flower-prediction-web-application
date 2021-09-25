from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def home():
    data1= request.form['a']
    data2= request.form['b']
    data3= request.form['c']
    data4= request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    predict = model.predict(arr)

    return render_template('after.html', data=predict)


if __name__ == "__main__":
    app.run(debug=True)