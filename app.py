from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)
model = joblib.load('model.lb') 

encoders = {}
category_maps = {
    'gender': ['male', 'female'],
    'smoker': ['yes', 'no'],
    'region': ['northeast', 'northwest', 'southeast', 'southwest']
}

for col, classes in category_maps.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes) 
    encoders[col] = le

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Encode categorical variables
        encoded_gender = encoders['gender'].transform([gender])[0]
        encoded_smoker = encoders['smoker'].transform([smoker])[0]
        encoded_region = encoders['region'].transform([region])[0]

        input_data = np.array([[age, encoded_gender, bmi, children, encoded_smoker, encoded_region]])

        # Predict insurance charges
        pred = model.predict(input_data)[0]
        prediction = round(float(pred), 2)

    return render_template('project.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
