from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('svm_model.sav')
scaler = joblib.load('scaler.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[key]) for key in request.form]
        input_array = np.array(data).reshape(1, -1)
        std_data = scaler.transform(input_array)
        result = model.predict(std_data)
        prob = model.predict_proba(std_data)[0][1]
        outcome = 'Diabetic' if result[0] == 1 else 'Not Diabetic'
        return jsonify({'result': outcome, 'probability': round(prob * 100, 2)})
    except:
        return jsonify({'result': 'Error', 'probability': 0})

if __name__ == '__main__':
    app.run(debug=True)
