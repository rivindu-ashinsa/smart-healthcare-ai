from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            bp = float(request.form['bp'])
            glucose = float(request.form['glucose'])

            features = np.array([[age, bp, glucose]])
            prediction = model.predict(features)

            result = "High Risk" if prediction[0] == 1 else "Low Risk"

            return render_template('index.html', prediction_text=f'Prediction: {result}')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
