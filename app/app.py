from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('heart_pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = model.predict(features)

            result = "High Risk" if prediction[0] == 1 else "Low Risk"
            print("\n--- Form Submission Received ---")
            print(f"Age: {age}")
            print(f"Sex: {sex}")
            print(f"Chest Pain Type: {cp}")
            print(f"Resting Blood Pressure: {trestbps}")
            print(f"Cholesterol: {chol}")
            print(f"Fasting Blood Sugar > 120mg/dl: {fbs}")
            print(f"Rest ECG Result: {restecg}")
            print(f"Maximum Heart Rate Achieved: {thalach}")
            print(f"Exercise Induced Angina: {exang}")
            print(f"Oldpeak (ST depression): {oldpeak}")
            print(f"Slope of ST segment: {slope}")
            print(f"Number of major vessels (ca): {ca}")
            print(f"Thalassemia Test Result: {thal}")
            print("---------------------------------\n")
            return render_template('index.html', prediction_text=f'Prediction: {result}')

        except Exception as e:
            return render_template('heart_pred.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
