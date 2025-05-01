from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# model = pickle.load(open('D:\jupyter\smart-healthcare-ai\models\heart-pred\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('heart_pred.html')

@app.route('/predictHeart', methods=['POST'])
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
            # ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang','oldpeak', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0','restecg_1', 'restecg_2', 'thal_1', 'thal_2', 'thal_3', 'slope_0','slope_1', 'slope_2']
            # features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            # prediction = model.predict_proba(features)


            # result = "High Risk" if prediction[0] == 1 else "Low Risk"
            print("\n--- Form Submission Received ---")
            print(f"Age: {type(age)}")
            print(f"Sex: {type(sex)}")
            print(f"Chest Pain Type: {type(cp)}")
            print(f"Resting Blood Pressure: {type(trestbps)}")
            print(f"Cholesterol: {type(chol)}")
            print(f"Fasting Blood Sugar > 120mg/dl: {type(fbs)}")
            print(f"Rest ECG Result: {type(restecg)}")
            print(f"Maximum Heart Rate Achieved: {type(thalach)}")
            print(f"Exercise Induced Angina: {type(exang)}")    
            print(f"Oldpeak (ST depression): {type(oldpeak)}")
            print(f"Slope of ST segment: {type(slope)}")
            print(f"Number of major vessels (ca): {type(ca)}")
            print(f"Thalassemia Test Result: {type(thal)}")
            print("---------------------------------\n")
            # return render_template('index.html', prediction_text=f'Prediction: {result}')

        except Exception as e:
            return render_template('heart_pred.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
