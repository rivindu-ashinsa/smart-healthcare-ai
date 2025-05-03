from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.encoders import CustomCategoricalEncoder, CustomStandardScaler

app = Flask(__name__)

# preprocessor and model
with open('models/heart-pred/preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

with open('models/heart-pred/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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

            pred_df = pd.DataFrame({
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }, index=[0])

            # Preprocess the input data
            X_new = preprocessor.transform(pred_df)

            # Make predictions
            y_pred_new = model.predict(X_new)
            y_pred_new_proba = model.predict_proba(X_new)

            # Print the prediction and probability
            print(f"Prediction: {y_pred_new[0]}")
            print(f"Probability of No Disease: {y_pred_new_proba[0][0]:.2f}")

            return render_template('heart_pred.html', prediction_text=f'Probability of No Disease: {y_pred_new_proba[0][0]:.2f}')

        except Exception as e:
            print(e)
            return render_template('heart_pred.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
