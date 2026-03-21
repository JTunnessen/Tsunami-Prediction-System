from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('tsunami_model_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture the 4 inputs from your HTML form
    mag = float(request.form['magnitude'])
    dep = float(request.form['depth'])
    cdi_val = float(request.form['cdi'])
    mmi_val = float(request.form['mmi'])

    # 2. Build the full dictionary with ALL 12 columns in order
    # We use neutral/average values for the ones the user didn't provide
    input_data = {
        'magnitude': [mag],
        'cdi': [cdi_val],
        'mmi': [mmi_val],
        'sig': [500],        # Average significance
        'nst': [50],         # Average number of stations
        'dmin': [0.5],       # Average distance
        'gap': [90],         # Average gap
        'depth': [dep],
        'latitude': [0.0],   # Neutral location
        'longitude': [0.0],  # Neutral location
        'Year': [2024],
        'Month': [1]
    }

    # 3. Create the DataFrame
    df_input = pd.DataFrame(input_data)

    # 4. Ensure the column order is EXACTLY what the model expects
    cols_order = ['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude', 'Year', 'Month']
    df_input = df_input[cols_order]

    # 5. Run the prediction
    prediction = model.predict(df_input)[0]
    
    result = "TSUNAMI LIKELY" if prediction == 1 else "NO TSUNAMI"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)