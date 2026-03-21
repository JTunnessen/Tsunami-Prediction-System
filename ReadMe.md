# 🌊 Tsunami Prediction System
**Machine Learning Pipeline & Flask Web Application**

This project uses a **Decision Tree Classifier** optimized via **Grid Search** to predict the likelihood of a tsunami based on seismic activity data (magnitude, depth, intensity, etc.). The system is wrapped in a **Flask web interface** and is ready for deployment to **Azure App Service**.

## 🚀 Features
* **Automated Preprocessing:** Uses Scikit-Learn `Pipelines` to handle missing values and feature scaling (StandardScaler/OneHotEncoder).
* **Hyperparameter Tuning:** Optimized using `GridSearchCV` for maximum accuracy.
* **Interactive Web Interface:** A user-friendly HTML form to input earthquake parameters.
* **Cloud Ready:** Configured for seamless deployment to Azure.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **ML Libraries:** Scikit-Learn, Pandas, Joblib
* **Web Framework:** Flask
* **Visualization:** Seaborn, Matplotlib
* **Deployment:** Azure App Service, Gunicorn

## 📂 Project Structure
```text
├── app.py                      # Flask Application Logic
├── earthquake_data_tsunami.csv  # Raw Dataset
├── train_model.py              # ML Training & Grid Search Script
├── tsunami_model_pipeline.pkl   # Serialized Winning Model
├── requirements.txt            # Python Dependencies
├── Procfile                    # Azure/Heroku Deployment Config
└── templates/
    └── index.html              # Web Interface (HTML/CSS)
```

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd tsunami-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional):**
   If you want to re-run the Grid Search tournament:
   ```bash
   python train_model.py
   ```

4. **Run the Web App:**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## ☁️ Deployment to Azure
To push this project to a live URL using the Azure CLI:

1. **Login to Azure:**
   ```bash
   az login
   ```

2. **Deploy via App Service:**
   ```bash
   az webapp up --runtime PYTHON:3.9 --sku F1 --name your-unique-app-name
   ```

## 📊 Model Performance
* **Accuracy:** ~89.83%
* **Recall (Tsunami Catch Rate):** 89%
* **Best Parameters:** `max_depth: 10`, `criterion: 'gini'`

