import joblib

# Load your model
model = joblib.load('tsunami_model_pipeline.pkl')

# Extract the feature names the preprocessor saw during fit
# This works for scikit-learn version 1.0+
try:
    expected_columns = model.named_steps['preprocessor'].feature_names_in_
    print("Your model expects these columns in this order:")
    print(list(expected_columns))
except AttributeError:
    print("Could not find feature names. Check your scikit-learn version.")