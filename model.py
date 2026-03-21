import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv('earthquake_data_tsunami.csv') 

# 2. Define Target and Features
X = df.drop(columns=['tsunami']) 
y = df['tsunami']

# 3. Triple Split (Train, Val, Test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

# 4. Preprocessing Setup
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 5. Define the Pipeline (Empty shell for now)
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 6. Hyperparameter Tuning (The Tournament)
param_grid = {
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__min_samples_split': [2, 5, 10]
}

print("Starting Grid Search (this may take a moment)...")
grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Select the winning model
model_pipeline = grid_search.best_estimator_
print(f"Best settings found: {grid_search.best_params_}")

# 7. Internal Validation Check
val_predictions = model_pipeline.predict(X_val)
print("\n--- VALIDATION REPORT (BEST MODEL) ---")
print(classification_report(y_val, val_predictions))

# 8. Save the Winning Model
model_filename = 'tsunami_model_pipeline.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"[SUCCESS] Best model saved as {model_filename}")

# 9. Pause Function for Final Testing
def run_testing_phase():
    confirm = input("\nWould you like to run the FINAL TEST on unseen data? (y/n): ")
    
    if confirm.lower() == 'y':
        print("\n--- STARTING FINAL TEST PHASE ---")
        loaded_model = joblib.load(model_filename)
        final_preds = loaded_model.predict(X_test)
        
        print(f"Final Test Accuracy: {accuracy_score(y_test, final_preds):.2%}")
        print("\nFinal Classification Report:")
        print(classification_report(y_test, final_preds))
        
        cm = confusion_matrix(y_test, final_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Tsunami', 'Tsunami'], 
                    yticklabels=['No Tsunami', 'Tsunami'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Final Test Set Confusion Matrix')
        plt.show()
    else:
        print("\nTesting skipped.")

run_testing_phase()