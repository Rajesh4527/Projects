import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define dataset path
dataset_path = r"C:\Cropify-Crop-Recommendation-System-main (1)\Cropify-Crop-Recommendation-System-main\Dataset\Crop_recommendation.csv"

# Verify if dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}. Please check the path.")

# Load dataset
df = pd.read_csv(dataset_path, encoding='utf-8')

# Select only the 7 required features
required_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[required_features]  # Use only the required columns
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a new Random Forest model
rdf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rdf_clf.fit(X_train, y_train)

# Save the trained model
model_path = r"C:\Cropify-Crop-Recommendation-System-main (1)\Cropify-Crop-Recommendation-System-main\Model\RDF_model.pkl"
joblib.dump(rdf_clf, model_path)

print("✅ Model retrained and saved successfully!")
