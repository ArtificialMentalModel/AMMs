from joblib import load
import numpy as np

# Load the saved model
model = load('regression_model.joblib')
print("Model loaded successfully!")

# Example hardcoded input
input_data = [{'token': '2', 'alternatives': [{'token': '2', 'probability': 0.6384092569351196}, {'token': '3', 'probability': 0.34171587228775024}, {'token': '4', 'probability': 0.011692874133586884}, {'token': '1', 'probability': 0.008036387152969837}, {'token': '5', 'probability': 0.00012202608195366338}]}]

# Parse the input list to extract probabilities as features
def extract_features(input_list):
    features = []
    for entry in input_list:
        if 'alternatives' in entry:
            probabilities = [alt['probability'] for alt in entry['alternatives']]
            features.append(probabilities)
    return features

# Extract features
features = extract_features(input_data)

# Ensure the features are in the correct shape (e.g., 2D array)
features_array = np.array(features).reshape(len(features), -1)

# Predict using the loaded model
predicted_output = model.predict(features_array)
rounded_output = np.round(predicted_output).astype(int)

print(f"Predicted Output (Rounded): {rounded_output}")
