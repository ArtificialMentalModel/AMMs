import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import ast

# Load Excel files
file1 = pd.ExcelFile('StudyByModels/Discover Your Inner Self - Deutsch_Llama2_with_logits_fewshot_latest.xlsx')
file2 = pd.ExcelFile('StudyByModels/Discover Your Inner Self_English_Llama2_with_logits_fewshot_latest.xlsx')

#file1 = pd.ExcelFile('StudyByModels/Discover Your Inner Self - Deutsch_Llama3_with_logits_regression.xlsx')
#file2 = pd.ExcelFile('StudyByModels/Discover Your Inner Self_English_Llama3_with_logits_regression.xlsx')

# Extract TIPI sheet data
df1 = file1.parse('TIPI')
df2 = file2.parse('TIPI')

# Combine the data
df = pd.concat([df1, df2], ignore_index=True)
#df = df.sample(frac=1, random_state=None).reset_index(drop=True)

# Clean Actual Output to extract digits
df['Actual Output'] = df['Actual Output'].str.extract('(\d+)').astype(int)

def parse_logits(row):
    # Convert the row to a Python object if it's stored as a string
    logits = ast.literal_eval(row) if isinstance(row, str) else row
    probabilities = {}
    
    # Check if the logits is a list and parse accordingly
    if isinstance(logits, list):
        for logit in logits:
            if 'alternatives' in logit:
                for i, alt in enumerate(logit['alternatives']):
                    if 'probability' in alt:
                        probabilities[f"Alt_{i+1}_Prob"] = alt['probability']
                    else:
                        probabilities[f"Alt_{i+1}_Prob"] = 0  # Default to 0 if probability is missing
    return pd.Series(probabilities)

logit_features = df['Logits'].apply(parse_logits)
df = pd.concat([df, logit_features], axis=1)


# Assuming `df` is your combined DataFrame with features and target
X = df.filter(like='Alt_')  # Feature columns (probabilities)
y = df['Actual Output']     # Target column (integer values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# Convert target to NumPy arrays for TensorFlow
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define the neural network model
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden Layer 2
    tf.keras.layers.Dense(1, activation='linear')  # Output Layer (regression output)
])

# Compile the model
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model2.fit(X_train, y_train, epochs=5, batch_size=8, validation_split=0.2, verbose=1)

# Save the trained model
#model.save('regression_nn_model.h5')
#print("Model saved as 'regression_nn_model.h5'")

# Evaluate the model
y_pred = model2.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_rounded)

print(f"Mean Squared Error: {mse}")
print(f"Accuracy: {accuracy * 100:.2f}%")
