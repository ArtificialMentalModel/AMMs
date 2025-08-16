import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# Load Excel files
#file1 = pd.ExcelFile('Datasets/Mistral_Phi/Discover Your Inner Self - Deutsch_Mistral_7B_Finetuned.xlsx')
file2 = pd.ExcelFile('Datasets/Mistral_Phi/Discover Your Inner Self_English_Phi_3_Finetuned.xlsx')

#df1 = file1.parse('TIPI')
df2 = file2.parse('TIPI')

df = df2
#df = pd.concat([df1, df2], ignore_index=True)

# Extract digits from actual output
df['Actual Output'] = df['Actual Output'].str.extract('(\d+)').astype(int)

# Parse logits
def parse_logits(row):
    logits = ast.literal_eval(row) if isinstance(row, str) else row
    probabilities = {}
    if isinstance(logits, list):
        pass
    else:
        logits = [logits]
    if isinstance(logits, list):
        for logit in logits:
            print(logit)
            if 'alternatives' in logit:
                for i, alt in enumerate(logit['alternatives']):
                    if 'probability' in alt:
                        probabilities[f"Alt_{i+1}_Prob"] = alt['probability']
                    else:
                        probabilities[f"Alt_{i+1}_Prob"] = 0
    return pd.Series(probabilities)

logit_features = df['Logits'].apply(parse_logits)
print(logit_features)
df = pd.concat([df, logit_features], axis=1)

# Features and target
X = df.filter(like='Alt_')
print(X)
y = df['Actual Output']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

# Mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred_rounded)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Sensitivity & Specificity
def to_yes_no(value):
    return "YES" if value in [1, 2] else "NO"

actual_labels = y_test.apply(to_yes_no).values
pred_labels = np.array([to_yes_no(val) for val in y_pred_rounded])

TP = FP = TN = FN = 0
for actual, pred in zip(actual_labels, pred_labels):
    if actual == "YES" and pred == "YES":
        TP += 1
    elif actual == "NO" and pred == "YES":
        FP += 1
    elif actual == "NO" and pred == "NO":
        TN += 1
    elif actual == "YES" and pred == "NO":
        FN += 1

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

sensitivity_percent = sensitivity * 100
specificity_percent = specificity * 100

print(f"Sensitivity (Recall for YES): {sensitivity_percent:.2f}%")
print(f"Specificity (Recall for NO): {specificity_percent:.2f}%")