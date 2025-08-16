import pandas as pd
import re

# Replace with actual file paths
file_paths = [
    '/Users/ebad/Documents/GitHub/fedwell/StudyByModels/TIPI_BERT_predictions_Discover_Your_Inner_Self_-_Deutsch_June_10,_2024_06.43_-_Generated_(8)_english.xlsx',
    '/Users/ebad/Documents/GitHub/fedwell/StudyByModels/TIPI_BERT_predictions_Discover_Your_Inner_Self_ssh_english.xlsx'
]

# Initialize confusion matrix components
TP = FP = TN = FN = 0

# Regex pattern to extract a single digit
digit_pattern = re.compile(r'\d')

# Define function to convert score to YES/NO
def to_yes_no(score):
    if score in [1, 2]:
        return "YES"
    elif score in [3, 4, 5]:
        return "NO"
    else:
        return None  # invalid score

# Process each file
for file_path in file_paths:
    df = pd.read_excel(file_path, sheet_name='TIPI')

    for _, row in df.iterrows():
        try:
            actual_match = digit_pattern.search(str(row['Actual Output']))
            model_match = digit_pattern.search(str(row['Model_Output_query_1']))

            if actual_match and model_match:
                actual_score = int(actual_match.group())
                model_score = int(model_match.group())

                actual_label = to_yes_no(actual_score)
                model_label = to_yes_no(model_score)

                if actual_label and model_label:
                    if actual_label == "YES" and model_label == "YES":
                        TP += 1
                    elif actual_label == "NO" and model_label == "YES":
                        FP += 1
                    elif actual_label == "NO" and model_label == "NO":
                        TN += 1
                    elif actual_label == "YES" and model_label == "NO":
                        FN += 1
        except Exception:
            continue

# Calculate sensitivity and specificity
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

# Print results
print(f"Sensitivity (Recall for YES): {sensitivity:.2f}")
print(f"Specificity (Recall for NO): {specificity:.2f}")
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
