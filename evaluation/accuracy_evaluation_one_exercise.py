import pandas as pd
import re
import os
from typing import Dict, List, Tuple
from collections import defaultdict

def to_yes_no(score: int) -> str:
    return "YES" if score in [1, 2] else "NO"

def extract_score(text: str, is_llama2: bool = False) -> int:
    if pd.isna(text):
        return None

    text = str(text).strip()

    if is_llama2:
        dict_pattern = r'{\s*["\']?(?:score|rating|value)["\']?\s*:\s*(\d)\s*}'
        dict_match = re.search(dict_pattern, text, re.IGNORECASE)
        if dict_match:
            score = int(dict_match.group(1))
            if 1 <= score <= 5:
                return score

        score_pattern = r'(?:score|rating|value)\s*:?\s*(\d)'
        score_match = re.search(score_pattern, text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                return score

        return 0

    dict_pattern = r'["\']?(?:score|rating|value)["\']?\s*:\s*(\d)'
    dict_match = re.search(dict_pattern, text, re.IGNORECASE)
    if dict_match:
        score = int(dict_match.group(1))
        if 1 <= score <= 5:
            return score

    score_patterns = [
        r'(?:^|\s)(?:score|rating|value)\s*:?\s*(\d)(?:\s|$)',
        r'(?:^|\s)(\d)(?:\s|$)',
        r'(\d)'
    ]

    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

    return None

def calculate_metrics(actual_scores: List[int], predicted_scores: List[int], total_expected: int = 116) -> Dict:
    correct_predictions = sum(1 for a, p in zip(actual_scores, predicted_scores) if a == p)
    total_processed = len(actual_scores)
    missing_samples = total_expected - total_processed
    total = total_expected
    accuracy = (correct_predictions / total) * 100

    actual_binary = [to_yes_no(score) for score in actual_scores]
    predicted_binary = [to_yes_no(score) for score in predicted_scores]

    TP = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "YES" and p == "YES")
    FP = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "NO" and p == "YES")
    TN = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "NO" and p == "NO")
    FN = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "YES" and p == "NO")
    FN += missing_samples

    sensitivity = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "total_samples": total,
        "processed_samples": total_processed,
        "missing_samples": missing_samples,
        "correct_predictions": correct_predictions
    }

def get_column_names(query_num: int, df_columns: List[str]) -> Tuple[str, str]:
    return f"Actual_Output_query_{query_num}", f"Model_Output_query_{query_num}"

def process_files(file_paths: List[str], debug: bool = False) -> Dict:
    query_num = 1  # Only process query 1
    results = {
        f"query_{query_num}": {"actual": [], "predicted": [], "invalid_rows": []}
    }

    english_files = [f for f in file_paths if '_english.xlsx' in f]
    german_files = [f for f in file_paths if '_german.xlsx' in f]

    for file_group, language in [(english_files, "English"), (german_files, "German")]:
        for file_path in file_group:
            try:
                is_llama2 = "Llama-2" in file_path
                df = pd.read_excel(file_path, sheet_name='TIPI')
                actual_col, model_col = get_column_names(query_num, df.columns.tolist())

                if actual_col in df.columns and model_col in df.columns:
                    for idx, row in df.iterrows():
                        actual_val = row[actual_col]
                        model_val = row[model_col]

                        try:
                            actual_score = extract_score(actual_val)
                            model_score = extract_score(model_val, is_llama2=is_llama2)

                            if actual_score is not None and model_score is not None:
                                results[f"query_{query_num}"]["actual"].append(actual_score)
                                results[f"query_{query_num}"]["predicted"].append(model_score)
                            else:
                                results[f"query_{query_num}"]["invalid_rows"].append((idx, str(actual_val), str(model_val)))
                        except Exception:
                            results[f"query_{query_num}"]["invalid_rows"].append((idx, str(actual_val), str(model_val)))
                            continue
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    return results

def get_model_files(directory: str) -> Dict[str, Dict[str, List[str]]]:
    model_files = defaultdict(lambda: {"zeroshot": [], "fewshot": []})
    for file in os.listdir(directory):
        if file.endswith('.xlsx') and not file.startswith('~'):
            file_path = os.path.join(directory, file)
            if '_zeroshot_' in file:
                model_name = file.split('_zeroshot_')[0]
                model_files[model_name]["zeroshot"].append(file_path)
            elif '_fewshot_' in file:
                model_name = file.split('_fewshot_')[0]
                model_files[model_name]["fewshot"].append(file_path)
    return model_files

def create_results_table(all_results: List[Dict]) -> pd.DataFrame:
    grouped_results = {}
    for result in all_results:
        model = result['model']
        exercise = result['exercise']
        shot_type = result['shot_type'].lower()
        metrics = result['metrics']
        key = (model, exercise)
        if key not in grouped_results:
            grouped_results[key] = {}
        grouped_results[key][f"{shot_type}_accuracy"] = f"{metrics['accuracy']:.1f}"
        grouped_results[key][f"{shot_type}_sensitivity"] = f"{metrics['sensitivity']:.1f}"
        grouped_results[key][f"{shot_type}_specificity"] = f"{metrics['specificity']:.1f}"

    df_data = []
    for (model, exercise), metrics in grouped_results.items():
        row = {
            'Model': model,
            'Exercise': exercise,
            'Zeroshot Accuracy (%)': metrics.get('zeroshot_accuracy', 'N/A'),
            'Zeroshot Sensitivity (%)': metrics.get('zeroshot_sensitivity', 'N/A'),
            'Zeroshot Specificity (%)': metrics.get('zeroshot_specificity', 'N/A'),
            'Fewshot Accuracy (%)': metrics.get('fewshot_accuracy', 'N/A'),
            'Fewshot Sensitivity (%)': metrics.get('fewshot_sensitivity', 'N/A'),
            'Fewshot Specificity (%)': metrics.get('fewshot_specificity', 'N/A')
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    return df

def main():
    directory = 'results_zeroshot_fewshot_all_one_exercise'
    model_files = get_model_files(directory)
    all_results = []
    debug = False
    query_num = 1

    for model_name, files in model_files.items():
        if files["zeroshot"]:
            zeroshot_results = process_files(files["zeroshot"], debug=debug)
            query_data = zeroshot_results[f"query_{query_num}"]
            metrics = calculate_metrics(query_data["actual"], query_data["predicted"])
            all_results.append({
                'model': model_name,
                'shot_type': 'Zeroshot',
                'exercise': query_num,
                'metrics': metrics
            })

        if files["fewshot"]:
            fewshot_results = process_files(files["fewshot"], debug=debug)
            query_data = fewshot_results[f"query_{query_num}"]
            metrics = calculate_metrics(query_data["actual"], query_data["predicted"])
            all_results.append({
                'model': model_name,
                'shot_type': 'Fewshot',
                'exercise': query_num,
                'metrics': metrics
            })

    results_df = create_results_table(all_results)
    results_df = results_df.sort_values(['Model', 'Exercise'])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\nComprehensive Results Table:")
    print(results_df.to_string(index=False))

    results_df.to_csv('model_comparison_results_one_exercise.csv', index=False)
    print("\nResults have been saved to 'model_comparison_results_one_exercise.csv'")

if __name__ == "__main__":
    main()
