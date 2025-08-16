import pandas as pd
import re
import os
from typing import Dict, List, Tuple
from collections import defaultdict

def to_yes_no(score: int) -> str:
    return "YES" if score in [1, 2] else "NO"

def extract_score(text: str, is_llama2: bool = False) -> int:
    """Extract a valid score (1-5) from text that might contain numbers in different formats."""
    if pd.isna(text):
        return None
    
    text = str(text).strip()
    
    # For Llama-2, only accept dictionary format
    if is_llama2:
        # First try to find a JSON-like structure
        dict_pattern = r'{\s*["\']?(?:score|rating|value)["\']?\s*:\s*(\d)\s*}'
        dict_match = re.search(dict_pattern, text, re.IGNORECASE)
        if dict_match:
            score = int(dict_match.group(1))
            if 1 <= score <= 5:
                return score
        
        # If no JSON found, try to find a score in the text
        score_pattern = r'(?:score|rating|value)\s*:?\s*(\d)'
        score_match = re.search(score_pattern, text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                return score
        
        # For Llama-2, if no dictionary format is found, return 0 to mark as incorrect
        return 0
    
    # For other models, try all formats
    dict_pattern = r'["\']?(?:score|rating|value)["\']?\s*:\s*(\d)'
    dict_match = re.search(dict_pattern, text, re.IGNORECASE)
    if dict_match:
        score = int(dict_match.group(1))
        if 1 <= score <= 5:
            return score
    
    # Try to find a number that's not part of other text
    score_patterns = [
        r'(?:^|\s)(?:score|rating|value)\s*:?\s*(\d)(?:\s|$)',  # score: 3 or rating: 4
        r'(?:^|\s)(\d)(?:\s|$)',  # just a number
        r'(\d)'  # any digit as last resort
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
    
    return None

def calculate_metrics(actual_scores: List[int], predicted_scores: List[int], total_expected: int = 116) -> Dict:
    # Count actual correct predictions
    correct_predictions = sum(1 for a, p in zip(actual_scores, predicted_scores) if a == p)
    
    # Consider missing predictions as incorrect
    total_processed = len(actual_scores)
    missing_samples = total_expected - total_processed
    
    # Total includes both processed and missing samples
    total = total_expected
    accuracy = (correct_predictions / total) * 100

    # Convert scores to YES/NO for sensitivity/specificity
    actual_binary = [to_yes_no(score) for score in actual_scores]
    predicted_binary = [to_yes_no(score) for score in predicted_scores]

    # Calculate confusion matrix
    TP = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "YES" and p == "YES")
    FP = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "NO" and p == "YES")
    TN = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "NO" and p == "NO")
    FN = sum(1 for a, p in zip(actual_binary, predicted_binary) if a == "YES" and p == "NO")

    # Add missing samples to FN (assuming worst case)
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
    """Get the appropriate column names based on the exercise number and available columns."""

    return f"Actual_Output_query_{query_num}", f"Model_Output_query_{query_num}"
    # First check for the new format (Llama-2-13b)
    if query_num == 1:
        # Try different possible column names for exercise 1
        possible_actual_cols = ["Actual Output"]
        possible_model_cols = ["Model_Output", "Model Output", "Model"]
        
        for actual_col in possible_actual_cols:
            for model_col in possible_model_cols:
                if actual_col in df_columns and model_col in df_columns:
                    return actual_col, model_col
    else:
        # Try different possible column names for exercises 2-4
        possible_actual_cols = [
            f"Actual_Output_query_{query_num}",
            f"Actual Output query {query_num}",
            f"Actual_{query_num}"
        ]
        possible_model_cols = [
            f"Model_Output_query_{query_num}",
            f"Model Output query {query_num}",
            f"Model_{query_num}"
        ]
        
        for actual_col in possible_actual_cols:
            for model_col in possible_model_cols:
                if actual_col in df_columns and model_col in df_columns:
                    return actual_col, model_col
    
    # If no matching columns found, return the default format
    if query_num == 1:
        return "Actual_Output", "Model_Output"
    else:
        return f"Actual_Output_query_{query_num}", f"Model_Output_query_{query_num}"

def process_files(file_paths: List[str], debug: bool = False) -> Dict:
    results = {
        f"query_{i}": {"actual": [], "predicted": [], "invalid_rows": []}
        for i in range(1, 5)
    }

    # Group files by language
    english_files = [f for f in file_paths if '_english.xlsx' in f]
    german_files = [f for f in file_paths if '_german.xlsx' in f]

    if debug:
        print(f"\nProcessing {len(english_files)} English files and {len(german_files)} German files")

    # Process English files
    for file_path in english_files:
        try:
            if debug:
                print(f"\nProcessing English file: {os.path.basename(file_path)}")
            
            # Check if this is a Llama-2 file
            is_llama2 = "Llama-2" in file_path
            
            df = pd.read_excel(file_path, sheet_name='TIPI')
            
            if debug and is_llama2:
                print("Available columns:", df.columns.tolist())
                print(f"Is Llama-2 file: {is_llama2}")
            
            for query_num in range(1, 5):
                actual_col, model_col = get_column_names(query_num, df.columns.tolist())
                
                if debug and is_llama2:
                    print(f"\nUsing columns for query {query_num}:")
                    print(f"Actual column: {actual_col}")
                    print(f"Model column: {model_col}")
                
                if actual_col in df.columns and model_col in df.columns:
                    if debug and is_llama2:
                        print(f"Total rows in file: {len(df)}")
                        print("\nFirst 5 rows of data:")
                        print(df[[actual_col, model_col]].head().to_string())
                    
                    for idx, row in df.iterrows():
                        actual_val = row[actual_col]
                        model_val = row[model_col]
                        
                        if debug and is_llama2:
                            print(f"\nRow {idx}:")
                            print(f"Actual value: '{actual_val}'")
                            print(f"Model value: '{model_val}'")
                        
                        try:
                            actual_score = extract_score(actual_val)
                            model_score = extract_score(model_val, is_llama2=is_llama2)
                            
                            if actual_score is not None and model_score is not None:
                                results[f"query_{query_num}"]["actual"].append(actual_score)
                                results[f"query_{query_num}"]["predicted"].append(model_score)
                                if debug and is_llama2:
                                    print(f"Valid scores - Actual: {actual_score}, Predicted: {model_score}")
                            else:
                                if debug and is_llama2:
                                    print(f"Invalid scores - Actual: {actual_score}, Predicted: {model_score}")
                                results[f"query_{query_num}"]["invalid_rows"].append((idx, str(actual_val), str(model_val)))
                        except Exception as e:
                            if debug and is_llama2:
                                print(f"Error processing row: {str(e)}")
                            results[f"query_{query_num}"]["invalid_rows"].append((idx, str(actual_val), str(model_val)))
                            continue
                else:
                    if debug and is_llama2:
                        print(f"Columns {actual_col} and/or {model_col} not found in file")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Process German files
    for file_path in german_files:
        try:
            if debug:
                print(f"\nProcessing German file: {os.path.basename(file_path)}")
            
            # Check if this is a Llama-2 file
            is_llama2 = "Llama-2" in file_path
            
            df = pd.read_excel(file_path, sheet_name='TIPI')
            
            if debug and is_llama2:
                print("Available columns:", df.columns.tolist())
                print(f"Is Llama-2 file: {is_llama2}")
            
            for query_num in range(1, 5):
                actual_col, model_col = get_column_names(query_num, df.columns.tolist())
                
                if debug and is_llama2:
                    print(f"\nUsing columns for query {query_num}:")
                    print(f"Actual column: {actual_col}")
                    print(f"Model column: {model_col}")
                
                if actual_col in df.columns and model_col in df.columns:
                    if debug and is_llama2:
                        print(f"Total rows in file: {len(df)}")
                        print("\nFirst 5 rows of data:")
                        print(df[[actual_col, model_col]].head().to_string())
                    
                    for idx, row in df.iterrows():
                        actual_val = row[actual_col]
                        model_val = row[model_col]
                        
                        if debug and is_llama2:
                            print(f"\nRow {idx}:")
                            print(f"Actual value: '{actual_val}'")
                            print(f"Model value: '{model_val}'")
                        
                        try:
                            actual_score = extract_score(actual_val)
                            model_score = extract_score(model_val, is_llama2=is_llama2)
                            
                            if actual_score is not None and model_score is not None:
                                results[f"query_{query_num}"]["actual"].append(actual_score)
                                results[f"query_{query_num}"]["predicted"].append(model_score)
                                if debug and is_llama2:
                                    print(f"Valid scores - Actual: {actual_score}, Predicted: {model_score}")
                            else:
                                if debug and is_llama2:
                                    print(f"Invalid scores - Actual: {actual_score}, Predicted: {model_score}")
                                results[f"query_{query_num}"]["invalid_rows"].append((idx, str(actual_val), str(model_val)))
                        except Exception as e:
                            if debug and is_llama2:
                                print(f"Error processing row: {str(e)}")
                            results[f"query_{query_num}"]["invalid_rows"].append((idx, str(actual_val), str(model_val)))
                            continue
                else:
                    if debug and is_llama2:
                        print(f"Columns {actual_col} and/or {model_col} not found in file")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    if debug:
        for query_num in range(1, 5):
            print(f"\nQuery {query_num} Summary:")
            print(f"Valid pairs: {len(results[f'query_{query_num}']['actual'])}")
            print(f"Invalid rows: {len(results[f'query_{query_num}']['invalid_rows'])}")
            if len(results[f"query_{query_num}"]["actual"]) > 0:
                print("\nSample of valid scores:")
                for i in range(min(5, len(results[f"query_{query_num}"]["actual"]))):
                    print(f"Actual: {results[f'query_{query_num}']['actual'][i]}, Predicted: {results[f'query_{query_num}']['predicted'][i]}")
            print("\nSample of invalid rows:")
            for idx, actual, model in results[f"query_{query_num}"]["invalid_rows"][:5]:
                print(f"Row {idx}: Actual='{actual}', Model='{model}'")

    return results

def get_model_files(directory: str) -> Dict[str, Dict[str, List[str]]]:
    """Group files by model name and shot type."""
    model_files = defaultdict(lambda: {"zeroshot": [], "fewshot": []})
    
    for file in os.listdir(directory):
        if file.endswith('.xlsx') and not file.startswith('~'):
            file_path = os.path.join(directory, file)
            
            # Extract model name (everything before _zeroshot or _fewshot)
            if '_zeroshot_' in file:
                model_name = file.split('_zeroshot_')[0]
                model_files[model_name]["zeroshot"].append(file_path)
            elif '_fewshot_' in file:
                model_name = file.split('_fewshot_')[0]
                model_files[model_name]["fewshot"].append(file_path)
    
    return model_files

def create_results_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame with zeroshot and fewshot results side by side."""
    # Create a dictionary to store results by model and exercise
    grouped_results = {}
    
    for result in all_results:
        model = result['model']
        exercise = result['exercise']
        shot_type = result['shot_type'].lower()
        metrics = result['metrics']
        
        key = (model, exercise)
        if key not in grouped_results:
            grouped_results[key] = {}
        
        # Store metrics for each shot type
        grouped_results[key][f"{shot_type}_accuracy"] = f"{metrics['accuracy']:.1f}"
        grouped_results[key][f"{shot_type}_sensitivity"] = f"{metrics['sensitivity']:.1f}"
        grouped_results[key][f"{shot_type}_specificity"] = f"{metrics['specificity']:.1f}"
    
    # Convert to DataFrame format
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
    directory = 'Studies_All_exercises'
    model_files = get_model_files(directory)
    all_results = []
    
    # Enable debug mode for Llama-2 zeroshot files
    debug = False
    
    # Process each model's results
    for model_name, files in model_files.items():
        # Process zeroshot results
        if files["zeroshot"]:
            zeroshot_results = process_files(files["zeroshot"], debug=debug)
            for query_num in range(1, 5):
                query_data = zeroshot_results[f"query_{query_num}"]
                metrics = calculate_metrics(query_data["actual"], query_data["predicted"])
                all_results.append({
                    'model': model_name,
                    'shot_type': 'Zeroshot',
                    'exercise': query_num,
                    'metrics': metrics
                })
        
        # Process fewshot results
        if files["fewshot"]:
            fewshot_results = process_files(files["fewshot"], debug=debug)
            for query_num in range(1, 5):
                query_data = fewshot_results[f"query_{query_num}"]
                metrics = calculate_metrics(query_data["actual"], query_data["predicted"])
                all_results.append({
                    'model': model_name,
                    'shot_type': 'Fewshot',
                    'exercise': query_num,
                    'metrics': metrics
                })
    
    # Create and display the results table
    results_df = create_results_table(all_results)
    
    # Sort the DataFrame for better readability
    results_df = results_df.sort_values(['Model', 'Exercise'])
    
    # Display the full table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\nComprehensive Results Table:")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults have been saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()
