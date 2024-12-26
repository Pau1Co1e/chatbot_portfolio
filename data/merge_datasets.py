import pandas as pd
import json
import os
import numpy as np

# File paths
squad_train_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/squad_v2/squad_v2_train.parquet"
squad_validate_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/squad_v2/squad_v2_validate.parquet"
custom_data_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/custom/custom_train.json"
output_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/final_cleaned_merged_dataset.json"


# Load SQuAD v2 datasets
def load_squad_data(file_path):
    return pd.read_parquet(file_path)


# Load custom dataset
def load_custom_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Validate and clean data
def validate_and_clean(data):
    cleaned_data = []
    for entry in data:
        try:
            # Ensure required keys exist
            if all(key in entry for key in ["context", "question", "answers"]):
                # Clean answers
                answers = entry["answers"]
                if isinstance(answers, dict) and "text" in answers and "answer_start" in answers:
                    if len(answers["text"]) > 0 and len(answers["answer_start"]) > 0:
                        cleaned_data.append(entry)
        except Exception as e:
            print(f"Error processing entry: {e}")
    return cleaned_data


# Merge datasets
def merge_datasets(squad_train, squad_validate, custom_data):
    # Convert SQuAD to list of dicts
    squad_combined = squad_train.to_dict("records") + squad_validate.to_dict("records")
    # Combine with custom data
    merged_data = squad_combined + custom_data
    # Validate and clean merged data
    return validate_and_clean(merged_data)


# Save to JSON
def save_to_json(data, output_path):
    """
    Save the cleaned and merged dataset to a JSON file.
    Converts numpy arrays to Python lists for JSON serialization.
    """
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):  # Handle NumPy floats
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):  # Handle NumPy ints
            return int(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert)


def main():
    # Load datasets
    print("Loading datasets...")
    squad_train = load_squad_data(squad_train_path)
    squad_validate = load_squad_data(squad_validate_path)
    custom_data = load_custom_data(custom_data_path)

    # Merge and clean datasets
    print("Merging and cleaning datasets...")
    final_dataset = merge_datasets(squad_train, squad_validate, custom_data)

    # Save the merged dataset
    print("Saving merged dataset...")
    save_to_json(final_dataset, output_path)
    print("Process completed successfully.")


if __name__ == "__main__":
    main()
