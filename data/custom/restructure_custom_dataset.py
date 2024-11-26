import json
import pandas as pd

def restructure_custom_dataset(input_json, output_parquet):
    """
    Restructure a custom dataset into the Hugging Face `datasets` format and save as Parquet.

    Args:
        input_json (str): Path to the input JSON file.
        output_parquet (str): Path to save the output Parquet file.
    """
    try:
        # Load the input JSON file
        with open(input_json, "r") as f:
            data = json.load(f)

        # Flatten the dataset
        records = []
        for category, entries in data.items():
            for entry in entries:
                records.append({
                    "context": entry["context"],
                    "question": entry["question"],
                    "answers": {
                        "text": entry["answers"]["text"],
                        "answer_start": entry["answers"]["answer_start"]
                    }
                })

        # Convert to a DataFrame
        df = pd.DataFrame(records)
        print(f"Restructured dataset contains {len(df)} records.")

        # Save to Parquet
        df.to_parquet(output_parquet, index=False)
        print(f"Dataset saved to {output_parquet}")

    except Exception as e:
        raise ValueError(f"Error restructuring dataset: {e}")


# Example usage
input_json = "custom_train.json"  # Path to your input JSON
output_parquet = "custom_train_restructured.parquet"  # Desired output Parquet path

restructure_custom_dataset(input_json, output_parquet)
