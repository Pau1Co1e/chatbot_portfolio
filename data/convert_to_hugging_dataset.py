from datasets import Dataset
import json
import os
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
input_path = os.path.join(BASE_DIR, "custom", "custom_train.json")
output_path = os.path.join(BASE_DIR, "custom", "custom_dataset_hf")

# Load the custom dataset
logger.info("Loading custom dataset...")
try:
    with open(input_path, "r") as f:
        custom_data = json.load(f)  # JSON loaded as a list
    logger.info("Custom dataset loaded successfully.")
except FileNotFoundError:
    logger.error(f"Input file not found at {input_path}. Please check the path.")
    raise

# Process data into Hugging Face-compatible format
records = []
for entry in custom_data:
    try:
        record = {
            "id": entry["id"],
            "title": entry["title"],
            "context": entry["context"],
            "question": entry["question"],
            "answers": {
                "text": entry["answers"]["text"],
                "answer_start": entry["answers"]["answer_start"]
            }
        }
        records.append(record)
    except KeyError as e:
        logger.warning(f"Missing key {e} in entry: {entry}. Skipping this entry.")

# Convert to Hugging Face Dataset using pandas as fallback
logger.info("Converting records to Hugging Face Dataset...")
records_df = pd.DataFrame(records)
hf_dataset = Dataset.from_pandas(records_df)
logger.info(f"Processed {len(records)} records into Hugging Face Dataset.")

# Split into train and validation datasets (80-20 split)
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
logger.info(f"Dataset split into {len(dataset_split['train'])} train and {len(dataset_split['test'])} validation samples.")

# Save the dataset
dataset_split.save_to_disk(output_path)
logger.info(f"Dataset saved to {output_path}")
