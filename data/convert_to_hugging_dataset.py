from datasets import Dataset
import json

# Paths
input_path = "/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data/custom/custom_train.json"  # Update with the path to your JSON file
output_path = "/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data/custom/custom_dataset_hf"  # Directory to save the processed dataset

# Load the custom dataset
with open(input_path, "r") as f:
    custom_data = json.load(f)

# Process data into Hugging Face-compatible format
records = []
for category, entries in custom_data.items():
    for entry in entries:
        record = {
            "context": entry["context"],
            "question": entry["question"],
            "answers": {
                "text": entry["answers"]["text"],
                "answer_start": entry["answers"]["answer_start"]
            },
            "category": category  # Optional: Keep the category for reference
        }
        records.append(record)

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_list(records)

# Split into train and validation datasets (80-20 split)
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)

# Save the dataset
dataset_split.save_to_disk(output_path)

print(f"Dataset saved to {output_path}")
