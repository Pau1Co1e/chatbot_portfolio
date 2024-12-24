import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, dataset_directory, tokenizer):
        self.dataset_directory = dataset_directory
        self.tokenizer = tokenizer  # Use the passed tokenizer instance
        self.datasets = None

    def load_datasets(self):
        """Load datasets from JSON files into Hugging Face DatasetDict."""
        squad_train_path = os.path.join(self.dataset_directory, "squad_v2", "train-00000-of-00001.parquet")
        squad_validate_path = os.path.join(self.dataset_directory, "squad_v2", "validation-00000-of-00001.parquet")
        custom_path = os.path.join(self.dataset_directory, "custom", "custom_train.json")
        print(f"custom_path: {custom_path}")
        print(f"squad_train_path: {squad_train_path}")
        print(f"squad_validate_path: {squad_validate_path}")
        print(f"custom_path: {custom_path}")


        self.datasets = DatasetDict({
            "train": Dataset.from_parquet(squad_train_path),
            "validate": Dataset.from_parquet(squad_validate_path),
            "custom": Dataset.from_json(custom_path)  # Load custom dataset
        })
        print("Datasets loaded successfully.")
        for split, dataset in self.datasets.items():
            print(f"Loaded '{split}' split with {len(dataset)} samples.")

    def inspect_datasets(self):
        """Inspect datasets for structure, content, and consistency."""
        print("Inspecting datasets...")
        print("Dataset keys:", self.datasets.keys())

        for split, dataset in self.datasets.items():
            print(f"{split.capitalize()} dataset sample:", dataset[0])

    def preprocess_datasets(self):
        """Tokenize and preprocess datasets for causal language modeling."""

        def tokenize(examples):
            # Concatenate context and question for causal modeling
            inputs = [f"Question: {q} Context: {c}" for q, c in zip(examples["question"], examples["context"])]

            # Generate tokenized inputs
            tokenized_inputs = self.tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

            # Generate labels for each example
            labels = []
            for answer_list in examples["answers"]:
                if answer_list["text"]:  # Check if "text" is non-empty
                    labels.append(
                        self.tokenizer(answer_list["text"][0], max_length=512, padding="max_length", truncation=True)[
                            "input_ids"])
                else:
                    labels.append([0] * 512)  # Pad labels if no answer is provided

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        print("Starting tokenization...")
        for split in self.datasets.keys():
            # Check if tokenization is needed (i.e., original columns exist)
            if "question" in self.datasets[split].column_names:
                # Check which columns exist in the dataset
                current_columns = self.datasets[split].column_names
                columns_to_remove = [col for col in ["answers", "question", "context", "title", "id"] if
                                     col in current_columns]

                # Apply tokenization and remove only existing columns
                self.datasets[split] = self.datasets[split].map(
                    tokenize,
                    batched=True,
                    remove_columns=columns_to_remove  # Only remove columns that are present
                )
                print(f"Tokenized dataset for split '{split}'.")
            else:
                print(f"Skipping tokenization for split '{split}' (already tokenized).")
        print("Datasets tokenized successfully.")

    def run_pipeline(self):
        """Run the full EDA pipeline."""
        self.load_datasets()
        self.inspect_datasets()
        self.preprocess_datasets()
        print("Pipeline executed successfully.")
