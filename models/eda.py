import os
from datasets import Dataset, DatasetDict

class EDA:
    def __init__(self, dataset_directory, tokenizer):
        self.dataset_directory = dataset_directory
        self.tokenizer = tokenizer  # Use the passed tokenizer instance
        self.datasets = None

    def load_datasets(self):
        """Load datasets from JSON and Parquet files into a DatasetDict."""
        squad_train_path = os.path.join(self.dataset_directory, "squad_v2", "train-00000-of-00001.parquet")
        squad_validate_path = os.path.join(self.dataset_directory, "squad_v2", "validation-00000-of-00001.parquet")
        custom_path = os.path.join(self.dataset_directory, "custom", "custom_train.json")

        self.datasets = DatasetDict({
            "train": Dataset.from_parquet(squad_train_path),
            "validate": Dataset.from_parquet(squad_validate_path),
            "custom": Dataset.from_json(custom_path)
        })
        print("Datasets loaded successfully.")
        for split, dataset in self.datasets.items():
            print(f"Loaded '{split}' split with {len(dataset)} samples.")

    def preprocess_datasets(self):
        """Tokenize and preprocess datasets for question-answering."""

        def tokenize(examples):
            # Concatenate context and question for causal modeling
            inputs = [f"Question: {q} Context: {c}" for q, c in zip(examples["question"], examples["context"])]

            # Tokenize inputs
            tokenized_inputs = self.tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

            # Initialize start and end positions
            tokenized_inputs["start_positions"] = []
            tokenized_inputs["end_positions"] = []

            # Align start and end positions with tokenized inputs
            for i, answer_list in enumerate(examples["answers"]):
                if answer_list["text"]:  # Ensure answers exist
                    start_char = answer_list["answer_start"][0]
                    end_char = start_char + len(answer_list["text"][0])

                    # Map character positions to token positions
                    token_start_index = tokenized_inputs.char_to_token(i, start_char)
                    token_end_index = tokenized_inputs.char_to_token(i, end_char - 1)

                    # If mapping is successful, set token positions
                    if token_start_index is not None and token_end_index is not None:
                        tokenized_inputs["start_positions"].append(token_start_index)
                        tokenized_inputs["end_positions"].append(token_end_index)
                    else:
                        tokenized_inputs["start_positions"].append(0)
                        tokenized_inputs["end_positions"].append(0)
                else:
                    # Default to 0 if no answer exists
                    tokenized_inputs["start_positions"].append(0)
                    tokenized_inputs["end_positions"].append(0)

            return tokenized_inputs

        print("Starting tokenization...")
        self.datasets = self.datasets.map(
            tokenize,
            batched=True,
            remove_columns=["id", "title", "context", "question", "answers"]  # Keep only tokenized fields
        )
        print("Datasets tokenized successfully.")

    def run_pipeline(self):
        """Run the full EDA pipeline."""
        self.load_datasets()
        self.preprocess_datasets()
        print("Pipeline executed successfully.")
