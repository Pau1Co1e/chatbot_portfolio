import os

import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets, load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from data import squad_v2, custom
import pyarrow as pa
import pyarrow.parquet as pq


class EDM:
    def __init__(self, dataset_directory, model_name):
        """
        Initialize the EDM class.

        Args:
            dataset_directory (str): Path to the directory containing data.
            model_name (str): Hugging Face model name for tokenization and analysis.
        """
        self.dataset_directory = dataset_directory
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.datasets = None
        self.df_train = None
        self.df_validate = None
        self.df_custom = None

    def load_datasets(self):
        """Load train and validation data from the specified directory."""
        # Paths
        restructured_parquet_path = os.path.join(self.dataset_directory, "custom", "custom_train_restructured.parquet")
        squad_train_path = os.path.join(self.dataset_directory, "squad_v2", "squad_v2_train.parquet")
        squad_validate_path = os.path.join(self.dataset_directory, "squad_v2", "squad_v2_validate.parquet")

        # Ensure all paths exist
        if not os.path.exists(restructured_parquet_path):
            raise FileNotFoundError(f"Custom Parquet file not found at {restructured_parquet_path}")
        if not os.path.exists(squad_train_path):
            raise FileNotFoundError(f"SQuAD train file not found at {squad_train_path}")
        if not os.path.exists(squad_validate_path):
            raise FileNotFoundError(f"SQuAD validation file not found at {squad_validate_path}")

        # Load datasets using Hugging Face's `load_dataset`
        try:
            self.datasets = load_dataset(
                "parquet",  # Format
                data_files={
                    "train": squad_train_path,
                    "validate": squad_validate_path,
                    "custom": restructured_parquet_path
                }
            )
            print("Datasets loaded successfully.")
        except Exception as e:
            raise ValueError(f"Error loading datasets with Hugging Face: {e}")

        # Log dataset information
        for split, dataset in self.datasets.items():
            print(f"Loaded '{split}' split with {len(dataset)} samples.")

        return self.datasets

    @staticmethod
    def combine_datasets(custom_dataset, squad_dataset):
        combined_dataset = concatenate_datasets([custom_dataset, squad_dataset])
        return combined_dataset.train_test_split(test_size=0.2, seed=42)

    @staticmethod
    def load_custom_dataset(data):
        flattened_data = []
        for category, entries in data.items():
            for entry in entries:
                for text, start in zip(entry["answers"]["text"], entry["answers"]["answer_start"]):
                    flattened_data.append({
                        "category": category,
                        "question": entry["question"],
                        "context": entry["context"],
                        "answer_text": text,
                        "answer_start": start
                    })
        return Dataset.from_pandas(pd.DataFrame(flattened_data))

    def preprocess_datasets(self):
        """Tokenize datasets and convert to Pandas DataFrames."""

        # Tokenize the dataset using Hugging Face tokenizer
        def tokenize_function(examples):
            return self.tokenizer(
                examples["question"],
                examples["context"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        # Tokenize each split
        self.datasets = self.datasets.map(tokenize_function, batched=True)
        print("Datasets tokenized successfully.")

        # Access tokenized fields for inspection (before converting to Pandas)
        token_lengths = [len(example["input_ids"]) for example in self.datasets["custom"]]
        print(f"Token Lengths for Custom Dataset: {token_lengths}")

        # Convert to Pandas DataFrame for further analysis
        self.datasets.set_format(type="pandas")
        self.df_train = self.datasets["train"].to_pandas()
        self.df_validate = self.datasets["validate"].to_pandas()
        self.df_custom = self.datasets["custom"].to_pandas()
        print("Datasets converted to DataFrames.")

    @staticmethod
    def check_null_values(df, df_name):
        """
        Print null value statistics for a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            df_name (str): Name of the DataFrame (for display).
        """
        print(f"\nNull Values in {df_name}:")
        print(df.isnull().sum())

    @staticmethod
    def plot_question_type_frequency(df, question_types):
        """
        Plot the frequency of specific question types.

        Args:
            df (pd.DataFrame): The DataFrame containing the 'question' column.
            question_types (list): List of question types to analyze.
        """
        counts = {q: df["question"].str.startswith(q).sum() for q in question_types}
        pd.Series(counts).sort_values().plot.barh()
        plt.title("Frequency of Question Types")
        plt.xlabel("Count")
        plt.ylabel("Question Type")
        plt.show()

    def random_question_by_type(self, question_type):
        """
        Return a random question that starts with the specified question type.

        Args:
            question_type (str): The question type to filter (e.g., "What").
        """
        filtered_questions = self.df_train.loc[self.df_train["question"].str.startswith(question_type), "question"]
        if filtered_questions.empty:
            raise ValueError(f"No questions found starting with '{question_type}'.")
        return filtered_questions.sample(n=1).iloc[0]

    def random_question_context_pair(self):
        """
        Select a random question-context pair from the training dataset.

        Returns:
            tuple: A question and its corresponding context.
        """
        random_row = self.df_train.sample(n=1).iloc[0]
        question = random_row["question"]
        context = random_row["context"]
        return question, context

    @staticmethod
    def is_question_related_to_context(question, context, threshold=1):
        """
        Check if a question is related to the context by matching keywords.

        Args:
            question (str): The question string.
            context (str): The context string.
            threshold (int): Minimum number of overlapping keywords.

        Returns:
            bool: True if related, False otherwise.
        """
        question_tokens = set(question.lower().split())
        context_tokens = set(context.lower().split())
        overlap = question_tokens.intersection(context_tokens)
        return len(overlap) >= threshold

    def tokenize_and_inspect(self, question, context):
        """
        Tokenize question and context, then inspect the tokenized output.

        Args:
            question (str): The question string.
            context (str): The context string.

        Returns:
            dict: Tokenized output with 'input_ids' and 'attention_mask'.
        """
        if not self.is_question_related_to_context(question, context):
            print("The question and context are unrelated!")
            return None

        inputs = self.tokenizer(
            question,
            context,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        print(f"Input IDs: \n{inputs['input_ids']}")
        print(f"Attention Mask: \n{inputs['attention_mask']}")
        return inputs

    def run_pipeline(self):
        """
        Execute the full pipeline for data loading, preprocessing, and analysis.
        """
        # Load datasets
        self.load_datasets()

        # Tokenize and preprocess datasets
        self.preprocess_datasets()

        # Check for null values
        self.check_null_values(self.df_train, "Train DataFrame")
        self.check_null_values(self.df_validate, "Validation DataFrame")
        self.check_null_values(self.df_custom, "Custom DataFrame")

        # Populate `id` and `title` with placeholder values for the custom dataset
        edm.df_custom["id"] = edm.df_custom["id"].fillna("custom-id-placeholder")
        edm.df_custom["title"] = edm.df_custom["title"].fillna("custom-title-placeholder")

        # Plot question type frequency
        question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]
        self.plot_question_type_frequency(self.df_train, question_types)

        # Select a random question-context pair
        question, context = self.random_question_context_pair()
        print(f"Question: {question}")
        print(f"Context: {context}")

        # Tokenize and inspect a single pair
        inputs = self.tokenize_and_inspect(question, context)
        if inputs:
            print("Tokenization successful!")


if __name__ == "__main__":
    edm = EDM(dataset_directory="/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data/", model_name="deepset/roberta-base-squad2")
    edm.run_pipeline()
