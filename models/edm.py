import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from data import squad_v2, custom
import pyarrow as pa


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

    def load_datasets(self):
        """Load train and validation data from the specified directory."""
        # Path to the custom JSON file
        json_path = os.path.join(self.dataset_directory, "custom", "custom_train.json")

        # Check if the JSON file exists
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")

        # Load JSON data into a pandas DataFrame
        try:
            df = pd.read_json(json_path, lines=False)  # Adjust `lines` based on your JSON structure
            print("JSON data loaded successfully into DataFrame.")
        except ValueError as e:
            raise ValueError(f"Error reading JSON file: {e}")

        # Convert pandas DataFrame to pyarrow Table
        try:
            table = pa.Table.from_pandas(df)
            print("DataFrame converted to pyarrow Table.")
        except Exception as e:
            raise ValueError(f"Error converting DataFrame to pyarrow Table: {e}")

        # Define the path for the Parquet file
        parquet_path = os.path.join(self.dataset_directory, "custom", "custom_train.parquet")

        # Write the table to a Parquet file
        try:
            pq.write_table(table, parquet_path)
            print(f"Parquet file written successfully at {parquet_path}")
        except Exception as e:
            raise ValueError(f"Error writing Parquet file: {e}")

        # Alternatively, you can use pandas to write Parquet directly:
        # df.to_parquet(parquet_path, index=False)
        # print(f"Parquet file written successfully at {parquet_path} using pandas.")

        # Load the datasets using Hugging Face's `load_dataset`
        try:
            self.datasets = load_dataset(
                "parquet",
                data_files={
                    "train": os.path.join(self.dataset_directory, "squad_v2", "squad_v2_train.parquet"),
                    "validate": os.path.join(self.dataset_directory, "squad_v2", "squad_v2_validate.parquet"),
                    "custom": parquet_path  # Use the newly created Parquet file
                },
            )
            print("Datasets loaded successfully.")
        except Exception as e:
            raise ValueError(f"Error loading datasets: {e}")

        # Optional: Print dataset details
        for split in self.datasets:
            print(f"Loaded {split} split with {len(self.datasets[split])} samples.")

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
        """Convert train and validation data into Pandas DataFrames."""
        self.datasets.set_format(type="pandas")
        self.df_train = self.datasets["train"].to_pandas()
        self.df_validate = self.datasets["validate"].to_pandas()
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
        # Load data
        self.load_datasets()

        # Preprocess data
        self.preprocess_datasets()

        # Check null values
        self.check_null_values(self.df_train, "Train DataFrame")
        self.check_null_values(self.df_validate, "Validation DataFrame")

        # Plot question type frequency
        question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]
        self.plot_question_type_frequency(self.df_train, question_types)

        # Select a random question-context pair
        question, context = self.random_question_context_pair()
        print(f"Question: {question}")
        print(f"Context: {context}")

        # Tokenize and inspect
        inputs = self.tokenize_and_inspect(question, context)
        if inputs:
            print("Tokenization successful!")

