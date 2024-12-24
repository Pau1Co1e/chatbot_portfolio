import os
import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from data import squad_v2, custom
import pyarrow as pa
import pyarrow.parquet as pq



class EDA:
    def __init__(self, dataset_directory, model_name):
        self.dataset_directory = dataset_directory
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.datasets = None
        self.df_train = None
        self.df_validate = None
        self.df_custom = None

    @staticmethod
    def check_null_values(df, df_name):
        """
        Print null value statistics for a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            df_name (str): Name of the DataFrame.
        """
        print(f"\nChecking for null values in {df_name}...")
        null_counts = df.isnull().sum()
        print(null_counts)
        if null_counts.any():
            print(f"\nWarning: {df_name} contains null values!")
        else:
            print(f"{df_name} has no null values.")

    def random_question_by_type(self, question_type):
        """
        Return a random question of the specified type from the training dataset.

        Args:
            question_type (str): The question type to filter by (e.g., "What").

        Returns:
            str: A random question of the specified type.
        """
        filtered_questions = self.df_train.loc[self.df_train["question"].str.startswith(question_type), "question"]
        if filtered_questions.empty:
            raise ValueError(f"No questions found starting with '{question_type}'.")
        random_question = filtered_questions.sample(n=1).iloc[0]
        print(f"\nRandom '{question_type}' Question: {random_question}")
        return random_question

    def random_question_context_pair(self):
        """
        Select a random question-context pair from the training dataset.

        Returns:
            tuple: A question and its corresponding context.
        """
        random_row = self.df_train.sample(n=1).iloc[0]
        question, context = random_row["question"], random_row["context"]
        print(f"\nRandom Question-Context Pair:\nQuestion: {question}\nContext: {context}")
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
        print(f"Keyword Overlap: {overlap}")
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
            print("The question and context appear to be unrelated.")
            return None

        tokenized_output = self.tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        print(f"\nTokenized Question-Context Pair:")
        print(f"Input IDs: {tokenized_output['input_ids']}")
        print(f"Attention Mask: {tokenized_output['attention_mask']}")
        return tokenized_output


    def load_datasets(self):
        # Paths
        squad_train_path = os.path.join(self.dataset_directory, "squad_v2", "squad_v2_train.parquet")
        squad_validate_path = os.path.join(self.dataset_directory, "squad_v2", "squad_v2_validate.parquet")
        custom_path = os.path.join(self.dataset_directory, "custom", "custom_train_restructured.parquet")

        # Load and combine datasets
        datasets = DatasetDict({
            "train": Dataset.from_parquet(squad_train_path),
            "validate": Dataset.from_parquet(squad_validate_path),
            "custom": Dataset.from_parquet(custom_path)
        })
        print("Datasets loaded successfully.")
        for split, dataset in datasets.items():
            print(f"Loaded '{split}' split with {len(dataset)} samples.")
        self.datasets = datasets

    def save_datasets(self, output_dir="tokenized_datasets"):
        os.makedirs(output_dir, exist_ok=True)
        self.df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        self.df_validate.to_csv(os.path.join(output_dir, "validate.csv"), index=False)
        self.df_custom.to_csv(os.path.join(output_dir, "custom.csv"), index=False)
        print(f"Tokenized DataFrames saved to {output_dir}")

    def save_tokenized_datasets(self, output_dir="tokenized_datasets"):
        """Save tokenized datasets to a specified directory and log saved file paths."""
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "tokenized_train.parquet")
        validate_path = os.path.join(output_dir, "tokenized_validate.parquet")
        custom_path = os.path.join(output_dir, "tokenized_custom.parquet")

        self.df_train.to_parquet(train_path, index=False)
        self.df_validate.to_parquet(validate_path, index=False)
        self.df_custom.to_parquet(custom_path, index=False)

        print(f"Tokenized DataFrames saved to:\n- {train_path}\n- {validate_path}\n- {custom_path}")

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
        """
        Tokenize and preprocess datasets, then convert them to Pandas DataFrames.
        """

        def tokenize_and_align_labels(examples):
            # Tokenize question and context
            tokenized_inputs = self.tokenizer(
                examples["question"],
                examples["context"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            # Initialize start and end positions
            start_positions = []
            end_positions = []

            # Align tokenized inputs with answer spans
            for i, answer in enumerate(examples["answers"]):
                if len(answer["answer_start"]) == 0:  # No answer
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    start_char = answer["answer_start"][0]
                    end_char = start_char + len(answer["text"][0])

                    # Map character positions to token positions
                    token_start = tokenized_inputs.char_to_token(i, start_char)
                    token_end = tokenized_inputs.char_to_token(i, end_char - 1)

                    # Handle cases where mapping fails
                    if token_start is None:
                        token_start = self.tokenizer.model_max_length
                    if token_end is None:
                        token_end = self.tokenizer.model_max_length

                    start_positions.append(token_start)
                    end_positions.append(token_end)

            tokenized_inputs["start_positions"] = start_positions
            tokenized_inputs["end_positions"] = end_positions
            return tokenized_inputs

        print("Starting tokenization with label alignment...")
        self.datasets = self.datasets.map(tokenize_and_align_labels, batched=True)
        print("Datasets tokenized and aligned successfully.")

        self.datasets.set_format(type="pandas")

        # Convert splits to DataFrames
        try:
            self.df_train = self.datasets["train"].to_pandas()
            self.df_validate = self.datasets["validate"].to_pandas()
            self.df_custom = self.datasets["custom"].to_pandas()
            print("Datasets converted to DataFrames.")
        except KeyError as e:
            raise RuntimeError(f"Error converting datasets to DataFrames: {e}")

    def inspect_data(self):
        """Print sample rows from datasets for inspection."""
        print("\nSample rows from Train DataFrame:")
        print(self.df_train.head())
        print("\nSample rows from Validate DataFrame:")
        print(self.df_validate.head())
        print("\nSample rows from Custom DataFrame:")
        print(self.df_custom.head())

    def run_pipeline(self):
        """
        Execute the full pipeline for data loading, preprocessing, and analysis.
        """
        # Load datasets
        self.load_datasets()

        # Preprocess datasets
        self.preprocess_datasets()

        # Verify DataFrame conversion
        if self.df_train is None or self.df_validate is None or self.df_custom is None:
            raise RuntimeError("One or more DataFrames are not properly initialized after preprocessing.")

        # Check for null values
        self.check_null_values(self.df_train, "Train DataFrame")
        self.check_null_values(self.df_validate, "Validation DataFrame")
        self.check_null_values(self.df_custom, "Custom DataFrame")

        # Additional processing or analysis steps
        print("Pipeline executed successfully.")

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

# if __name__ == "__main__":
#     edm = EDM(
#         dataset_directory="/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data",
#         model_name="deepset/roberta-base-squad2"
#     )
#     edm.run_pipeline()
#     edm.plot_question_type_frequency(df=edm.df_custom, question_types=["What"])
#
# if __name__ == "__main__":
#     edm = EDM(
#         dataset_directory="/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data",
#         model_name="deepset/roberta-base-squad2"
#     )
#     edm.run_pipeline()
