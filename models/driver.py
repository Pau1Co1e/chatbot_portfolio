from transformers import AutoTokenizer
from datasets import Dataset
from edm import EDM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the EDM pipeline
#edm = EDM(dataset_directory="/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data", model_name="deepset/roberta-base-squad2")
edm = EDM(
    dataset_directory="/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio/data",
    model_name="deepset/roberta-base-squad2"
)
edm.run_pipeline()

# Ensure data are loaded and preprocessed
edm.load_datasets()
edm.preprocess_datasets()

# Access the train DataFrame
df_train = edm.df_train

# Create a Hugging Face Dataset from the Pandas DataFrame
dataset = Dataset.from_pandas(df_train)

# Select a random question of type 'What'
question = edm.random_question_by_type("What")

# Select a random context from the training dataset
context = df_train["context"].sample(n=1).iloc[0]

# Print the selected question and context
print(f"Question: {question}")
print(f"Context: {context}")


# Tokenize the question and context
def preprocess_function(examples):
    return edm.tokenizer(
        examples["question"],
        examples["context"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

def fix_answers_schema(row):
    """
    Ensure the answers column aligns with the Hugging Face schema.

    Args:
        row (dict): A row from the DataFrame.

    Returns:
        dict: Fixed answers' schema.
    """
    if not isinstance(row, dict):
        # Handle cases where the row is not a dictionary
        return {"text": [], "answer_start": []}

    # Get the answers field safely
    answers = row.get("answers", {})

    # Ensure 'answers' is a dictionary and contains 'text' and 'answer_start'
    if not isinstance(answers, dict):
        return {"text": [], "answer_start": []}

    return {
        "text": answers.get("text", []),
        "answer_start": answers.get("answer_start", [])
    }

# Apply the fix to the DataFrame
import pandas as pd

# Load the custom dataset
df_custom = pd.read_parquet("../data/custom/custom_train_restructured.parquet")

# Ensure 'answers' column exists and is properly formatted
if "answers" not in df_custom.columns:
    df_custom["answers"] = [{}] * len(df_custom)  # Add default empty answers for missing rows

# Apply the fix function to every row in the 'answers' column
df_custom["answers"] = df_custom["answers"].apply(fix_answers_schema)

# Save the fixed DataFrame
df_custom.to_parquet("../data/custom/custom_train_restructured_fixed.parquet", index=False)

print("Fixed 'answers' schema and saved the DataFrame.")

# Reload the fixed DataFrame
df_custom_fixed = pd.read_parquet("../data/custom/custom_train_restructured_fixed.parquet")

# Verify the structure of 'answers'
print(df_custom_fixed["answers"].head())

# Proceed with tokenization and further steps
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(f"Sample Input IDs: {tokenized_dataset['input_ids'][:3]}")
print(f"Sample Attention Mask: {tokenized_dataset['attention_mask'][:3]}")
