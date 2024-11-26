from transformers import AutoTokenizer
from datasets import Dataset
from edm import EDM
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the EDM pipeline with the correct dataset path
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

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Print tokenized data for verification
print(f"Sample Input IDs: {tokenized_dataset['input_ids'][:3]}")
print(f"Sample Attention Mask: {tokenized_dataset['attention_mask'][:3]}")