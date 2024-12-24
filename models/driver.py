from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import Dataset
from eda import EDA  # Assuming this is a custom EDA class
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and model
dataset_directory = "../data"
model_name = "deepset/roberta-base-squad2"

# Initialize and run the EDA pipeline
edm = EDA(dataset_directory=dataset_directory, model_name=model_name)
edm.run_pipeline()

# Ensure datasets are preprocessed and tokenized
if "start_positions" not in edm.datasets["train"].column_names or "end_positions" not in edm.datasets["train"].column_names:
    print("Tokenizing and aligning labels...")
    edm.preprocess_datasets()

# Exploratory Data Analysis
def keyword_matching_analysis(df):
    """Analyze keyword overlap between questions and contexts."""
    df['keyword_overlap'] = df.apply(
        lambda x: len(set(x['question'].lower().split()) & set(x['context'].lower().split())), axis=1
    )
    plt.figure(figsize=(10, 6))
    df['keyword_overlap'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Keyword Overlap Between Questions and Contexts")
    plt.xlabel("Number of Matching Keywords")
    plt.ylabel("Frequency")
    plt.show()
    print("Keyword Matching Analysis Completed")

def answerable_vs_unanswerable_analysis(df):
    """Analyze the distribution of answerable vs. unanswerable questions."""
    df['is_answerable'] = df['answers'].apply(lambda x: len(x["text"]) > 0 if "text" in x else False)
    answerable_count = df['is_answerable'].value_counts()
    plt.figure(figsize=(8, 6))
    answerable_count.plot(kind='bar', color=['green', 'red'], alpha=0.7, edgecolor='black')
    plt.title("Answerable vs. Unanswerable Questions")
    plt.xlabel("Answerable (True) or Unanswerable (False)")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=['Unanswerable', 'Answerable'], rotation=0)
    plt.show()
    print(f"Answerable Questions: {answerable_count.get(True, 0)}")
    print(f"Unanswerable Questions: {answerable_count.get(False, 0)}")

def correlation_analysis(df):
    """Plot the correlation matrix of numerical features."""
    df["answer_length"] = df["answers"].apply(lambda x: len(x["text"][0]) if "text" in x and len(x["text"]) > 0 else 0)
    numerical_features = ["answer_length", "start_positions", "end_positions"]
    correlation_matrix = df[numerical_features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

# Run EDA analyses
print("Performing Keyword Matching Analysis...")
keyword_matching_analysis(edm.df_train)

print("Analyzing Answerable vs. Unanswerable Questions...")
answerable_vs_unanswerable_analysis(edm.df_train)

print("Performing Correlation Analysis...")
correlation_analysis(edm.df_train)

# Split datasets for training and validation
train_size = int(0.8 * len(edm.datasets["train"]))
tokenized_train = edm.datasets["train"].select(range(train_size))
tokenized_val = edm.datasets["train"].select(range(train_size, len(edm.datasets["train"])))

# Load the model
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,  # Reduced for memory optimization
    per_device_eval_batch_size=4,  # Reduced for memory optimization
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    logging_steps=500,
    fp16=torch.cuda.is_available(),  # Mixed precision if CUDA is available
    gradient_accumulation_steps=2,  # Simulate larger batch sizes
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating the model...")
evaluation_metrics = trainer.evaluate()
print(f"Evaluation Results: {evaluation_metrics}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Fine-tuned model and tokenizer saved successfully.")

# Test the model with a QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
test_question = "What is your name?"
test_context = "My name is Paul Coleman."
result = qa_pipeline(question=test_question, context=test_context)
print(f"Test Answer: {result['answer']}")
