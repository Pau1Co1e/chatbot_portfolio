from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from eda import EDA
import torch
import pandas as pd
import os

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

# Test: Select a random question and context
question = edm.random_question_by_type("What")
context = edm.df_train["context"].sample(n=1).iloc[0]
print(f"Test Question: {question}")
print(f"Test Context: {context}")

# Exploratory Data Analysis: Plot question type frequency
question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]
edm.plot_question_type_frequency(edm.df_train, question_types)

# Inspect a random question-context pair
question, context = edm.random_question_context_pair()
print(f"Inspecting Random Pair:\nQuestion: {question}\nContext: {context}")
edm.tokenize_and_inspect(question, context)
# Ensure custom dataset 'answers' schema is fixed
custom_dataset_path = os.path.join(dataset_directory, "custom/custom_train_restructured.parquet")
fixed_custom_dataset_path = os.path.join(dataset_directory, "custom/custom_train_restructured_fixed.parquet")

if not os.path.exists(fixed_custom_dataset_path):
    print("Fixing 'answers' schema for custom dataset...")
    df_custom = pd.read_parquet(custom_dataset_path)

    # Ensure 'answers' column exists and apply schema fix
    if "answers" not in df_custom.columns:
        df_custom["answers"] = [{}] * len(df_custom)

    def fix_answers_schema(row):
        """Ensure 'answers' column aligns with Hugging Face schema."""
        if not isinstance(row, dict):
            return {"text": [], "answer_start": []}
        answers = row.get("answers", {})
        if not isinstance(answers, dict):
            return {"text": [], "answer_start": []}
        return {
            "text": answers.get("text", []),
            "answer_start": answers.get("answer_start", []),
        }

    df_custom["answers"] = df_custom["answers"].apply(fix_answers_schema)
    df_custom.to_parquet(fixed_custom_dataset_path, index=False)
    print("Fixed 'answers' schema and saved the DataFrame.")

# Split datasets for training and validation
train_size = int(0.8 * len(edm.datasets["train"]))
tokenized_train = edm.datasets["train"].select(range(train_size))
tokenized_val = edm.datasets["train"].select(range(train_size, len(edm.datasets["train"])))

# Load the model
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    logging_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=edm.tokenizer,
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
edm.tokenizer.save_pretrained("./fine_tuned_model")
print("Fine-tuned model and tokenizer saved successfully.")

# Test the model with a QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=edm.tokenizer)
result = qa_pipeline(question=question, context=context)
print(f"Test Answer: {result['answer']}")