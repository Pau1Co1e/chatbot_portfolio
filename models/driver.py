from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)
from eda import EDA
import torch
import os

# Define base directories dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(BASE_DIR)
DATA_DIR = os.path.join("data")
print(f"DATA_DIR: {DATA_DIR}")
LOCAL_MODEL_DIR = "./converted_llama_hf"

FINE_TUNED_MODEL_DIR = os.path.join(BASE_DIR, "models", "fine_tuned_model")

# Detect and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load local model and tokenizer
print(f"Loading model from: {LOCAL_MODEL_DIR}")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use appropriate dtype
)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, max_model_length=512)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Pass the tokenizer to EDA
eda = EDA(dataset_directory=DATA_DIR, tokenizer=tokenizer)
eda.run_pipeline()

# Ensure datasets are preprocessed and tokenized
if "start_positions" not in eda.datasets["train"].column_names or "end_positions" not in eda.datasets["train"].column_names:
    print("Tokenizing and aligning labels...")
    eda.preprocess_datasets()
    print("Processed dataset fields:", eda.datasets["train"].column_names)

# Split datasets for training and validation
train_size = int(0.8 * len(eda.datasets["train"]))
tokenized_train = eda.datasets["train"].select(range(train_size))
tokenized_val = eda.datasets["train"].select(range(train_size, len(eda.datasets["train"])))

# Define training arguments
training_args = TrainingArguments(
    output_dir=FINE_TUNED_MODEL_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust for memory limits
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    logging_steps=500,
    fp16=torch.cuda.is_available(),  # Mixed precision if CUDA is available
    gradient_accumulation_steps=4,  # Simulate larger batch sizes
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,  # Preprocessed dataset
    eval_dataset=tokenized_val,    # Preprocessed dataset
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
model.save_pretrained(FINE_TUNED_MODEL_DIR)
tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
print(f"Fine-tuned model and tokenizer saved to {FINE_TUNED_MODEL_DIR}")

# Test the fine-tuned model with a QA pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
test_question = "What is your name?"
test_context = "My name is Paul Coleman."
result = qa_pipeline(f"Question: {test_question} Context: {test_context}")
print(f"Test Answer: {result[0]['generated_text']}")
