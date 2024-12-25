from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from eda import EDA
import torch
import os
import torch.nn as nn

# Define base directories dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(BASE_DIR)
DATA_DIR = os.path.join("data")
print(f"DATA_DIR: {DATA_DIR}")
MODEL_NAME = "distilbert-base-uncased-distilled-squad"  # Use a smaller model

FINE_TUNED_MODEL_DIR = os.path.join(BASE_DIR, "models", "fine_tuned_model")

# Detect and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pass the tokenizer to EDA
eda = EDA(dataset_directory=DATA_DIR, tokenizer=tokenizer)
eda.run_pipeline()

# Split datasets for training and validation
train_size = int(0.8 * len(eda.datasets["train"]))
tokenized_train = eda.datasets["train"].select(range(train_size))
tokenized_val = eda.datasets["train"].select(range(train_size, len(eda.datasets["train"])))

# Define a custom trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]

        # Compute the loss using CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return (total_loss, outputs) if return_outputs else total_loss

# Define training arguments
training_args = TrainingArguments(
    output_dir=FINE_TUNED_MODEL_DIR,
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,  # Adjust for memory limits
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    logging_steps=100,
    fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False
)

# Initialize the custom trainer
trainer = CustomTrainer(
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
model.save_pretrained(FINE_TUNED_MODEL_DIR)
tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
print(f"Fine-tuned model and tokenizer saved to {FINE_TUNED_MODEL_DIR}")

# Test the fine-tuned model with a QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
test_question = "What is your name?"
test_context = "My name is Paul Coleman."
result = qa_pipeline(question=test_question, context=test_context)
print(f"Test Answer: {result['answer']}")
