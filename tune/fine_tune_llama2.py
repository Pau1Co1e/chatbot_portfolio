from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

# Paths
dataset_path = "custom_dataset_hf"  # Path to the dataset
output_dir = "fine_tuned_llama2"    # Directory to save the fine-tuned model
model_name = "meta-llama/Llama-2-7b-hf"  # Llama 2 base model

# Load Dataset
dataset = load_from_disk(dataset_path)

# Load Pretrained Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    "./converted_llama_hf",
    device_map="auto",  # Automatically offload layers as needed
    torch_dtype="auto"  # Use mixed precision if supported
)
# Tokenization Function
def preprocess_data(examples):
    inputs = [f"Question: {q} Context: {c}" for q, c in zip(examples["question"], examples["context"])]
    targets = examples["answers"]["text"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize Dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    report_to="none",  # Disable reporting to prevent unnecessary logs
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=eda.df_train,
    eval_dataset=eda.df_validate,
    tokenizer=eda.tokenizer,  # Ensure tokenizer is passed here
)

# Fine-Tune the Model
trainer.train()

# Save the Model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model fine-tuned and saved to {output_dir}")