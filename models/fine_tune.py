from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import torch
import os
import evaluate
import pandas as pd

# Mixed Precision Scaler
scaler = torch.amp.GradScaler("cuda")

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Dynamic paths for data loading
SQUAD_DATA_DIR = "datasets/rajpurkar/squad"
CUSTOM_DATA_DIR = "./custom_data"
OUTPUT_DIR = "./fine_tuned_model"
CACHE_DIR = "./cache"

# Load SQuAD datasets dynamically
def load_squad_datasets():
    try:
        # Load SQuAD dataset from Hugging Face Hub
        squad_dataset = load_dataset("squad")

        # Return training and validation datasets
        return squad_dataset["train"], squad_dataset["validation"]
    except Exception as e:
        print(f"Error loading SQuAD dataset: {e}")
        raise

# Custom dataset loader
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

# Combine datasets
def combine_datasets(custom_dataset, squad_dataset):
    combined_dataset = concatenate_datasets([custom_dataset, squad_dataset])
    return combined_dataset.train_test_split(test_size=0.2, seed=42)

# Clean and filter dataset
def clean_and_filter_dataset(dataset):
    def clean_example(example):
        # Replace missing or None 'answers' with default structure
        example["answers"] = example.get("answers") or {"text": [], "answer_start": []}
        example["answers"]["text"] = example["answers"].get("text", [])
        example["answers"]["answer_start"] = example["answers"].get("answer_start", [])
        return example

    def is_valid(example):
        return (
            "answers" in example
            and len(example["answers"]["text"]) > 0
            and len(example["answers"]["answer_start"]) > 0
        )

    dataset = dataset.map(clean_example)
    return dataset.filter(is_valid)

# Preprocess batch function
def preprocess_data(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    start_positions = []
    end_positions = []

    for i, answer in enumerate(examples["answers"]):
        offsets = tokenized["offset_mapping"][i]
        try:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            start_token = next(idx for idx, (s, e) in enumerate(offsets) if s <= start_char < e)
            end_token = next(idx for idx, (s, e) in enumerate(offsets) if s < end_char <= e)
        except StopIteration as e:
            print(f"Token offset issue at index {i}: {e}")
            start_token, end_token = 0, 0  # Default to invalid range

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized.update({"start_positions": start_positions, "end_positions": end_positions})
    return {key: val for key, val in tokenized.items() if key != "offset_mapping"}  # Clean up

# Load or preprocess dataset with caching
def load_or_preprocess_dataset(train_dataset, val_dataset, tokenizer, cache_dir=CACHE_DIR):
    train_cache_path = os.path.join(cache_dir, "train")
    val_cache_path = os.path.join(cache_dir, "val")

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        # Load cached datasets
        print("Loading cached datasets...")
        tokenized_train_dataset = load_from_disk(train_cache_path)
        tokenized_val_dataset = load_from_disk(val_cache_path)
    else:
        # Preprocess datasets
        print("Tokenizing datasets...")
        tokenized_train_dataset = train_dataset.map(
            preprocess_data,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["context", "question", "answers"]],
        )

        tokenized_val_dataset = val_dataset.map(
            preprocess_data,
            batched=True,
            remove_columns=[col for col in val_dataset.column_names if col not in ["context", "question", "answers"]],
        )
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Tokenized validation dataset size: {len(tokenized_val_dataset)}")

        # Save datasets to cache
        os.makedirs(cache_dir, exist_ok=True)
        tokenized_train_dataset.save_to_disk(train_cache_path)
        tokenized_val_dataset.save_to_disk(val_cache_path)

    return tokenized_train_dataset, tokenized_val_dataset

# Compute metricsprint
# Compute metrics
def compute_metrics(metric, predictions, references):
    # Unpack logits
    start_logits, end_logits = predictions

    # Ensure the predictions align with references
    num_predictions = min(len(start_logits), len(references[0]))
    start_logits = start_logits[:num_predictions]
    end_logits = end_logits[:num_predictions]

    # Generate predicted answers
    predicted_answers = []
    for i in range(num_predictions):
        start_index = start_logits[i].argmax()
        end_index = end_logits[i].argmax()

        predicted_answer = tokenizer.decode(
            tokenized_val_dataset[i]["input_ids"][start_index:end_index + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predicted_answers.append({
            "id": str(i),
            "prediction_text": predicted_answer,
        })

    # Generate formatted references
    formatted_references = []
    for i in range(num_predictions):
        context = val_dataset[i]["context"]
        start, end = references[0][i], references[1][i]
        true_answer = context[start:end + 1]
        formatted_references.append({
            "id": str(i),
            "answers": {"text": [true_answer], "answer_start": [start]},
        })

    # Debugging logs
    print(f"Number of predictions: {len(predicted_answers)}, Number of references: {len(formatted_references)}")

    # Return metrics
    return metric.compute(predictions=predicted_answers, references=formatted_references)

# Main execution
if __name__ == "__main__":
    # Custom dataset definition
    datasets = dict()
    metric = evaluate.load("squad")

    # Load datasets
    print("Loading datasets...")
    custom_dataset = load_custom_dataset(datasets)
    squad_train, squad_val = load_squad_datasets()

    # Combine datasets
    combined_dataset = combine_datasets(custom_dataset, squad_train)
    train_dataset, val_dataset = combined_dataset["train"], combined_dataset["test"]

    # Clean and filter datasets
    train_dataset = clean_and_filter_dataset(train_dataset)
    val_dataset = clean_and_filter_dataset(val_dataset)

    # Tokenizer and model
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # Preprocess and cache datasets
    tokenized_train_dataset, tokenized_val_dataset = load_or_preprocess_dataset(
        train_dataset, val_dataset, tokenizer
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps
        save_strategy="steps",
        save_steps=500,  # Save checkpoints every 500 steps
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.005,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        tf32=True
    )

    # Trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda p: compute_metrics(
            metric, p.predictions, p.label_ids
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    trainer.train()

    # Save model and tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved!")

