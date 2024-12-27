from transformers import AlbertTokenizerFast, AlbertForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
import os
import torch

# Detect device (CUDA, MPS, or CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load datasets
def load_cleaned_datasets():
    merged_dataset_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/final_cleaned_merged_dataset.json"
    data = load_dataset("json", data_files={"train": merged_dataset_path, "validation": merged_dataset_path})
    return data["train"], data["validation"]

# Preprocess function to handle unanswerable questions
def preprocess_for_albert(examples):
    """
    Preprocess examples for ALBERT model.
    Handles unanswerable questions by assigning start_positions and end_positions to 0.
    """
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]

    input_encodings = tokenizer(
        questions, contexts, truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, answer in enumerate(answers):
        offsets = input_encodings["offset_mapping"][i]
        start_char = answer["answer_start"][0] if answer["answer_start"] else -1
        end_char = start_char + len(answer["text"][0]) if start_char != -1 else -1

        # Handle unanswerable questions
        if start_char == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start_index = next((idx for idx, (start, _) in enumerate(offsets) if start == start_char), None)
            token_end_index = next((idx for idx, (_, end) in enumerate(offsets) if end == end_char), None)

            if token_start_index is not None and token_end_index is not None:
                start_positions.append(token_start_index)
                end_positions.append(token_end_index)
            else:
                start_positions.append(0)
                end_positions.append(0)
            print(f"Question: {questions[i]}")
            print(f"Context: {contexts[i]}")
            print(f"Answer: {answer}")
            print(f"Offsets: {offsets}")
            print(f"Start: {start_char}, End: {end_char}")
            print(f"Token Start Index: {token_start_index}, Token End Index: {token_end_index}")

    # Remove offset mapping for model input
    input_encodings.pop("offset_mapping")

    input_encodings["start_positions"] = start_positions
    input_encodings["end_positions"] = end_positions
    return input_encodings

# Load ALBERT tokenizer and model
tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
model = AlbertForQuestionAnswering.from_pretrained("albert-base-v2").to(device)

# Load and preprocess datasets
train_dataset, validation_dataset = load_cleaned_datasets()
tokenized_train = train_dataset.map(preprocess_for_albert, batched=True)
tokenized_validation = validation_dataset.map(preprocess_for_albert, batched=True)

fp16 = torch.cuda.is_available()

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./albert_results",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,  # Simulates batch size of 256
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=fp16,
    max_grad_norm=1.0,
    logging_dir="./logs",
    no_cuda=not torch.cuda.is_available(),  # Automatically handle CUDA availability
)

# Define Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_validation,  # Added eval_dataset
#     data_collator=lambda data: tokenizer.pad(data, return_tensors="pt"),
# )

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator,
)
# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_albert")
tokenizer.save_pretrained("./fine_tuned_albert")
print(f"ALBERT model and tokenizer saved at: {os.path.abspath('./fine_tuned_albert')}")
