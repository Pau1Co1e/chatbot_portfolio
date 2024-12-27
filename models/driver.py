from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering
)
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch
import os

# Detect device (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_or_preprocess_dataset(train_path, valid_path, tokenizer, map_fn):
    train_out_dir = "./train_processed"
    valid_out_dir = "./valid_processed"

    # Train dataset
    try:
        if os.path.exists(train_out_dir) and os.path.isdir(train_out_dir):
            print(f"Loading preprocessed train dataset from {train_out_dir} ...")
            train_dataset = load_from_disk(train_out_dir)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        print("Preprocessing train dataset...")
        raw_train = load_dataset("json", data_files={"train": train_path})["train"]
        train_dataset = raw_train.map(
            lambda x: map_fn(x, tokenizer),
            batched=True,
            num_proc=8
        )
        print(f"Saving preprocessed train dataset to {train_out_dir} ...")
        train_dataset.save_to_disk(train_out_dir)

    # Validation dataset
    try:
        if os.path.exists(valid_out_dir) and os.path.isdir(valid_out_dir):
            print(f"Loading preprocessed validation dataset from {valid_out_dir} ...")
            validation_dataset = load_from_disk(valid_out_dir)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        print("Preprocessing validation dataset...")
        raw_valid = load_dataset("json", data_files={"validation": valid_path})["validation"]
        validation_dataset = raw_valid.map(
            lambda x: map_fn(x, tokenizer),
            batched=True,
            num_proc=8
        )
        print(f"Saving preprocessed validation dataset to {valid_out_dir} ...")
        validation_dataset.save_to_disk(valid_out_dir)

    return train_dataset, validation_dataset

def preprocess_for_albert(examples, tokenizer):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]

    input_encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, answer in enumerate(answers):
        offsets = input_encodings["offset_mapping"][i]

        # Use length checks to avoid ambiguous boolean checks
        if len(answer["answer_start"]) == 0 or len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Another optional check to ensure valid boundaries
        if start_char < 0 or end_char < 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start_index = None
        token_end_index = None

        # Find the token index for the start_char
        for idx, (offset_start, offset_end) in enumerate(offsets):
            if offset_start <= start_char < offset_end:
                token_start_index = idx
                break

        # Find the token index for the end_char
        for idx, (offset_start, offset_end) in enumerate(offsets):
            if offset_start < end_char <= offset_end:
                token_end_index = idx
                break

        if (token_start_index is None or
            token_end_index is None or
            token_start_index > token_end_index):
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)

    input_encodings.pop("offset_mapping")
    input_encodings["start_positions"] = start_positions
    input_encodings["end_positions"] = end_positions
    return input_encodings

# Load SQuAD metric
# metric = evaluate.load("squad")
# def compute_metrics(pred, eval_dataset, tokenizer):
#     """
#     Compute metrics for a QA model, including Exact Match (EM) and F1 Score.
#
#     Args:
#         pred (transformers.EvalPrediction): Predictions from the Hugging Face Trainer.
#             Contains 'predictions' (start and end logits) and 'label_ids' (true start and end indices).
#         eval_dataset (Dataset): The evaluation dataset used during validation.
#         tokenizer: Hugging Face tokenizer.
#
#     Returns:
#         dict: Dictionary containing EM and F1 scores.
#     """
#     # Extract start and end logits and move them to the CPU
#     start_logits = torch.tensor(pred.predictions[0]).to(device).cpu().numpy()
#     end_logits = torch.tensor(pred.predictions[1]).to(device).cpu().numpy()
#
#     # Extract true start and end positions from the labels
#     true_starts, true_ends = pred.label_ids
#
#     # Convert logits to predicted indices
#     pred_starts = np.argmax(start_logits, axis=1)
#     pred_ends = np.argmax(end_logits, axis=1)
#
#     # Prepare predictions and references for SQuAD metric
#     predictions = []
#     references = []
#
#     for i, example in enumerate(eval_dataset):
#         # Predicted text
#         input_ids = example["input_ids"]
#         pred_text = pred_offset_to_text(pred_starts[i], pred_ends[i], input_ids, tokenizer)
#
#         # Reference (true) text
#         ref_texts = [tokenizer.decode(input_ids[true_starts[i]:true_ends[i] + 1], skip_special_tokens=True)]
#
#         predictions.append({"id": str(i), "prediction_text": pred_text})
#         references.append({"id": str(i), "answers": {"text": ref_texts}})
#
#     # Compute the metrics using the SQuAD metric
#     result = metric.compute(predictions=predictions, references=references)
#
#     return {
#         "exact_match": result["exact_match"],
#         "f1": result["f1"],
#     }

def pred_offset_to_text(start_idx, end_idx, input_ids, tokenizer):
    """
    Convert start and end indices into text.

    Args:
        start_idx (int): Predicted start index.
        end_idx (int): Predicted end index.
        input_ids (list): Tokenized input IDs.
        tokenizer: Hugging Face tokenizer used for decoding.

    Returns:
        str: Decoded text for the predicted span.
    """
    if start_idx > end_idx:
        return ""

    span_tokens = input_ids[start_idx:end_idx + 1]
    return tokenizer.decode(span_tokens, skip_special_tokens=True)


# Main training script
def main():
    # 1. Initialize tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", truncation=True, max_length=512)
    model = AutoModelForQuestionAnswering.from_pretrained("albert-base-v2").to(device)

    # 2. Load or preprocess dataset (single pass)
    train_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/final_cleaned_merged_dataset.json"
    valid_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/data/final_cleaned_merged_dataset.json"

    train_dataset, validation_dataset = load_or_preprocess_dataset(
        train_path,
        valid_path,
        tokenizer,
        preprocess_for_albert
    )

    # 3. Convert to torch format
    train_dataset.set_format("torch")
    validation_dataset.set_format("torch")

    # 4. Define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # 5. Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4
    )

    eval_dataloader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4
    )

    # 6. Define training arguments
    training_args = TrainingArguments(
        output_dir="./albert_results",
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=500,
        save_total_limit=2,
        learning_rate=1e-5,
        per_device_train_batch_size=48,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        tf32=True,
        max_grad_norm=1.0,
        logging_dir="./logs",
        no_cuda=not torch.cuda.is_available(),
        logging_steps=2000
    )

    # # 7. Wrap compute_metrics
    # def wrapped_compute_metrics(pred):
    #     return compute_metrics(pred, eval_dataset=validation_dataset, tokenizer=tokenizer)

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,
        data_collator=data_collator,
        # compute_metrics=wrapped_compute_metrics
    )

    # 9. Train
    trainer.train()

    # 10. Optional: Save final checkpoint and dataset
    trainer.save_model("./fine_tuned_albert")        # same as model.save_pretrained(...)
    tokenizer.save_pretrained("./fine_tuned_albert")

    # If you really want to store the final dataset, do it here
    train_dataset.save_to_disk("/results/albert_train.cache", num_proc=8)

    print(f"ALBERT model and tokenizer saved at: {os.path.abspath('./fine_tuned_albert')}")

if __name__ == "__main__":
    main()
