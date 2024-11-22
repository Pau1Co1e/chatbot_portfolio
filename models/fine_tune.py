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
import shutil


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
        # Load the SQuAD dataset directly from Hugging Face, bypassing any cached files
        print("Loading SQuAD dataset from Hugging Face...")
        squad_dataset = load_dataset("squad", cache_dir=None)  # Do not use caching
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
        example["answers"] = example.get("answers") or {"text": [], "answer_start": []}
        example["answers"]["text"] = example["answers"].get("text", [])
        example["answers"]["answer_start"] = example["answers"].get("answer_start", [])
        return example

    def is_valid(example):
        return (
            "answers" in example and
            len(example["answers"]["text"]) > 0 and
            len(example["answers"]["answer_start"]) > 0
        )

    # Clean dataset
    print(f"Dataset size before cleaning: {len(dataset)}")
    dataset = dataset.map(clean_example, load_from_cache_file=False)
    print(f"Dataset size after cleaning: {len(dataset)}")

    # Filter dataset
    dataset = dataset.filter(is_valid, load_from_cache_file=False)
    print(f"Dataset size after filtering: {len(dataset)}")
    return dataset

# Preprocess batch function
import numpy as np

def preprocess_data(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
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
        except StopIteration:
            print(
                f"Token alignment failed for example {i}: "
                f"Question: '{examples['question'][i]}', "
                f"Answer: '{answer['text']}', "
                f"Context: '{examples['context'][i][:200]}...' "  # Truncate context for readability
            )
            start_token, end_token = 0, 0

        start_positions.append(start_token)
        end_positions.append(end_token)

    # Convert tensors to lists to ensure compatibility with Hugging Face datasets
    tokenized = {key: val.tolist() if isinstance(val, torch.Tensor) else val for key, val in tokenized.items()}
    tokenized.update({
        "start_positions": start_positions,
        "end_positions": end_positions,
    })
    return tokenized

# Load or preprocess dataset with caching
def load_or_preprocess_dataset(train_dataset, val_dataset, tokenizer, cache_dir=CACHE_DIR, force_retokenize=False):
    train_cache_path = os.path.join(cache_dir, "train")
    val_cache_path = os.path.join(cache_dir, "val")

    # Delete cache if re-tokenization is forced
    if force_retokenize:
        if os.path.exists(train_cache_path):
            print("Deleting cached training dataset...")
            shutil.rmtree(train_cache_path)
        if os.path.exists(val_cache_path):
            print("Deleting cached validation dataset...")
            shutil.rmtree(val_cache_path)

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path) and not force_retokenize:
        print("Loading cached datasets...")
        tokenized_train_dataset = load_from_disk(train_cache_path)
        tokenized_val_dataset = load_from_disk(val_cache_path)
    else:
        print(f"Training dataset size before tokenization: {len(train_dataset)}")
        tokenized_train_dataset = train_dataset.map(
            lambda x: preprocess_data(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        print(f"Training dataset size after tokenization: {len(tokenized_train_dataset)}")

        print(f"Validation dataset size before tokenization: {len(val_dataset)}")
        tokenized_val_dataset = val_dataset.map(
            lambda x: preprocess_data(x, tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names,
        )
        print(f"Validation dataset size after tokenization: {len(tokenized_val_dataset)}")

        os.makedirs(cache_dir, exist_ok=True)
        tokenized_train_dataset.save_to_disk(train_cache_path)
        tokenized_val_dataset.save_to_disk(val_cache_path)

    print(f"Tokenized train dataset size: {len(tokenized_train_dataset)}")
    print(f"Tokenized validation dataset size: {len(tokenized_val_dataset)}")

    return tokenized_train_dataset, tokenized_val_dataset

# Compute metrics
def compute_metrics(metric, predictions, references):
    # Unpack logits
    start_logits, end_logits = predictions

    # Ensure alignment between predictions and references
    num_references = min(len(start_logits), len(references[0]), len(val_dataset))

    predicted_answers = []
    for i in range(num_references):
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

    formatted_references = [
        {
            "id": str(i),
            "answers": {
                "text": [val_dataset[i]["context"][references[0][i]:references[1][i] + 1]],
                "answer_start": [references[0][i]],
            },
        }
        for i in range(num_references)
    ]

    print(f"Number of predictions: {len(predicted_answers)}, Number of references: {len(formatted_references)}")
    return metric.compute(predictions=predicted_answers, references=formatted_references)




# Main execution
if __name__ == "__main__":
    datasets = {
    "education_and_certifications": [
        {
            "question": "What is your educational background?",
            "context": "Paul Coleman holds a Master of Science in Computer Science from Utah Valley University (expected August 2025) and a Bachelor of Science in Computer Science (graduated August 2022) with a GPA of 3.26.",
            "answers": {
                "text": ["Master of Science in Computer Science from Utah Valley University",
                         "Bachelor of Science in Computer Science"],
                "answer_start": [19, 110]
            }
        },
        {
            "question": "What certifications do you have?",
            "context": "Paul Coleman has a Programmer Certification from Utah Valley University (2020), was on the Dean's List in 2020 and 2021, and holds a CompTIA A+ Certification from Dell Inc.",
            "answers": {
                "text": ["Programmer Certification", "CompTIA A+ Certification"],
                "answer_start": [19, 120]
            }
        }
    ],
    "professional_skills": [
        {
            "question": "What are your professional skills?",
            "context": "Paul Coleman is skilled in AI & Machine Learning, model development, algorithm design, NLP, web development with Flask and JavaScript, and scalable systems design. His programming expertise includes Python, C#, and Java.",
            "answers": {
                "text": ["AI & Machine Learning", "Flask and JavaScript", "Python, C#, and Java"],
                "answer_start": [19, 80, 147]
            }
        }
    ],
    "work_experience": [
        {
            "question": "Where have you worked recently?",
            "context": "Paul Coleman worked as a Full Stack Software Engineer at ResNexus from September 2023 to December 2023. He developed backend APIs and frontend components using the Microsoft tech stack for hotel and rental management products.",
            "answers": {
                "text": ["Full Stack Software Engineer at ResNexus"],
                "answer_start": [18]
            }
        }
    ],
    "volunteer_experience": [
        {
            "question": "What volunteer experience do you have?",
            "context": "Paul Coleman organized a local donation event for the Northern Utah Habitat for Humanity Clothing Drive in June 2014, supporting children and families in the Dominican Republic.",
            "answers": {
                "text": ["Northern Utah Habitat for Humanity Clothing Drive"],
                "answer_start": [34]
            }
        }
    ],
    "ai_concepts": [
        {
            "question": "How has Paul applied Interpretable Machine Learning in projects?",
            "context": "Paul Coleman has experience with Interpretable Machine Learning techniques like LIME and SHAP. He has applied these methods to financial prediction models to explain outputs to stakeholders in the FinTech industry.",
            "answers": {
                "text": ["Interpretable Machine Learning techniques like LIME and SHAP"],
                "answer_start": [42],
            }
        },
        {
            "question": "What are Paul’s contributions to Transfer Learning?",
            "context": "Paul has implemented Transfer Learning in NLP applications, utilizing pre-trained language models such as BERT for sentiment analysis and chatbots. He optimized transfer learning workflows for rapid deployment in AI systems.",
            "answers": {
                "text": ["Transfer Learning in NLP applications"],
                "answer_start": [24],
            }
        },
        {
            "question": "What is Paul’s experience with Reinforcement Learning?",
            "context": "Paul worked on Reinforcement Learning algorithms for autonomous mobile robots during his time at Utah Valley University. He focused on developing agents capable of decision-making in dynamic environments using Q-Learning and Deep Q-Networks.",
            "answers": {
                "text": ["Reinforcement Learning algorithms for autonomous mobile robots"],
                "answer_start": [17],
            }
        },
        {
            "question": "How has Paul leveraged NLP in his projects?",
            "context": "Paul specializes in Natural Language Processing, having developed conversational AI systems using Transformer-based models like GPT. His projects include chatbots for customer support and advanced text summarization tools.",
            "answers": {
                "text": ["Natural Language Processing"],
                "answer_start": [17],
            }
        },
        {
            "question": "What role has Anomaly Detection played in Paul’s work?",
            "context": "Paul implemented Anomaly Detection techniques in cybersecurity applications to identify fraud and unusual network behavior. He used autoencoders and statistical methods to improve system reliability.",
            "answers": {
                "text": ["Anomaly Detection techniques in cybersecurity applications"],
                "answer_start": [17],
            }
        },
        {
            "question": "How has Paul used Time Series Analysis?",
            "context": "Paul has applied Time Series Analysis in financial forecasting, developing models to predict stock price movements and economic trends. He has experience using ARIMA, LSTM networks, and seasonal decomposition techniques.",
            "answers": {
                "text": ["Time Series Analysis in financial forecasting"],
                "answer_start": [17],
            }
        },
        {
            "question": "What is Paul’s expertise in Computer Vision?",
            "context": "Paul has developed Computer Vision models for object detection and image segmentation. His projects include building a vision-based quality inspection system for manufacturing and applying OpenCV and TensorFlow for object tracking.",
            "answers": {
                "text": ["Computer Vision models for object detection and image segmentation"],
                "answer_start": [17],
            }
        },
        {
            "question": "What types of Neural Networks has Paul worked with?",
            "context": "Paul has worked extensively with convolutional neural networks (CNNs) for image analysis and recurrent neural networks (RNNs) for sequential data. His expertise includes designing custom architectures for real-world applications.",
            "answers": {
                "text": ["convolutional neural networks (CNNs) for image analysis"],
                "answer_start": [35],
            }
        },
        {
            "question": "What is Paul’s experience with Autonomous Mobile Robots?",
            "context": "Paul led a project on a pair of humanoid Autonomous Mobile Robots at UVU, focusing on navigation and decision-making using reinforcement learning and sensor fusion techniques. He optimized the robots for university recruitment events, a promotional film, and a gala fund raiser.",
            "answers": {
                "text": ["Autonomous Mobile Robots at UVU"],
                "answer_start": [26],
            }
        },
        {
            "question": "What are Paul’s accomplishments in Conversational AI?",
            "context": "Paul developed a portfolio chatbot using a custom-trained DistilBERT model integrated with Flask. He also worked on multi-turn dialogue systems for virtual assistants in customer support domains.",
            "answers": {
                "text": ["a portfolio chatbot using a custom-trained DistilBERT model"],
                "answer_start": [17],
            }
        }
    ]
}
    metric = evaluate.load("squad")

    print("Loading datasets...")
    custom_dataset = load_custom_dataset(datasets)
    squad_train, squad_val = load_squad_datasets()

    combined_dataset = combine_datasets(custom_dataset, squad_train)
    train_dataset, val_dataset = combined_dataset["train"], combined_dataset["test"]

    train_dataset = clean_and_filter_dataset(train_dataset)
    val_dataset = clean_and_filter_dataset(val_dataset)

    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    tokenized_train_dataset, tokenized_val_dataset = load_or_preprocess_dataset(
        train_dataset, val_dataset, tokenizer, force_retokenize=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
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
        tf32=True,
    )

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

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved!")
