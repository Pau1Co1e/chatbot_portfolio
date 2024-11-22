from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
import os
import evaluate
import pandas as pd

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Dynamic paths for data loading
SQUAD_DATA_DIR = "hf://datasets/rajpurkar/squad/"
CUSTOM_DATA_DIR = "./custom_data"
OUTPUT_DIR = "./fine_tuned_model"

# Load SQuAD datasets dynamically
def load_squad_datasets(data_dir=SQUAD_DATA_DIR):
    splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet'}
    train_df = pd.read_parquet(os.path.join(data_dir, splits["train"]))
    val_df = pd.read_parquet(os.path.join(data_dir, splits["validation"]))
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

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

# Preprocessing function
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
        offsets = tokenized.offset_mapping[i]
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

def replace_none_values(dataset):
    def replace_none(example):
        # Replace missing or None 'answers' with default structure
        example["answers"] = example.get("answers") or {"text": [], "answer_start": []}
        example["answers"]["text"] = example["answers"].get("text") or []
        example["answers"]["answer_start"] = example["answers"].get("answer_start") or []
        return example

    # Apply the replacement to all examples in the dataset
    return dataset.map(replace_none)

# Filter invalid examples
def filter_invalid_examples(dataset):
    def is_valid(example):
        if example.get("answers") is None:
            print(f"Invalid example found: {example}")  # Log invalid examples
            return False
        return (
            "text" in example["answers"]
            and "answer_start" in example["answers"]
            and len(example["answers"]["text"]) > 0
            and len(example["answers"]["answer_start"]) > 0
        )

    return dataset.filter(is_valid)

# Main execution
if __name__ == "__main__":
    # Custom dataset definition
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

    # Load datasets
    print("Loading datasets...")
    custom_dataset = load_custom_dataset(datasets)
    squad_train, squad_val = load_squad_datasets()
    combined_dataset = combine_datasets(custom_dataset, squad_train)

    train_dataset, val_dataset = combined_dataset["train"], combined_dataset["test"]

    train_dataset = replace_none_values(train_dataset)
    val_dataset = replace_none_values(val_dataset)

    # Filter invalid examples
    train_dataset = filter_invalid_examples(train_dataset)
    val_dataset = filter_invalid_examples(val_dataset)

    print(train_dataset[:5])  # Inspect the first few examples for issues
    # Tokenizer
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # Metrics
    metric = evaluate.load("squad")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        formatted_predictions = [
            {"id": str(i), "prediction_text": pred} for i, pred in enumerate(predictions)
        ]
        formatted_references = [
            {"id": str(i), "answers": label} for i, label in enumerate(labels)
        ]
        return metric.compute(predictions=formatted_predictions, references=formatted_references)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    print("Starting training...")
    trainer.train()

    # Save model and tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved!")
