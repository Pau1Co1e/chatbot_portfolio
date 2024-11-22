from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
import evaluate
import pandas as pd

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define datasets
datasets = {
    "education_and_certifications": [
        {
            "question": "What is your educational background?",
            "context": "Paul Coleman holds a Master of Science in Computer Science from Utah Valley University (expected August 2025) and a Bachelor of Science in Computer Science (graduated August 2022) with a GPA of 3.26.",
            "answers": {
                "text": ["Master of Science in Computer Science from Utah Valley University", "Bachelor of Science in Computer Science"],
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
            "context": "Paul led a project on Autonomous Mobile Robots at UVU, focusing on navigation and decision-making using reinforcement learning and sensor fusion techniques. He optimized the robots for warehouse management tasks.",
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

# Combine datasets into a single list
# Flatten and convert to DataFrame
flattened_data = []

for category, entries in datasets.items():
    for entry in entries:
        for text, start in zip(entry["answers"]["text"], entry["answers"]["answer_start"]):
            flattened_data.append({
                "category": category,
                "question": entry["question"],
                "context": entry["context"],
                "answer_text": text,
                "answer_start": start
            })

# Create a pandas DataFrame
df = pd.DataFrame(flattened_data)

# Inspect the DataFrame
print(df.head())

# Perform some basic EDA
print("\nEDA Summary:")
print(df.info())
print(df.describe(include="all"))
print(df.isnull().sum())
print(df.isnull().sum())

print("\nUnique Questions:")
print(df['question'].unique())
print("\nContext Length Statistics:")
print(df['context'].apply(len).describe())

# Convert pandas DataFrame back to Hugging Face Dataset
custom_dataset = Dataset.from_pandas(df)

# Load SQuAD dataset for augmentation
squad_dataset = load_dataset(
    "json",
    data_files={
        "train": "train-v1.1.json",
        "validation": "dev-v1.1.json",
    },
)

# Combine custom data with SQuAD data
combined_dataset = concatenate_datasets([custom_dataset, squad_dataset["train"]])

# Split combined dataset into training and validation sets
split_dataset = combined_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"\nTrain Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")

# Tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_data(examples):
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
        try:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            offsets = tokenized.offset_mapping[i]

            start_token = next(idx for idx, (s, e) in enumerate(offsets) if s <= start_char < e)
            end_token = next(idx for idx, (s, e) in enumerate(offsets) if s < end_char <= e)

            start_positions.append(start_token)
            end_positions.append(end_token)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            start_positions.append(0)
            end_positions.append(0)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    del tokenized["offset_mapping"]  # Remove unnecessary data
    return tokenized

# Filter invalid examples
def filter_invalid_examples(dataset):
    return dataset.filter(lambda example:
        "text" in example["answers"] and
        "answer_start" in example["answers"] and
        len(example["answers"]["text"]) > 0 and
        len(example["answers"]["answer_start"]) > 0
    )

def replace_none_values(dataset):
    def replace_none(example):
        # Replace None with defaults
        example["answers"] = example.get("answers") or {"text": [], "answer_start": []}
        example["answers"]["text"] = example["answers"].get("text") or []
        example["answers"]["answer_start"] = example["answers"].get("answer_start") or []
        return example

    return dataset.map(replace_none)

def drop_none_values(dataset):
    def is_valid(example):
        # Check if answers is None or invalid
        return (
            example.get("answers") is not None
            and isinstance(example["answers"], dict)
            and "text" in example["answers"]
            and "answer_start" in example["answers"]
        )

    return dataset.filter(is_valid)

# Replace or drop None values
train_dataset = replace_none_values(train_dataset)  # or drop_none_values
val_dataset = replace_none_values(val_dataset)  # or drop_none_values

# Filter invalid examples
train_dataset = filter_invalid_examples(train_dataset)
val_dataset = filter_invalid_examples(val_dataset)

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=train_dataset.column_names
)
tokenized_val_dataset = val_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=val_dataset.column_names
)

print("\nTokenized Datasets Ready")

# Model
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# Metrics (uses evaluate)
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

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
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

# Train and save
trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")