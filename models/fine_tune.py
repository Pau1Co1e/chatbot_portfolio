from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

# Define custom data
custom_data = [
    {
        "question": "What is your educational background?",
        "context": "Paul Coleman holds a Master of Science in Computer Science from Utah Valley University (expected August 2025) and a Bachelor of Science in Computer Science (graduated August 2022) with a GPA of 3.26.",
        "answers": {
            "text": ["Master of Science in Computer Science from Utah Valley University", "Bachelor of Science in Computer Science"],
            "answer_start": [19, 110]
        }
    },
    {
        "question": "What are your professional skills?",
        "context": "Paul Coleman is skilled in AI & Machine Learning, model development, algorithm design, NLP, web development with Flask and JavaScript, and scalable systems design. His programming expertise includes Python, C#, and Java.",
        "answers": {
            "text": ["AI & Machine Learning", "Flask and JavaScript", "Python, C#, and Java"],
            "answer_start": [19, 80, 147]
        }
    },
    # Additional entries truncated for brevity
]

# Load custom dataset
custom_dataset = Dataset.from_dict({
    "question": [entry["question"] for entry in custom_data],
    "context": [entry["context"] for entry in custom_data],
    "answers": [entry["answers"] for entry in custom_data]
})

# Load model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess the data
def preprocess_data(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding=True
    )

    start_positions = []
    end_positions = []

    # Process each example in the batch
    for i in range(len(examples["answers"])):
        answer = examples["answers"][i]  # Get the answers dictionary for the i-th example
        start_idx = answer["answer_start"][0]  # Take the first answer (assumes one answer per example)
        end_idx = start_idx + len(answer["text"][0])  # Compute the end index
        start_positions.append(tokenized.char_to_token(i, start_idx))
        end_positions.append(tokenized.char_to_token(i, end_idx - 1))

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

tokenized_dataset = custom_dataset.map(preprocess_data, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=50,
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset
)

trainer.train()

# Save fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")