import collections
from datasets import Dataset, concatenate_datasets
import os
import logging
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import evaluate
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator, EvalPrediction
)


# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

metric = evaluate.load("squad")

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths
CUSTOM_DATA_DIR = "../models/dataset/"
OUTPUT_DIR = "../models/fine_tuned_model/"
CACHE_DIR = "../models/cache/"
MODEL_DIR = "../models/fine_tuned_model/"

# Load datasets from parquet files
def load_squad_datasets_from_parquet(train_path, validation_path):
    try:
        logger.info("Loading SQuAD dataset from parquet files...")
        train_df = pd.read_parquet(train_path)
        validation_df = pd.read_parquet(validation_path)
        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(validation_df)
        return train_dataset, validation_dataset
    except Exception as e:
        logger.error(f"Error loading SQuAD dataset from parquet files: {e}")
        raise


# Load custom dataset
def load_custom_dataset(flat_data):
    """
    Convert flattened data into Hugging Face Dataset schema.
    """
    processed_data = [
        {
            "id": entry["id"],
            "context": entry["context"],
            "question": entry["question"],
            "answers": {
                "text": entry["answer_text"],  # Use answer_text directly
                "answer_start": entry["answer_start"],  # Use answer_start directly
            },
        }
        for entry in flat_data
    ]
    return Dataset.from_pandas(pd.DataFrame(processed_data))


# Combine datasets with schema alignment
def combine_datasets(custom_dataset, squad_dataset):
    from datasets import Features, Sequence, Value

    # Define the common feature schema
    dataset_features = Features({
        "id": Value("string"),
        "title": Value("string"),
        "context": Value("string"),
        "question": Value("string"),
        "answers": {
            "text": Sequence(Value("string")),
            "answer_start": Sequence(Value("int32")),
        }
    })

    # Ensure alignment of schema for both datasets
    def align_schema(example):
        return {
            "id": example["id"],
            "title": example.get("title", "custom"),  # Add a default title if missing
            "context": example["context"],
            "question": example["question"],
            "answers": {
                "text": example["answers"]["text"],
                "answer_start": [int(x) for x in example["answers"]["answer_start"]],
            }
        }

    # Align schema for custom dataset
    # custom_dataset = custom_dataset.map(align_schema, batched=False)
    custom_dataset = custom_dataset.map(align_schema, batched=False, new_fingerprint="custom_align_schema")
    # Align schema for SQuAD dataset
    squad_dataset = squad_dataset.map(align_schema, batched=False, new_fingerprint="squad_align_schema")

    # Cast both datasets to the common schema
    custom_dataset = custom_dataset.cast(dataset_features)
    squad_dataset = squad_dataset.cast(dataset_features)

    # Verify features
    print("Custom Dataset Features after casting:", custom_dataset.features)
    print("SQuAD Dataset Features after casting:", squad_dataset.features)

    # Ensure alignment for all features
    assert custom_dataset.features == squad_dataset.features, "Dataset features do not match!"

    # Concatenate datasets
    combined_dataset = concatenate_datasets([custom_dataset, squad_dataset])

    # Split into train and validation sets
    return combined_dataset.train_test_split(test_size=0.2, seed=42)


# Preprocess the data for tokenization
def preprocess_eval_data(examples, tokenizer):
    # Tokenize the inputs
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,  # Necessary for mapping tokens back to context
        padding="max_length",
    )

    # Mapping from new feature to original example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    # Add 'example_id' to each feature
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Map feature to its corresponding example ID
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping values that are not part of the context
        sequence_ids = tokenized_examples.sequence_ids(i)
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offset_mapping[i])
        ]

    return tokenized_examples


def preprocess_train_data(examples, tokenizer):
    # Tokenize the inputs with truncation and padding, but keep the overflows using a stride.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,  # Necessary for post-processing
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a mapping from feature to example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # Map feature back to its example.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers are given, set start and end positions to CLS index.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(tokenizer.cls_token_id)
            tokenized_examples["end_positions"].append(tokenizer.cls_token_id)
        else:
            # Start and end character positions of the answer in the context.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the start and end token indices corresponding to the answer.
            token_start_index = 0
            while tokenized_examples.sequence_ids(i)[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(tokenized_examples["input_ids"][i]) - 1
            while tokenized_examples.sequence_ids(i)[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in case of long contexts split into spans)
            if offsets[token_start_index][0] > end_char or offsets[token_end_index][1] < start_char:
                tokenized_examples["start_positions"].append(tokenizer.cls_token_id)
                tokenized_examples["end_positions"].append(tokenizer.cls_token_id)
            else:
                # Move the token_start_index and token_end_index to the actual start and end of the answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def load_or_preprocess_dataset(train_dataset, val_dataset, tokenizer, cache_dir=CACHE_DIR, force_retokenize=False):
    train_cache_path = os.path.join(cache_dir, "train")
    val_cache_path = os.path.join(cache_dir, "val")

    if force_retokenize or not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path)):
        logger.info("Tokenizing datasets...")

        tokenized_train_dataset = train_dataset.map(
            preprocess_train_data,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=os.cpu_count(),
            fn_kwargs={'tokenizer': tokenizer}
        )

        tokenized_val_dataset = val_dataset.map(
            preprocess_eval_data,
            batched=True,
            remove_columns=val_dataset.column_names,
            num_proc=os.cpu_count(),
            fn_kwargs={'tokenizer': tokenizer}
        )

        # Save the tokenized datasets
        tokenized_train_dataset.save_to_disk(train_cache_path)
        tokenized_val_dataset.save_to_disk(val_cache_path)
    else:
        logger.info("Loading tokenized datasets from cache...")
        tokenized_train_dataset = load_from_disk(train_cache_path)
        tokenized_val_dataset = load_from_disk(val_cache_path)

    return tokenized_train_dataset, tokenized_val_dataset


def postprocess_qa_predictions(
    examples,
    features,
    predictions,
    version_2_with_negative=False,
    n_best_size=20,
    max_answer_length=30,
    null_score_diff_threshold=0.0,
):

    all_start_logits, all_end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        example_id = feature["example_id"]
        features_per_example[example_id_to_index[example_id]].append(i)

    final_predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]

            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = context[start_char:end_char]

                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": answer_text,
                        }
                    )

        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
        else:
            best_answer = {"text": "", "score": 0.0}

        if version_2_with_negative:
            if min_null_score is not None and min_null_score - best_answer["score"] > null_score_diff_threshold:
                final_predictions[example["id"]] = ""
            else:
                final_predictions[example["id"]] = best_answer["text"]
        else:
            final_predictions[example["id"]] = best_answer["text"]

    return final_predictions


def compute_metrics(p: EvalPrediction):
    # Unpack predictions
    start_logits, end_logits = p.predictions
    start_positions, end_positions = p.label_ids

    # Compute the loss between the predicted logits and the true labels
    from torch.nn.functional import cross_entropy

    # Convert logits and labels to tensors if they're not already
    start_logits = torch.tensor(start_logits)
    end_logits = torch.tensor(end_logits)
    start_positions = torch.tensor(start_positions)
    end_positions = torch.tensor(end_positions)

    # Compute the loss for start and end positions
    start_loss = cross_entropy(start_logits, start_positions).float()
    end_loss = cross_entropy(end_logits, end_positions).float()

    # Compute the average loss using float division
    total_loss = (start_loss + end_loss) / 2.0

    # Post-process predictions
    final_predictions = postprocess_qa_predictions(
        examples=val_dataset,
        features=tokenized_val_dataset,
        predictions=(start_logits.detach().numpy(), end_logits.detach().numpy()),
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0
    )

    # Prepare references
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_dataset]

    # Load the metric
    metric = evaluate.load("squad")

    # Compute the metrics
    squad_metrics = metric.compute(predictions=final_predictions, references=references)

    # Include the loss in the metrics dictionary
    squad_metrics["eval_loss"] = total_loss.item()

    # Print metrics
    print(f"Metrics: {squad_metrics}")

    return squad_metrics

# Debugging function to inspect features of a dataset
def inspect_features(dataset, name):
    logger.info(f"{name} Dataset Features: {dataset.features}")
    for feature, dtype in dataset.features.items():
        logger.info(f"Feature '{feature}' - Type: {dtype}")


# Main execution
if __name__ == "__main__":
    # Define the datasets for validation

    custom_data = [
        {
            "id": "education_and_certifications-0",
            "question": "What is your educational background?",
            "context": "Paul Coleman holds a Master of Science in Computer Science from Utah Valley University (expected August 2025) and a Bachelor of Science in Computer Science (graduated August 2022) with a GPA of 3.26.",
            "answer_text": ["Master of Science in Computer Science from Utah Valley University",
                            "Bachelor of Science in Computer Science"],
            "answer_start": [19, 110]
        },
        {
            "id": "education_and_certifications-1",
            "question": "What certifications do you have?",
            "context": "Paul Coleman has a Programmer Certification from Utah Valley University (2020), was on the Dean's List in 2020 and 2021, and holds a CompTIA A+ Certification from Dell Inc.",
            "answer_text": ["Programmer Certification", "CompTIA A+ Certification"],
            "answer_start": list(map(np.int32, [19, 120]))
        },
        {
            "id": "professional_skills-0",
            "question": "What are your professional skills?",
            "context": "Paul Coleman is skilled in AI & Machine Learning, model development, algorithm design, NLP, web development with Flask and JavaScript, and scalable systems design. His programming expertise includes Python, C#, and Java.",
            "answer_text": ["AI & Machine Learning", "model development", "algorithm design",
                            "NLP", "web development with Flask and JavaScript",
                            "scalable systems design", "Python", "C#", "Java"],
            "answer_start": list(map(np.int32, [19, 40, 59, 78, 112, 153, 182, 190, 195]))
        },
        {
            "id": "work_experience-0",
            "question": "Where have you worked recently?",
            "context": "Paul Coleman worked as a Full Stack Software Engineer at ResNexus from September 2023 to December 2023. He developed backend APIs and frontend components using the Microsoft tech stack for hotel and rental management products.",
            "answer_text": ["Full Stack Software Engineer at ResNexus", "ResNexus"],
            "answer_start": list(map(np.int32, [18, 52]))
        },
        {
            "id": "volunteer_experience-0",
            "question": "What volunteer experience do you have?",
            "context": "Paul Coleman organized a local donation event for the Northern Utah Habitat for Humanity Clothing Drive in June 2014, supporting children and families in the Dominican Republic.",
            "answer_text": ["Northern Utah Habitat for Humanity Clothing Drive",
                            "supporting children and families in the Dominican Republic"],
            "answer_start": list(map(np.int32, [34, 118]))
        },
        {
            "id": "ai_concepts-0",
            "question": "How has Paul applied Interpretable Machine Learning in projects?",
            "context": "Paul Coleman has experience with Interpretable Machine Learning techniques like LIME and SHAP. He has applied these methods to financial prediction models to explain outputs to stakeholders in the FinTech industry.",
            "answer_text": ["Interpretable Machine Learning techniques like LIME and SHAP",
                            "financial prediction models"],
            "answer_start": list(map(np.int32, [42, 88]))
        },
        {
            "id": "ai_concepts-1",
            "question": "What are Paul’s contributions to Transfer Learning?",
            "context": "Paul has implemented Transfer Learning in NLP applications, utilizing pre-trained language models such as BERT for sentiment analysis and chatbots. He optimized transfer learning workflows for rapid deployment in AI systems.",
            "answer_text": ["Transfer Learning in NLP applications",
                            "Optimized transfer learning workflows for rapid deployment"],
            "answer_start": list(map(np.int32, [24, 129]))
        },
        {
            "id": "ai_concepts-2",
            "question": "What is Paul’s experience with Reinforcement Learning?",
            "context": "Paul worked on Reinforcement Learning algorithms for autonomous mobile robots during his time at Utah Valley University. He focused on developing agents capable of decision-making in dynamic environments using Q-Learning and Deep Q-Networks.",
            "answer_text": ["Reinforcement Learning algorithms for autonomous mobile robots",
                            "developing agents capable of decision-making in dynamic environments"],
            "answer_start": list(map(np.int32, [17, 87]))
        },
        {
            "id": "ai_concepts-3",
            "question": "How has Paul leveraged NLP in his projects?",
            "context": "Paul specializes in Natural Language Processing, having developed conversational AI systems using Transformer-based models like GPT. His projects include chatbots for customer support, Q&A assistant, and advanced text summarization tools.",
            "answer_text": ["Natural Language Processing", "advanced text summarization tools",
                            "chatbots for customer support"],
            "answer_start": list(map(np.int32, [17, 137, 92]))
        },
        {
            "id": "ai_concepts-4",
            "question": "What role has Anomaly Detection played in Paul’s work?",
            "context": "Paul implemented Anomaly Detection techniques in cybersecurity applications to identify fraud and unusual network behavior. He used autoencoders and statistical methods to improve system reliability.",
            "answer_text": ["Anomaly Detection techniques in cybersecurity applications",
                            "improve system reliability", "identify fraud and unusual network behavior"],
            "answer_start": list(map(np.int32, [17, 137, 64]))
        },
        {
            "id": "ai_concepts-5",
            "question": "How has Paul used Time Series Analysis?",
            "context": "Paul has applied Time Series Analysis in financial forecasting, developing models to predict stock price movements and economic trends. He has experience using ARIMA, LSTM networks, and seasonal decomposition techniques.",
            "answer_text": ["Time Series Analysis in financial forecasting",
                            "seasonal decomposition techniques"],
            "answer_start": list(map(np.int32, [17, 159]))
        },
        {
            "id": "ai_concepts-6",
            "question": "What is Paul’s expertise in Computer Vision?",
            "context": "Paul has developed Computer Vision models for object detection and image segmentation. His projects include building a vision-based quality inspection system for manufacturing and applying OpenCV and TensorFlow for object tracking and facial image recognition.",
            "answer_text": ["Computer Vision models for object detection and image segmentation",
                            "object tracking", "facial image recognition"],
            "answer_start": list(map(np.int32, [17, 136, 159]))
        },
        {
            "id": "ai_concepts-7",
            "question": "What types of Neural Networks has Paul worked with?",
            "context": "Paul has worked with convolutional neural networks (CNNs) for computer vision tasks like facial and object detection, and recurrent neural networks (RNNs) and long short-term memory networks (LSTMs) for speech recognition in humanoid robotics.",
            "answer_text": ["convolutional neural networks (CNNs) for computer vision tasks",
                            "recurrent neural networks (RNNs)",
                            "long short-term memory networks (LSTMs)",
                            "speech recognition in humanoid robotics"],
            "answer_start": list(map(np.int32, [19, 104, 147, 197]))
        },
        {
            "id": "ai_concepts-8",
            "question": "What is Paul’s experience with Autonomous Mobile Robots?",
            "context": "Paul led a project on a pair of humanoid Autonomous Mobile Robots at UVU, focusing on navigation and decision-making using reinforcement learning and sensor fusion techniques.",
            "answer_text": ["Autonomous Mobile Robots at UVU",
                            "navigation and decision-making using reinforcement learning"],
            "answer_start": list(map(np.int32, [26, 76]))
        },
        {
            "id": "ai_concepts-9",
            "question": "What are Paul’s accomplishments in Conversational AI?",
            "context": "Paul developed a portfolio chatbot using a custom-trained DistilBERT model integrated with Flask. He also worked on multi-turn dialogue systems for virtual assistants in customer support domains.",
            "answer_text": ["a portfolio chatbot using a custom-trained DistilBERT model",
                            "multi-turn dialogue systems for virtual assistants"],
            "answer_start": list(map(np.int32, [18, 76]))
        }
    ]

    # Load custom dataset
    custom_dataset = load_custom_dataset(custom_data)
    print(custom_dataset[0])  # Inspect the first example

    # Load SQuAD datasets
    squad_train, squad_val = load_squad_datasets_from_parquet(
        os.path.join(CUSTOM_DATA_DIR, "train.parquet"),
        os.path.join(CUSTOM_DATA_DIR, "validation.parquet")
    )

    # Combine datasets
    combined_dataset = combine_datasets(custom_dataset, squad_train)
    train_dataset, val_dataset = combined_dataset["train"], combined_dataset["test"]

    # Initialize tokenizer and model
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # Tokenize datasets
    tokenized_train_dataset, tokenized_val_dataset = load_or_preprocess_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        cache_dir=CACHE_DIR,
        force_retokenize=True  # Force re-tokenization to ensure 'example_id' is included
    )

    print("Tokenized validation dataset columns:", tokenized_val_dataset.column_names)
    print("Sample of tokenized validation dataset:", tokenized_val_dataset[0])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",  # Ensure evaluation occurs every epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        num_train_epochs=3,
        weight_decay=0.005,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Ensure this key exists in metrics
        greater_is_better=False,  # Lower loss is better
        logging_steps=10,
    )

    # Initialize Trainer
    metric = evaluate.load("squad")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train and save model
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Model training and saving complete!")
