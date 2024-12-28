import pytest
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
import unicodedata
from word2number import w2n  # For converting words to numbers
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')


# Path to fine-tuned ALBERT model and tokenizer
MODEL_PATH = "/chatbot_portfolio/models/fine_tuned_albert"

# Load the custom dataset for testing
DATASET_PATH = "/chatbot_portfolio/data/final_cleaned_merged_dataset.json"
with open(DATASET_PATH, "r") as f:
    custom_data = json.load(f)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Prepare stopwords for normalization
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def normalize_text(text):
    """
    Normalize text by removing special characters (e.g., accents) and converting to lowercase.
    """
    text = unicodedata.normalize("NFKD", text)  # Decompose characters (e.g., é -> e + ́)
    text = "".join([c for c in text if not unicodedata.combining(c)])  # Remove diacritics
    return text.lower()

def extract_number(text):
    """
    Extract numeric values from text. Convert word numbers to digits if possible.
    """
    try:
        return w2n.word_to_num(text)  # Convert word to number if possible
    except ValueError:
        # Fallback: Count elements in a list if comma-separated
        tokens = text.split(',')
        if len(tokens) > 1:
            return len(tokens)
        return None

# Initialize a Sentence-BERT model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(predicted, expected_list):
    """
    Compute the highest semantic similarity between the predicted answer
    and a list of expected answers.
    """
    predicted_embedding = semantic_model.encode(predicted, convert_to_tensor=True)
    expected_embeddings = semantic_model.encode(expected_list, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(predicted_embedding, expected_embeddings)
    return similarities.max().item()  # Return the highest similarity score


@pytest.mark.parametrize("data", custom_data)
def test_albert_fine_tune(data):
    context = data["context"]
    question = data["question"]
    expected_answers = data["answers"]["text"]

    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Send inputs to the correct device

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get predicted start and end positions
    start_idx = start_logits.argmax(dim=-1).item()
    end_idx = end_logits.argmax(dim=-1).item()

    # Decode the answer
    if start_idx <= end_idx:
        pred_answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1], skip_special_tokens=True)
    else:
        pred_answer = "No answer found."

    # Normalize predicted and expected answers
    pred_tokens = [word for word in word_tokenize(normalize_text(pred_answer)) if word not in stop_words]
    expected_tokens_list = [
        [word for word in word_tokenize(normalize_text(ans)) if word not in stop_words]
        for ans in expected_answers
    ]

    # Token overlap ratio metric
    overlap_ratios = [
        len(set(pred_tokens).intersection(set(expected_tokens))) / max(len(expected_tokens), 1)
        for expected_tokens in expected_tokens_list
    ]
    max_overlap = max(overlap_ratios)

    # Compute semantic similarity
    semantic_similarity = compute_semantic_similarity(pred_answer, expected_answers)

    # Log information for debugging
    if max_overlap < 0.4 and semantic_similarity < 0.7:  # Debugging threshold
        print("\n[DEBUG - Low Overlap]")
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Expected: {expected_answers}")
        print(f"Predicted: {pred_answer}")
        print(f"Predicted Tokens: {pred_tokens}")
        print(f"Expected Tokens List: {expected_tokens_list}")
        print(f"Overlap Ratios: {overlap_ratios}")
        print(f"Maximum Overlap: {max_overlap:.2f}")
        print(f"Semantic Similarity: {semantic_similarity:.2f}")

    # Assert based on token overlap ratio or semantic similarity
    assert max_overlap >= 0.4 or semantic_similarity >= 0.7, (
        f"Low overlap and semantic similarity! Predicted: '{pred_answer}', "
        f"Expected: {expected_answers}, Overlap: {max_overlap:.2f}, "
        f"Semantic Similarity: {semantic_similarity:.2f}"
    )
