import pytest
from transformers import AlbertTokenizerFast, AlbertForQuestionAnswering
import torch

# Load the fine-tuned model and tokenizer
model_path = "/Users/paulcoleman/Documents/PersonalCode/chatbot_portfolio_albert/models/fine_tuned_albert"
tokenizer = AlbertTokenizerFast.from_pretrained(model_path)
model = AlbertForQuestionAnswering.from_pretrained(model_path)

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits

    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()

    if start_index <= end_index:
        answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1], skip_special_tokens=True)
    else:
        answer = "No answer found."

    return answer

@pytest.mark.parametrize(
    "context, question, expected",
    [
        (
            "Albert Einstein was a theoretical physicist who developed the theory of relativity, "
            "one of the two pillars of modern physics. His work is also known for its influence on "
            "the philosophy of science.",
            "What did Albert Einstein develop?",
            "the theory of relativity",
        ),
        (
            "Paul Coleman is an AI and Machine Learning Engineer with experience in computer vision, NLP, "
            "financial prediction models, anti-spam systems, and deepfake detection. He holds a Bachelor's "
            "in Computer Science from Utah Valley University and is pursuing a Master's degree with expected "
            "graduation in August 2025.",
            "What is Paul Coleman's expertise?",
            "AI and Machine Learning Engineer with experience in computer vision, NLP, financial prediction models, anti-spam systems, and deepfake detection",
        ),
        (
            "Paul Coleman is an AI and Machine Learning Engineer with experience in computer vision, NLP, "
            "financial prediction models, anti-spam systems, and deepfake detection. He holds a Bachelor's "
            "in Computer Science from Utah Valley University and is pursuing a Master's degree with expected "
            "graduation in August 2025.",
            "When will Paul Coleman graduate?",
            "August 2025",
        ),
        (
            "Albert Einstein was a theoretical physicist who developed the theory of relativity, "
            "one of the two pillars of modern physics. His work is also known for its influence on "
            "the philosophy of science.",
            "What are the two pillars of modern physics?",
            "No answer found.",
        ),
        (
            "Paul Coleman is an AI and Machine Learning Engineer with experience in computer vision, NLP, "
            "financial prediction models, anti-spam systems, and deepfake detection. He is also skilled "
            "in ASP.NET, Python, and Blazor frameworks.",
            "What programming frameworks does Paul Coleman use?",
            "ASP.NET, Python, and Blazor frameworks",
        ),
    ],
)
def test_answer_question(context, question, expected):
    actual = answer_question(question, context)

    # Normalize case for comparison
    actual_words = set(actual.lower().split())
    expected_words = set(expected.lower().split())

    # Special handling for "No Answer Found"
    if expected.lower() in ["no answer found.", "no answer"]:
        if actual.lower() in ["no answer found.", "no answer"]:
            assert True
        else:
            print(f"Context: {context}")
            print(f"Question: {question}")
            print(f"Expected: No Answer Found")
            print(f"Actual: {actual}")
            assert False, f"Expected 'No Answer Found', but got '{actual}'"
    else:
        # Check for coverage of key words
        common_words = actual_words.intersection(expected_words)
        required_coverage = 0.4  # Lower leniency to 40% of expected words
        coverage = len(common_words) / max(len(expected_words), 1)

        if coverage >= required_coverage:
            assert True
        else:
            print(f"Context: {context}")
            print(f"Question: {question}")
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            print(f"Coverage: {coverage:.2f}")
            assert False, f"Expected '{expected}', but got '{actual}'"
