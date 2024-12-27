import pytest
from transformers import AlbertTokenizerFast, AlbertForQuestionAnswering
import torch

# Load the fine-tuned model and tokenizer
model_path = "C:/Users/Paul/PycharmProjects/chatbot_portfolio/models/fine_tuned_albert"
tokenizer = AlbertTokenizerFast.from_pretrained(model_path)
model = AlbertForQuestionAnswering.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)


def answer_question_debug(question, context):
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
        answer = "<no_answer>"

    # Debugging information
    print("\n[DEBUG]")
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Predicted Answer: {answer}")
    print(f"Start Index: {start_index}, End Index: {end_index}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Start Logits: {start_logits.tolist()}")
    print(f"End Logits: {end_logits.tolist()}")

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
def test_answer_question_debug(context, question, expected):
    actual = answer_question_debug(question, context)

    # Handle no-answer cases
    if expected == "<no_answer>":
        assert actual == "<no_answer>", f"Expected no answer, got '{actual}'."
    else:
        # Compute token overlap for coverage
        expected_tokens = set(expected.lower().split())
        actual_tokens = set(actual.lower().split())
        coverage = len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)

        if coverage < 0.4:
            # Debugging information
            print(f"\n[DEBUG - Low Coverage]")
            print(f"Context: {context}")
            print(f"Question: {question}")
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            print(f"Coverage: {coverage:.2f}")

        assert coverage >= 0.4, f"Low coverage: Expected '{expected}', got '{actual}' (Coverage: {coverage:.2f})."
