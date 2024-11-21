from transformers import pipeline

def test_qa_pipeline():
    # Load fine-tuned model
    qa_pipeline = pipeline("question-answering", model="./fine_tuned_model", tokenizer="./fine_tuned_model")

    # Define test cases
    test_cases = [
        {
            "question": "What is your educational background?",
            "context": "Paul Coleman holds a Master of Science in Computer Science from Utah Valley University (expected August 2025) and a Bachelor of Science in Computer Science (graduated August 2022).",
            "expected": "Master of Science"
        },
        {
            "question": "What are your professional skills?",
            "context": "Paul Coleman is skilled in AI & Machine Learning, NLP, and scalable system design.",
            "expected": "AI & Machine Learning"
        }
    ]

    # Run test cases
    for case in test_cases:
        result = qa_pipeline(question=case["question"], context=case["context"])
        assert case["expected"] in result["answer"], f"Failed for question: {case['question']}"