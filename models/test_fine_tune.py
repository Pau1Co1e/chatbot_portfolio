from transformers import pipeline

def test_qa_pipeline():
    # Load fine-tuned model
    qa_pipeline = pipeline("question-answering", model="./model/fine_tune.py", tokenizer="./model/fine_tuned_model")

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
        },
        {
            "question": "Where have you worked recently?",
            "context": "Paul Coleman worked as a Full Stack Software Engineer at ResNexus from September 2023 to December 2023. He developed backend APIs and frontend components using the Microsoft tech stack for hotel and rental management products.",
            "expected": "Full Stack Software Engineer at ResNexus"
        },
        {
            "question": "What volunteer experience do you have?",
            "context": "Paul Coleman organized a local donation event for the Northern Utah Habitat for Humanity Clothing Drive in June 2014, supporting children and families in the Dominican Republic.",
            "expected": "Northern Utah Habitat for Humanity Clothing Drive"
        }
    ]

    # Run test cases
    for case in test_cases:
        result = qa_pipeline(question=case["question"], context=case["context"])
        assert case["expected"] in result["answer"], f"Failed for question: {case['question']}"