from datasets import Dataset
from difflib import SequenceMatcher
import torch
from transformers import pipe


test_cases = [
        {
            "question": "Where have you worked recently?",
            "context": "Paul Coleman worked as a Full Stack Software Engineer at ResNexus from September 2023 to December 2023. He developed backend APIs and frontend components using the Microsoft tech stack for hotel and rental management products.",
            "expected": [
                "Full Stack Software Engineer at ResNexus",
                "ResNexus"
            ]
        },
        {
            "question": "What volunteer experience do you have?",
            "context": "Paul Coleman organized a local donation event for the Northern Utah Habitat for Humanity Clothing Drive in June 2014, supporting children and families in the Dominican Republic.",
            "expected": [
                "Northern Utah Habitat for Humanity Clothing Drive",
                "supporting children and families in the Dominican Republic"
            ]
        },
        {
            "question": "What is your educational background?",
            "context": "Paul Coleman holds a Master of Science in Computer Science from Utah Valley University (expected August 2025) and a Bachelor of Science in Computer Science (graduated August 2022) with a GPA of 3.26.",
            "expected": [
                "Master of Science in Computer Science from Utah Valley University",
                "Bachelor of Science in Computer Science"
            ]
        },
        {
            "question": "What certifications do you have?",
            "context": "Paul Coleman has a Programmer Certification from Utah Valley University (2020), was on the Dean's List in 2020 and 2021, and holds a CompTIA A+ Certification from Dell Inc.",
            "expected": [
                "Programmer Certification",
                "CompTIA A+ Certification"
            ]
        },
        {
            "question": "What are Paul’s contributions to Transfer Learning?",
            "context": "Paul has implemented Transfer Learning in NLP applications, utilizing pre-trained language models such as BERT for sentiment analysis and chatbots. He optimized transfer learning workflows for rapid deployment in AI systems.",
            "expected": [
                "He optimized transfer learning workflows for rapid deployment in AI systems",
                "Optimized transfer learning workflows for rapid deployment in AI systems"
            ]
        },
        {
            "question": "What is Paul’s experience with Reinforcement Learning?",
            "context": "Paul worked on Reinforcement Learning algorithms for autonomous mobile robots during his time at Utah Valley University. He focused on developing agents capable of decision-making in dynamic environments using Q-Learning and Deep Q-Networks.",
            "expected": [
                "He focused on developing agents capable of decision-making in dynamic environments",
                "Developing agents capable of decision-making in dynamic environments"
            ]
        },
        {
            "question": "What are Paul's professional skills?",
            "context": "Paul Coleman is skilled in AI & Machine Learning, model development, algorithm design, NLP, web development with Flask and JavaScript, and scalable systems design. His programming expertise includes Python, C#, and Java.",
            "expected": [
                "AI & Machine Learning",
                "model development",
                "algorithm design",
                "NLP",
                "web development with Flask and JavaScript",
                "scalable systems design",
                "Python",
                "C#",
                "Java"
            ]
        },
        {
            "question": "How has Paul applied Interpretable Machine Learning in projects?",
            "context": "Paul Coleman has experience with Interpretable Machine Learning techniques like LIME and SHAP. He has applied these methods to financial prediction models.",
            "expected": ["financial prediction models",
                         "interpretable machine learning techniques like LIME and SHAP"
                         ]

        },
        {
            "question": "What are Paul’s contributions to Transfer Learning?",
            "context": "Paul has implemented Transfer Learning in NLP applications, utilizing pre-trained language models such as BERT for sentiment analysis and chatbots. He optimized transfer learning workflows for rapid deployment in AI systems.",
            "expected": [
                "He optimized transfer learning workflows for rapid deployment in AI systems",
                "Optimized transfer learning workflows for rapid deployment in AI systems"
            ]
        },
        {
            "question": "What is Paul’s experience with Reinforcement Learning?",
            "context": "Paul worked on Reinforcement Learning algorithms for autonomous mobile robots during his time at Utah Valley University. He focused on developing agents capable of decision-making in dynamic environments using Q-Learning and Deep Q-Networks.",
            "expected": [
                "He focused on developing agents capable of decision-making in dynamic environments",
                "Developing agents capable of decision-making in dynamic environments"
            ]
        },
        {
            "question": "How has Paul leveraged NLP in his projects?",
            "context": "Paul is well versed with Natural Language Processing, having developed conversational AI systems using Transformer-based models like GPT. His projects include chatbots for customer support, Q&A assistant, and advanced text summarization tools.",
            "expected": ["He has implemented GPT transformer-based models",
                         "conversational AI for robotics and chatbots",
                         "advanced text summarization tools",
                         "advanced text summarization tools.",
                         "tools for advanced text summarization"
                         ]
        },
        {
            "question": "What role has Anomaly Detection played in Paul’s work?",
            "context": "Paul implemented Anomaly Detection techniques in cybersecurity applications to identify fraud and unusual network behavior. He used autoencoders and statistical methods to improve system reliability.",
            "expected": [
                "Improve system reliability",
                "Paul implemented Anomaly Detection techniques in cybersecurity applications",
                "identify fraud and unusual network behavior"
            ]
        },
        {
            "question": "How has Paul used Time Series Analysis?",
            "context": "Paul has applied Time Series Analysis in financial forecasting, developing models to predict stock price movements and economic trends. He has experience using ARIMA, LSTM networks, and seasonal decomposition techniques.",
            "expected": [
                "Time Series Analysis in financial forecasting",
                "developing models to predict stock price movements and economic trends",
                "seasonal decomposition techniques"
            ]
        },
        {
            "question": "What is Paul’s expertise in Computer Vision?",
            "context": "Paul has developed Computer Vision models for object detection and image segmentation. His projects include interactive games with an AI, like 'rock, paper, scissors' and applying OpenCV and TensorFlow for object tracking and facial image recognition.",
            "expected": [
                "Object detection and image segmentation",
                "interactive games with an AI like 'rock, paper, scissors'",
                "object tracking",
                "facial image recognition",
                "applying OpenCV and TensorFlow"
            ]
        },
        {
            "question": "What types of Neural Networks has Paul worked with?",
            "context": "Paul has worked with convolutional neural networks (CNNs) for computer vision tasks like facial and object detection, and recurrent neural networks (RNNs) and long short-term memory networks (LSTMs) for speech recognition in humanoid robotics. He has also developed and implemented policies using neural networks for decision-making tasks.",
            "expected": [
                "convolutional neural networks (CNNs) for computer vision tasks",
                "facial and object detection",
                "recurrent neural networks (RNNs)",
                "long short-term memory networks (LSTMs)",
                "speech recognition in humanoid robotics",
                "developed and implemented policies using neural networks",
                "recurrent neural networks",
                "long short-term memory networks",
                "recurrent neural networks",
                "convolutional neural networks"
            ]
        },
        {
            "question": "What is Paul’s experience with Autonomous Mobile Robots?",
            "context": "Paul led a project on a pair of humanoid Autonomous Mobile Robots at UVU, focusing on navigation and decision-making using reinforcement learning and sensor fusion techniques. He optimized the robots for university recruitment events, a promotional film, and a gala fund raiser.",
            "expected": [
                "Autonomous Mobile Robots at UVU",
                "navigation and decision-making using reinforcement learning",
                "sensor fusion techniques",
                "optimized the robots for university recruitment events",
                "navigation and decision-making",
                "Humanoid Robotics"
            ]
        },
        {
            "question": "What are Paul’s accomplishments in Conversational AI?",
            "context": "Paul developed a portfolio chatbot using a custom-trained DistilBERT model integrated with Flask. He also worked on multi-turn dialogue systems for virtual assistants in customer support domains.",
            "expected": [
                "a portfolio chatbot using a custom-trained DistilBERT model",
                "worked on multi-turn dialogue systems for virtual assistants in customer support domains"
            ]
        },
        {
            "question": "What is the subject?",
            "context": "Paul Coleman has a diverse skill set, including AI & Machine Learning and web development",
            "expected": ["AI & Machine Learning", "web development"]
        },
    ]

def test_qa_pipeline():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Convert test cases to a dictionary suitable for Dataset
    data = {
        "question": [case["question"] for case in test_cases],
        "context": [case["context"] for case in test_cases]
    }
    # Keep the expected field separate for evaluation
    expected_answers = [case["expected"] for case in test_cases]

    # Create a Dataset object
    dataset = Dataset.from_dict(data)

    qa_pipeline = pipeline("question-answering", model="../models/fine_tuned_model",
                           tokenizer="../models/fine_tuned_model", local_files_only=True, device=device)

    # Run the pipeline on the dataset
    results = qa_pipeline(dataset, batch_size=8)

    # Evaluate results
    for i, result in enumerate(results):
        case = test_cases[i]
        answer = result["answer"]
        expected = expected_answers[i]

        # Log the best match and similarity score
        print(f"Question: {case['question']}")
        print(f"Expected: {expected}")
        print(f"Got: {answer}")

        # This is optional; it's called Fuzzy Matching, and it allows for variations in phrasing.
        scores = [(exp, SequenceMatcher(None, answer, exp).ratio()) for exp in expected]
        best_match = max(scores, key=lambda x: x[1])
        print(f"Best Match: {best_match[0]} with score {best_match[1]:.2f}\n")

    def is_match(answer, expected, threshold=0.8):
        """Check if answer matches expected with a similarity threshold."""
        return SequenceMatcher(None, answer, expected).ratio() > threshold

    for i, case in enumerate(test_cases):
        result = results[i]
        scores = [
            (exp, SequenceMatcher(None, result["answer"], exp).ratio())
            for exp in expected_answers[i]
        ]
        best_match = max(scores, key=lambda x: x[1])

        print(f"Question: {case['question']}")
        print(f"Expected: {expected_answers[i]}")
        print(f"Got: {result['answer']}")
        print(f"Best Match: {best_match[0]} with score {best_match[1]:.2f}")
        print()

        if best_match[1] < 0.9:
            print(f"Warning: Low match score ({best_match[1]:.2f}) for question: {case['question']}")
        assert any(is_match(result["answer"], exp) for exp in expected_answers[i]), \
            f"Failed for question: {case['question']}\nExpected one of: {expected_answers[i]}\nGot: {result['answer']}"


# Run the test
test_qa_pipeline()

