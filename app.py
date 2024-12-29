import logging
import os
import re
import unicodedata
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from sentence_transformers import SentenceTransformer, util
from word2number import w2n
import torch
from itertools import cycle


# Environment Variables
PORT = int(os.getenv("PORT", 8000))
FLASK_APP_ORIGIN = os.getenv("FLASK_APP_ORIGIN", "https://codebloodedfamily.com")

if not FLASK_APP_ORIGIN:
    logging.warning("FLASK_APP_ORIGIN is not set. Using default: https://codebloodedfamily.com")
    
IDENTITY_QUESTIONS = [
    "What is your name?",
    "Who are you?",
    "What's your name?",
    "Tell me about yourself."
]

DEFAULT_CONTEXT = """
Paul Coleman is working towards a Master of Science in Computer Science from Utah Valley University and is expected to graduate Fall 2025; He holds a Bachelor of Science in Computer Science and graduated August 2022 with a GPA of 3.26. 
Paul Coleman has a Programmer Certification from Utah Valley University that he earned in 2020, was on the Dean's List in 2020 and 2021, and held a CompTIA A+ Certification from Dell Inc. 
Paul Coleman is skilled in AI & Machine Learning, model development, algorithm design, Natural Language Processing, web development with .NET, Flask and JavaScript, and scalable systems design. 
Paul is familiar with and has experience in all of the most popular programming languages such as C#, Java, Javascript, and C++; however his expertise is with Python and C#. 
Paul Coleman worked as a Full Stack Software Engineer at ResNexus from September 2023 to December 2023; He developed backend APIs and frontend components using the Microsoft tech stack for hotel and rental management products. 
Paul Coleman organized a local donation event for the Northern Utah Habitat for Humanity Clothing Drive in June 2014, supporting children and families in the Dominican Republic. 
Paul Coleman has experience with Interpretable Machine Learning techniques like LIME and SHAP; He has applied these methods to prediction models to explain outputs in an understandable way. 
Paul has implemented Transfer Learning in NLP applications, utilizing pre-trained language models such as BERT and ALBERT for sentiment analysis and chatbots; He optimized transfer learning workflows for rapid deployment in AI systems. 
Paul worked on Reinforcement Learning algorithms for autonomous mobile robots during his time at Utah Valley University; He focused on developing agents capable of decision-making in dynamic environments using Q-Learning and Deep Q-Networks. 
Paul specializes in Natural Language Processing, having developed conversational AI systems using Transformer-based models like GPT; His projects include chatbots for customer support and advanced text summarization tools. 
Paul implemented Anomaly Detection techniques in cybersecurity applications to identify fraud and unusual network behavior; He used autoencoders and statistical methods to improve system reliability. 
Paul has applied Time Series Analysis in financial forecasting, developing models to predict stock price movements and economic trends; He has experience using ARIMA, LSTM networks, and seasonal decomposition techniques. 
Paul has developed Computer Vision models for object detection and image segmentation; His projects include building a vision-based system for real time facial recognition and applying OpenCV and TensorFlow for object tracking. 
Paul has worked extensively with convolutional neural networks (CNNs) for image analysis and recurrent neural networks (RNNs) for sequential data; His expertise includes designing custom architectures for real-world applications. 
Paul led a project on humanoid Autonomous Mobile Robots at UVU, focusing on navigation, coordination and decision-making using reinforcement learning and sensor fusion techniques; He optimized the robots for interactive use at university recruitment events. 
Paul developed a portfolio chatbot using a fine tuned custom-trained ALBERT model integrated with Flask; He also worked on multi-turn dialogue systems for virtual assistants.
Whenever Paul has spare time, he loves to spend it with his twins and family; Paul also enjoys biking, video games, guitar, and staying current with novel innovations and technologies.
"""

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("faq_pipeline")

# Static stopwords list
stop_words = set(["the", "is", "in", "and", "to", "a", "of", "for", "on", "with", "as", "by", "at", "it"])

# Hugging Face Tokenizer and Sentence-BERT Model
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_fast=True)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

response_cycle = cycle([
    "This is CodeBloodedBot you're speaking to, but the person who created me is named Paul Coleman.",
    "I’m CodeBloodedBot, your friendly assistant! My creator is Paul Coleman, but I like to think I do all the heavy lifting!",
    "I'm called CodeBloodedBot, and I'm an AI model built by Paul to answer questions you have about him.",
    "CodeBloodedBot at your service! What questions do you have for me?",
])

# Load the Model at Startup
class ModelManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.pipeline = None

    async def load_model(self):
        async with self.lock:
            if self.pipeline is None:
                try:
                    device = get_device()
                    model_path = os.getenv("MODEL_PATH", "albert-base-v2")

                    logger.info(f"Loading model from: {model_path} on device: {device}")
                    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

                    self.pipeline = pipeline(
                        "question-answering",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if device.type != "cpu" else -1
                    )
                    logger.info("Model successfully loaded.")
                except Exception as e:
                    logger.error(f"Failed to load model. Error: {e}")
                    raise HTTPException(status_code=500, detail="Failed to load model.")


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    await model_manager.load_model()


# Pydantic Model for FAQ Request
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=200)
    # context: str = Field(..., max_length=1000)


# Helper Functions
def sanitize_text(text: str, max_length: int = 1000) -> str:
    sanitized = re.sub(r'\s+', ' ', text).strip()
    return sanitized[:max_length]


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)  # Decompose characters (e.g., é -> e + ́)
    text = "".join([c for c in text if not unicodedata.combining(c)])  # Remove diacritics
    return text.lower()


def tokenize_text(text):
    return tokenizer.tokenize(text)


def compute_semantic_similarity(predicted, expected_list):
    predicted_embedding = semantic_model.encode(predicted, convert_to_tensor=True)
    expected_embeddings = semantic_model.encode(expected_list, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(predicted_embedding, expected_embeddings)
    return similarities.max().item()


@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize the question
        sanitized_question = sanitize_text(faq_request.question, max_length=200)

        # Handle identity-specific questions
        if sanitized_question in IDENTITY_QUESTIONS:
            response = next(response_cycle)
            return {"answer": response}

        if not DEFAULT_CONTEXT:
            raise HTTPException(status_code=500, detail="Default context is missing.")
            
        sanitized_context = sanitize_text(DEFAULT_CONTEXT, max_length=1000)

        if not sanitized_question:
            raise HTTPException(status_code=422, detail="`question` cannot be empty.")

        # Ensure model pipeline is loaded
        if model_manager.pipeline is None:
            await model_manager.load_model()

        # Call the Q&A pipeline with the sanitized inputs
        result = await asyncio.to_thread(
            model_manager.pipeline,
            {"question": sanitized_question, "context": sanitized_context}
        )
        logger.info(f"Model result: {result}")

        pred_answer = result.get("answer", "No answer found.")

        # Post-process and compute semantic similarity
        pred_tokens = [word for word in pred_answer.lower().split() if word not in stop_words]
        expected_answers = [sanitized_context]
        expected_tokens_list = [
            [word for word in context.lower().split() if word not in stop_words]
            for context in expected_answers
        ]

        overlap_ratios = [
            len(set(pred_tokens).intersection(set(expected_tokens))) / max(len(expected_tokens), 1)
            for expected_tokens in expected_tokens_list
        ]
        max_overlap = max(overlap_ratios)

        semantic_similarity = compute_semantic_similarity(pred_answer, expected_answers)

        logger.info({
            'pred_answer': pred_answer,
            'max_overlap': max_overlap,
            'semantic_similarity': semantic_similarity
        })

        return {
            "answer": pred_answer,
            "max_overlap": max_overlap,
            "semantic_similarity": semantic_similarity
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.pipeline is not None
    }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
