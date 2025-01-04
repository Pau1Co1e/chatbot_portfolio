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
Paul Coleman is working towards a Master of Science in Computer Science from Utah Valley University and is expected to graduate Fall 2025...
"""  # Truncated for brevity.

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

# Hugging Face Tokenizer and Sentence-BERT Model (Use smaller models for efficiency)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
semantic_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

response_cycle = cycle([
    "This is CodeBloodedBot you're speaking to, but the person who created me is named Paul Coleman.",
    "I’m CodeBloodedBot, your friendly assistant! My creator is Paul Coleman, but I like to think I do all the heavy lifting!",
    "I'm called CodeBloodedBot, and I'm an AI model built by Paul to answer questions you have about him.",
    "CodeBloodedBot at your service! What questions do you have for me?",
])


# Load the Model On-Demand
class ModelManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.pipeline = None

    async def load_model(self):
        async with self.lock:
            if self.pipeline is None:
                try:
                    device = get_device()
                    model_path = os.getenv("MODEL_PATH", "distilbert-base-uncased")

                    logger.info(f"Loading model from: {model_path} on device: {device}")
                    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

                    self.pipeline = pipeline(
                        "question-answering",
                        model=model,
                        tokenizer=tokenizer,
                        device=-1  # Force CPU for memory savings
                    )
                    logger.info("Model successfully loaded.")
                except Exception as e:
                    logger.error(f"Failed to load model. Error: {e}")
                    raise HTTPException(status_code=500, detail="Failed to load model.")


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    logger.info("Chatbot disabled. Model will load on demand.")


# Pydantic Model for FAQ Request
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=200)


# Helper Functions
def sanitize_text(text: str, max_length: int = 1000) -> str:
    sanitized = re.sub(r'\s+', ' ', text).strip()
    return sanitized[:max_length]


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    return text.lower()


@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        sanitized_question = sanitize_text(faq_request.question, max_length=200)

        if sanitized_question in IDENTITY_QUESTIONS:
            response = next(response_cycle)
            return {"answer": response}

        if not model_manager.pipeline:
            logger.info("Loading model on demand.")
            await model_manager.load_model()

        result = await asyncio.to_thread(
            model_manager.pipeline,
            {"question": sanitized_question, "context": DEFAULT_CONTEXT}
        )
        logger.info(f"Model result: {result}")

        pred_answer = result.get("answer", "No answer found.")
        return {"answer": pred_answer}

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error.")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.pipeline is not None
    }


def get_device():
    return torch.device("cpu")  # Enforce CPU for scaling down
