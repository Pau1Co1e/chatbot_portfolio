import logging
import nltk
import os
import zipfile
from nltk.data import find, FileSystemPathPointer

# Add NLTK data path
# nltk.data.path.append('./nltk_data')

# Patch nltk.data.find to handle the punkt_tab error
def patched_find(resource_name, *args, **kwargs):
    if resource_name == "tokenizers/punkt_tab/english":
        # Redirect punkt_tab to the actual punkt/english.pickle file
        punkt_path = "./nltk_data/tokenizers/punkt/english.pickle"
        if os.path.exists(punkt_path):
            return FileSystemPathPointer(punkt_path)
        else:
            raise LookupError(f"Punkt resource not found at: {punkt_path}")
    return find(resource_name, *args, **kwargs)

nltk.data.find = patched_find

def initialize_nltk_resources():
    nltk.data.path.append('./nltk_data')

    # Ensure the punkt resource exists
    punkt_dir_path = './nltk_data/tokenizers/punkt'
    if not os.path.exists(f"{punkt_dir_path}/english.pickle"):
        logging.error("Expected punkt resource not found.")
        raise Exception("Punkt resource is missing or incomplete.")

    # If punkt.zip exists but punkt directory doesn't, extract it
    punkt_zip_path = './nltk_data/tokenizers/punkt.zip'
    if os.path.exists(punkt_zip_path) and not os.path.exists(punkt_dir_path):
        try:
            with zipfile.ZipFile(punkt_zip_path, 'r') as zip_ref:
                zip_ref.extractall('./nltk_data/tokenizers')
            logging.info("Extracted punkt.zip successfully.")
        except Exception as e:
            logging.error(f"Failed to extract punkt.zip: {e}")
            raise Exception("Failed to initialize NLTK punkt resources.")

    # Try downloading punkt if necessary
    try:
        nltk.download('punkt', download_dir='./nltk_data')
    except Exception as e:
        logging.error(f"Failed to download NLTK punkt resource: {e}")
        raise Exception("Failed to initialize NLTK resources.")

initialize_nltk_resources()

import torch
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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Environment Variables
PORT = int(os.getenv("PORT", 8000))
FLASK_APP_ORIGIN = os.getenv("FLASK_APP_ORIGIN", "https://codebloodedfamily.com")

if not FLASK_APP_ORIGIN:
    logging.warning("FLASK_APP_ORIGIN is not set. Using default: https://codebloodedfamily.com")

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

# Initialize stopwords
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    logger.warning("NLTK stopwords not found. Using an empty set.")
    stop_words = set()

# Sentence-BERT Model for Semantic Similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

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
                    model_path = os.getenv("MODEL_PATH", "/opt/render/models/fine_tuned_albert")

                    if not os.path.exists(model_path):
                        raise ValueError(f"Model path does not exist: {model_path}")
                    logger.info(f"Loading model from: {model_path} on device: {device}")

                    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    context: str = Field(..., max_length=1000)

# Helper Functions
def sanitize_text(text: str, max_length: int = 1000) -> str:
    sanitized = re.sub(r'\s+', ' ', text).strip()
    return sanitized[:max_length]

def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)  # Decompose characters (e.g., é -> e + ́)
    text = "".join([c for c in text if not unicodedata.combining(c)])  # Remove diacritics
    return text.lower()

def extract_number(text):
    try:
        return w2n.word_to_num(text)
    except ValueError:
        tokens = text.split(',')
        if len(tokens) > 1:
            return len(tokens)
        return None

def compute_semantic_similarity(predicted, expected_list):
    predicted_embedding = semantic_model.encode(predicted, convert_to_tensor=True)
    expected_embeddings = semantic_model.encode(expected_list, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(predicted_embedding, expected_embeddings)
    return similarities.max().item()

@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        if not sanitized_question:
            raise HTTPException(status_code=422, detail="`question` cannot be empty.")
        if not sanitized_context:
            raise HTTPException(status_code=422, detail="`context` cannot be empty.")

        # logger.info({
        #     'action': 'faq_pipeline_called',
        #     'question': sanitized_question,
        #     'context_snippet': sanitized_context[:50] + '...' if len(sanitized_context) > 50 else sanitized_context
        # })

        if model_manager.pipeline is None:
            await model_manager.load_model()

        result = await asyncio.to_thread(
            model_manager.pipeline,
            {"question": sanitized_question, "context": sanitized_context}
        )
        logger.info(f"Model result: {result}")

        pred_answer = result.get("answer", "No answer found.")

        # Post-process and compute semantic similarity
        pred_tokens = [word for word in word_tokenize(normalize_text(pred_answer)) if word not in stop_words]
        expected_answers = [faq_request.context]
        expected_tokens_list = [
            [word for word in word_tokenize(normalize_text(ans)) if word not in stop_words]
            for ans in expected_answers
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
