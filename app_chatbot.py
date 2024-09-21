import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline
import os
import re

# Environment Variables
PORT = os.getenv("PORT", 8000)
FASTAPI_CORS_ORIGINS = ["https://codebloodedfamily.com"]

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware for Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=FASTAPI_CORS_ORIGINS,  # Restrict to production domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Configuration for Production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("faq_pipeline")

# Load the Model
faq_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=0 if torch.cuda.is_available() else -1
)
logger.info(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Pydantic Model for FAQ Request
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=100, examples=["What is AI?"])
    context: str = Field(..., max_length=100, examples=["AI is the simulation of human intelligence."])

# Helper function for sanitization
def sanitize_text(text: str, max_length: int = 100) -> str:
    """
    Sanitizes input text by limiting length and removing excessive whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()[:max_length]

# Endpoint for FAQ Model
@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize inputs
        sanitized_question = sanitize_text(faq_request.question)
        sanitized_context = sanitize_text(faq_request.context)

        # Log the incoming request
        logger.info({
            'action': 'faq_pipeline_called',
            'question': sanitized_question,
            'context_snippet': sanitized_context
        })

        # Prepare inputs for the model
        inputs = {
            "question": sanitized_question,
            "context": sanitized_context
        }

        # Perform model inference
        with torch.no_grad():
            result = faq_pipeline(inputs)

        return {"answer": result["answer"]}

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
