import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline
import os

port = os.getenv("PORT", 8000)

# Initialize FastAPI app
# FastAPI app
app = FastAPI()
# Add CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],  # Adjust to specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Set up logging
logger = logging.getLogger("fastapi")
logging.basicConfig(level=logging.INFO)

# Load the model
faq_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=0 if torch.cuda.is_available() else -1
)

# Pydantic model for request data
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=100, examples=["What is AI?"])
    context: str = Field(..., max_length=100, examples=["AI is the simulation of human intelligence."])

# Endpoint for handling FAQ pipeline
@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    import re
    # Sanitize question and context using regex
    sanitized_question = re.sub(r'\s+', ' ', faq_request.question)[:100]
    sanitized_context = re.sub(r'\s+', ' ', faq_request.context)[:100]

    # Log the action using the logger we set up
    logger.info({
        'action': 'faq_pipeline_called',
        'question': sanitized_question,
        'context_snippet': sanitized_context
    })

    # Prepare inputs
    inputs = {
        "question": sanitized_question,
        "context": sanitized_context
    }

    # Perform inference using torch.no_grad()
    try:
        with torch.no_grad():
            result = faq_pipeline(inputs)
            del inputs
        return {"answer": result["answer"]}

    except Exception as e:
        # If something goes wrong, raise an HTTP 500 error
        logger.error(f"Error processing the question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the question: {str(e)}")
