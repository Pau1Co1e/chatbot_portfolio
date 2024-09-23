import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline
import os
import re
import asyncio

# Environment Variables
PORT = int(os.getenv("PORT", 5000))
FLASK_APP_ORIGIN = "https://codebloodedfamily.com"  # Flask app's origin

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware for Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FLASK_APP_ORIGIN],  # Allow requests from Flask app's origin
    allow_credentials=True,
    allow_methods=["POST"],  # Only allow POST if that's the only endpoint
    allow_headers=["*"],
)

# Logging Configuration for Production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("faq_pipeline")

# Load the Model at Startup
class ModelManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.pipeline = None

    async def load_model(self):
        async with self.lock:
            if self.pipeline is None:
                try:
                    self.pipeline = pipeline(
                        "question-answering",
                        model="distilbert-base-cased-distilled-squad",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
                except Exception as e:
                    logger.error(f"Failed to load the model: {e}")
                    raise

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    await model_manager.load_model()

# Pydantic Model for FAQ Request
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=200, description="The question to answer")
    context: str = Field(..., max_length=1000, description="The context for the question")

# Helper function for sanitization
def sanitize_text(text: str, max_length: int = 1000) -> str:
    """
    Sanitizes input text by limiting length and removing excessive whitespace.
    """
    sanitized = re.sub(r'\s+', ' ', text).strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized

# Endpoint for FAQ Model
@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize inputs
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        # Log the incoming request
        logger.info({
            'action': 'faq_pipeline_called',
            'question': sanitized_question,
            'context_snippet': (sanitized_context[:50] + '...') if len(sanitized_context) > 50 else sanitized_context
        })

        # Prepare inputs for the model
        inputs = {
            "question": sanitized_question,
            "context": sanitized_context
        }

        # Perform model inference with proper error handling
        if model_manager.pipeline is None:
            await model_manager.load_model()

        # Use asyncio to run the blocking inference in a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model_manager.pipeline, inputs)

        # Validate the model's response
        if "answer" not in result:
            logger.error("Model did not return an answer.")
            raise HTTPException(status_code=500, detail="Model failed to provide an answer.")

        return {"answer": result["answer"]}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")
