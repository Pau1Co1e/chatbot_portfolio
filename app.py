import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline
import os
import re
import asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio.client import Redis
import gc  # Import garbage collection

# Environment Variables
PORT = int(os.getenv("PORT", 8000))
FLASK_APP_ORIGIN = os.getenv("FLASK_APP_ORIGIN", "https://codebloodedfamily.com")

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

# Lazy Model Loading & Memory Management Improvements
class ModelManager:
    def __init__(self):
        self.lock = asyncio.Lock()  # Async lock to avoid concurrent model loading
        self.pipeline = None

    async def load_model(self):
        if self.pipeline is None:
            async with self.lock:
                # Check again after acquiring lock to avoid duplicate loading
                if self.pipeline is None:
                    try:
                        # Load the question-answering model lazily using transformers pipeline
                        self.pipeline = pipeline(
                            "question-answering",
                            model="distilbert-base-cased-distilled-squad",
                            device=-1  # Use CPU to reduce memory (no GPUs)
                        )
                        logger.info("Model successfully loaded on CPU.")
                    except Exception as e:
                        logger.error(f"Failed to load the model: {e}")
                        raise

    def clear_model(self):
        """
        Method to unload the model from memory to free up space after handling a request.
        """
        if self.pipeline:
            self.pipeline = None
            logger.info("Model cleared from memory to reduce usage.")
            gc.collect()  # Explicit garbage collection to free memory


model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    # We do not load the model on startup to save memory; it will load on-demand.
    # Redis setup for caching
    redis = Redis.from_url("redis://red-cror6njqf0us73eeqr0g:6379", decode_responses=False)

    # Initialize FastAPI cache with Redis backend
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Pydantic Model for FAQ Request
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=200)
    context: str = Field(..., max_length=1000)

# Helper function to sanitize inputs
def sanitize_text(text: str, max_length: int = 1000) -> str:
    """
    Sanitizes input text by removing excessive whitespace and truncating to the max length.
    """
    sanitized = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with one space and trim
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized

# Main API endpoint with caching and memory management
@app.post("/faq/")
@cache(expire=60)  # Cache results for 60 seconds
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize inputs
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        # Log the request
        logger.info({
            'action': 'faq_pipeline_called',
            'question': sanitized_question,
            'context_snippet': sanitized_context[:50] + '...' if len(sanitized_context) > 50 else sanitized_context
        })

        # Prepare inputs for the model
        inputs = {"question": sanitized_question, "context": sanitized_context}

        # Ensure model is loaded lazily
        if model_manager.pipeline is None:
            await model_manager.load_model()

        # Run model inference asynchronously
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model_manager.pipeline, inputs)

        # Check for valid response
        if "answer" not in result:
            logger.error("Model did not return an answer.")
            raise HTTPException(status_code=500, detail="Model failed to provide an answer.")

        # Clear model from memory after use to free up space
        model_manager.clear_model()

        return {"answer": result["answer"]}

    except HTTPException as http_exc:
        # Raise HTTP-specific exceptions
        raise http_exc
    except Exception as e:
        # Log and raise any unhandled exceptions
        logger.error(f"Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")
    finally:
        # Clear model after handling the request
        model_manager.clear_model()
        gc.collect()  # Explicit garbage collection

