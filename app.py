import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os
import re
import asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio.client import Redis
from fastapi_cache.decorator import cache

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

# Model Manager with Quantization and Lazy Loading
class ModelManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.pipeline = None
        self.tokenizer = None

    async def load_model(self):
        async with self.lock:
            if self.pipeline is None:
                try:
                    # Load the tokenizer and model with quantization
                    logger.info("Loading model...")
                    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

                    # Apply dynamic quantization to the model for memory optimization
                    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

                    self.pipeline = pipeline(
                        "question-answering",
                        model=model,
                        tokenizer=self.tokenizer,
                        device=-1  # CPU only
                    )
                    logger.info(f"Model loaded and quantized on CPU")
                except Exception as e:
                    logger.error(f"Failed to load the model: {e}")
                    raise

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    # Lazy loading of the model, no longer loading it immediately on startup
    # Redis setup for caching, ensuring it expects byte responses
    redis = Redis.from_url("redis://red-cror6njqf0us73eeqr0g:6379", decode_responses=False)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Pydantic Model for FAQ Request
class FAQRequest(BaseModel):
    question: str = Field(..., max_length=200)
    context: str = Field(..., max_length=1000)

# Helper function for sanitization
def sanitize_text(text: str, max_length: int = 1000) -> str:
    sanitized = re.sub(r'\s+', ' ', text).strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized

# Caching Decorator
@app.post("/faq/")
@cache(expire=60)  # Cache the result for 60 seconds
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        logger.info({
            'action': 'faq_pipeline_called',
            'question': sanitized_question,
            'context_snippet': sanitized_context[:50] + '...' if len(sanitized_context) > 50 else sanitized_context
        })

        inputs = {"question": sanitized_question, "context": sanitized_context}

        if model_manager.pipeline is None:
            await model_manager.load_model()

        # Run model inference
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model_manager.pipeline, inputs)

        if "answer" not in result:
            logger.error("Model did not return an answer.")
            raise HTTPException(status_code=500, detail="Model failed to provide an answer.")

        return {"answer": result["answer"]}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")
