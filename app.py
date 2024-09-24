import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
import os
import re
import asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio.client import Redis
from fastapi_cache.decorator import cache
from pydantic import ValidationError
import platform

# Check the system architecture and set the appropriate quantization engine
if platform.machine() == "x86_64":  # For x86 architecture
    if "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"
elif platform.machine() in ["arm64", "aarch64"]:  # For ARM-based architectures like Apple Silicon
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"

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


# Load the Model at Startup
class ModelManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.tokenizer = None
        self.model = None

    async def load_model(self):
        async with self.lock:
            if self.model is None or self.tokenizer is None:
                try:
                    # Load tokenizer and model
                    self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                    self.model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2')

                    # Apply dynamic quantization (architecture-agnostic)
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("ALBERT model and tokenizer loaded and quantized on CPU")
                except Exception as e:
                    logger.error(f"Failed to load the model and tokenizer: {e}")
                    raise

model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    await model_manager.load_model()

    # Redis setup for caching, ensuring it expects byte responses
    redis = Redis.from_url("redis://red-cror6njqf0us73eeqr0g:6379", decode_responses=False)

    # Initialize FastAPI cache with Redis backend
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


@app.post("/faq/")
@cache(expire=60)
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize inputs
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        # Validate inputs explicitly
        if not sanitized_question:
            raise HTTPException(status_code=422, detail="`question` cannot be empty.")
        if not sanitized_context:
            raise HTTPException(status_code=422, detail="`context` cannot be empty.")

        logger.info({
            'action': 'faq_pipeline_called',
            'question': sanitized_question,
            'context_snippet': sanitized_context[:50] + '...' if len(sanitized_context) > 50 else sanitized_context
        })

        # Ensure the model and tokenizer are loaded
        if model_manager.model is None or model_manager.tokenizer is None:
            await model_manager.load_model()

        # Tokenize the inputs
        inputs = model_manager.tokenizer(sanitized_question, sanitized_context, return_tensors="pt")

        # Run model inference using positional arguments
        def model_inference(input_ids, attention_mask):
            return model_manager.model(input_ids=input_ids, attention_mask=attention_mask)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model_inference, inputs["input_ids"], inputs["attention_mask"])

        # Extract the answer from the model's output
        start_scores, end_scores = result.start_logits, result.end_logits

        # Ensure that start_scores and end_scores are tensors
        all_tokens = model_manager.tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1

        # Handle empty or invalid answers
        if start_idx >= len(all_tokens) or end_idx > len(all_tokens):
            logger.error("Invalid model output: start or end index out of bounds.")
            raise HTTPException(status_code=500, detail="Model produced invalid output.")

        answer = ' '.join(all_tokens[start_idx:end_idx])

        if not answer:
            logger.error("Model did not return an answer.")
            raise HTTPException(status_code=500, detail="Model failed to provide an answer.")

        return {"answer": answer}

    except HTTPException as http_exc:
        raise http_exc
    except ValidationError as val_err:
        logger.error(f"Validation error: {val_err}")
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@app.get("/")
async def health_check():
    return {"status": "healthy"}
