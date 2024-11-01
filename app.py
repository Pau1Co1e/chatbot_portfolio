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
from fastapi_cache.decorator import cache
from pydantic import ValidationError

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
        self.pipeline = None

    async def load_model(self):
        async with self.lock:
            if self.pipeline is None:
                try:
                    self.pipeline = pipeline(
                        "question-answering",
                        model="distilbert-base-cased-distilled-squad",
                        device=-1  # CPU only
                    )
                    logger.info(f"Model loaded on CPU")
                except Exception as e:
                    logger.error(f"Failed to load the model: {e}")
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

        # Keyword detection for branching
        keywords = {
            "experience": get_experience_response,
            "skills": get_skills_response,
            "education": get_education_response,
            "projects": get_projects_response
        }

        # Check for keywords in the question and branch accordingly
        for key, response_function in keywords.items():
            if key in sanitized_question.lower():
                return await response_function()

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
    except ValidationError as val_err:
        logger.error(f"Validation error: {val_err}")
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# Response functions for keyword-based branching
async def get_experience_response():
    return {
        "answer": "I have over 5 years of experience in software development, focusing on artificial intelligence, "
                  "machine learning, full-stack development, and robotics. I've worked extensively with frameworks like Flask, "
                  "ASP.NET, and FastAPI, and contributed to projects ranging from financial prediction models to "
                  "deepfake detection systems."
    }

async def get_skills_response():
    return {
        "answer": "My core skills include Python programming, machine learning (using frameworks such as TensorFlow "
                  "and PyTorch), full-stack web development with technologies like Flask, ASP.NET, and SQLAlchemy, "
                  "and project development in AI, finance, and cybersecurity."
    }

async def get_education_response():
    return {
        "answer": "I am currently pursuing my Master's degree in Computer Science at Utah Valley University, "
                  "focusing on AI and machine learning for finance and cybersecurity. I graduated with a BSc in "
                  "Computer Science in 2022."
    }

async def get_projects_response():
    return {
        "answer": "Some of my notable projects include a fractal dimension calculator, a stock market analysis predictor, "
                  "and a chatbot application integrated into my personal portfolio website. These projects highlight my "
                  "expertise in machine learning, mathematics, and practical AI applications."
    }

@app.get("/")
async def health_check():
    return {"status": "healthy"}
