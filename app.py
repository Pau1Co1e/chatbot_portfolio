import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AlbertTokenizerFast
import os
import re
import asyncio
# from fastapi_cache2 import FastAPICache
# from fastapi_cache2.backends.redis import RedisBackend
# from fastapi_cache import FastAPICache
# from fastapi_cache.backends.redis import RedisBackend
# from redis.asyncio.client import Redis
# from fastapi_cache.decorator import cache
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
                    device = get_device()
                    model_path = "/opt/render/models/fine_tuned_albert"
                    tokenizer = AlbertTokenizerFast.from_pretrained(model_path)  # Ensure tokenizer compatibility
                    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

                    self.pipeline = pipeline(
                        "question-answering",
                        model=model_path,
                        tokenizer=model_path,
                        device=device
                    )
                    logger.info(f"Model loaded on device: {next(model.parameters()).device}")
                    logger.info(f"Model config: {model.config}")
                    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
                    logger.info(f"Custom model and tokenizer loaded from {model_path} on {device}")
                except Exception as e:
                    logger.error(f"Failed to load model or tokenizer. Error: {e}")
                    raise


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    await model_manager.load_model()

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
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize inputs
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        # Validate inputs
        if not sanitized_question or not sanitized_context:
            raise HTTPException(status_code=422, detail="`question` and `context` cannot be empty.")

        # Load model and tokenizer
        if model_manager.pipeline is None:
            await model_manager.load_model()

        model = model_manager.pipeline.model
        tokenizer = model_manager.pipeline.tokenizer

        # Tokenize inputs
        tokenized_inputs = tokenizer(
            sanitized_question,
            sanitized_context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Force CPU usage
        device = torch.device("cpu")
        model.to(device)
        tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

        # Run inference
        outputs = model(**tokenized_inputs)
        start_index = outputs.start_logits.argmax().item()
        end_index = outputs.end_logits.argmax().item()

        # Decode the predicted answer
        predicted_answer = tokenizer.decode(
            tokenized_inputs["input_ids"][0][start_index:end_index + 1]
        )
        logger.info(f"Predicted answer: {predicted_answer}")

        # Return the result
        return {"answer": predicted_answer}

    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        logger.error(f"Request details: question={faq_request.question}, context={faq_request.context}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

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
                  "and an AI chatbot application integrated into my personal portfolio website. These projects highlight my "
                  "expertise in machine learning, mathematics, and practical AI applications."
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy",
            "model_loaded": model_manager.pipeline is not None
            }

def get_device():
    # # Check for GPU (CUDA)
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # # Check for Metal (macOS GPU support)
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # # Default to CPU if no GPU is available
    # else:
    return torch.device("cpu")
