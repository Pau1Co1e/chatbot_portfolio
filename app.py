import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os
import re
import asyncio

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

                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

                    self.pipeline = pipeline(
                        "question-answering",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if device.type != "cpu" else -1
                    )
                    logger.info(f"Custom model and tokenizer loaded from {model_path} on {device}")
                except Exception as e:
                    logger.error(f"Failed to load model or tokenizer. Error: {e}")
                    raise HTTPException(status_code=500, detail="Failed to load model.")


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
    return sanitized[:max_length]


@app.post("/faq/")
async def call_faq_pipeline(faq_request: FAQRequest):
    try:
        # Sanitize inputs
        sanitized_question = sanitize_text(faq_request.question, max_length=200)
        sanitized_context = sanitize_text(faq_request.context, max_length=1000)

        if not sanitized_question:
            raise HTTPException(status_code=422, detail="`question` cannot be empty.")
        if not sanitized_context:
            raise HTTPException(status_code=422, detail="`context` cannot be empty.")

        logger.info({
            'action': 'faq_pipeline_called',
            'question': sanitized_question,
            'context_snippet': sanitized_context[:50] + '...' if len(sanitized_context) > 50 else sanitized_context
        })

        # Branching logic based on keywords
        keywords = {
            "experience": get_experience_response,
            "skills": get_skills_response,
            "education": get_education_response,
            "projects": get_projects_response
        }
        for key, response_function in keywords.items():
            if key in sanitized_question.lower():
                return await response_function()

        if model_manager.pipeline is None:
            await model_manager.load_model()

        # Run model inference
        result = await asyncio.to_thread(
            model_manager.pipeline,
            {"question": sanitized_question, "context": sanitized_context}
        )
        logger.info(f"Model result: {result}")

        answer = result.get("answer", "No answer found.")
        return {"answer": answer}

    except HTTPException as http_exc:
        raise http_exc
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
                  "and an AI chatbot application integrated into my personal portfolio website. These projects highlight my "
                  "expertise in machine learning, mathematics, and practical AI applications."
    }


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
