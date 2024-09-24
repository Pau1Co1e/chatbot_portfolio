import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import os
import re
import asyncio

# Environment Variables
PORT = int(os.getenv("PORT", 8000))  # Updated default port to 8000
FLASK_APP_ORIGIN = os.getenv("FLASK_APP_ORIGIN", "https://codebloodedfamily.com")  # Made configurable

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware for Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from Flask app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
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
        """
        Lazy loading of the model. Loads the model only once and reuses it.
        Forces execution on CPU.
        """
        async with self.lock:
            if self.pipeline is None:
                try:
                    # Load model only if not already loaded
                    device = "cpu"  # Force CPU execution
                    self.pipeline = pipeline(
                        "question-answering",
                        model="distilbert-base-cased-distilled-squad",
                        device=-1  # Force to CPU by setting device=-1
                    )
                    logger.info(f"Model loaded on CPU")
                except Exception as e:
                    logger.error(f"Failed to load the model: {e}")
                    raise


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    # Lazy model loading; not loading model on startup anymore.
    logger.info("Application started. Model will be loaded upon first request.")


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


# Helper function for lazy model loading
async def get_model():
    """
    Ensures the model is loaded and returns it. If already loaded, returns the cached model.
    """
    if model_manager.pipeline is None:
        await model_manager.load_model()
    return model_manager.pipeline


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

        # Get the model (load it if not already loaded)
        pipeline = await get_model()

        # Use asyncio to run the blocking inference in a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pipeline, inputs)

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
