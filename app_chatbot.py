from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import re
import torch
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

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
    # Sanitize question and context using regex
    sanitized_question = re.sub(r'\s+', ' ', faq_request.question)[:100]
    sanitized_context = re.sub(r'\s+', ' ', faq_request.context)[:100]

    # Log the action (optional logging logic)
    app.logger.info({
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
        raise HTTPException(status_code=500, detail=f"Error processing the question: {str(e)}")
