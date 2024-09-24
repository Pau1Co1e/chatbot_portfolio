import sys
import os

# Add the parent directory to the Python path so that app can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Now this import should work
from fastapi.testclient import TestClient

client = TestClient(app)

def test_faq_endpoint():
    response = client.post("/faq/", json={"question": "What is your name?", "context": "I am testing the chatbot."})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)

def test_healthcheck():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "healthy"}


def test_faq_invalid_input():
    response = client.post("/faq/", json={"question": "", "context": ""})
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "`question` cannot be empty."


def test_faq_long_input():
    long_context = "a" * 2000  # Exceeds the 1000-character limit
    response = client.post("/faq/", json={"question": "What is your name?", "context": long_context})
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data
    assert any(item["msg"] == "ensure this value has at most 1000 characters" for item in data["detail"])

def test_faq_output_format():
    response = client.post("/faq/", json={"question": "What is your name?", "context": "Testing context."})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0  # Ensure the answer is not empty