import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.chat import ChatRequest

client = TestClient(app)


def test_openai_models_route():
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"]


def test_chat_request_accepts_plain_question():
    assert ChatRequest(question="What tech?").question == "What tech?"


def test_chat_request_accepts_openai_messages():
    request = ChatRequest(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What tech is listed?"},
        ]
    )
    assert request.question == "What tech is listed?"


def test_chat_request_requires_a_question():
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_endpoint_rejects_empty_body():
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422
