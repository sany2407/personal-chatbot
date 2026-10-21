from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    """Accepts either a plain ``question`` or an OpenAI-style ``messages`` array."""

    question: str | None = Field(default=None, min_length=2, max_length=2000)
    messages: list[ChatMessage] | None = None
    session_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def _resolve_question(self) -> "ChatRequest":
        if self.question and self.question.strip():
            return self
        if self.messages:
            for message in reversed(self.messages):
                if message.role == "user" and message.content.strip():
                    self.question = message.content.strip()
                    return self
        raise ValueError(
            "Provide 'question' or a 'messages' array containing a user message."
        )


class Source(BaseModel):
    title: str
    source: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[Source] = []
    grounded: bool
