import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import app.agent.graph as graph_module
from app.agent.memory import get_checkpointer
from app.core.config import get_settings


class FakeModel:
    """Records every prompt it receives and returns a deterministic answer."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(list(messages))
        return AIMessage(content=f"answer-{len(self.prompts)}")


class FakeStore:
    def similarity_search(self, query, k=4):
        return [
            Document(
                page_content="Built Python and FastAPI services for fleet tooling.",
                metadata={"title": "Profile", "source": "knowledge/profile.md"},
            )
        ]


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    get_settings.cache_clear()
    get_checkpointer.cache_clear()
    graph_module._graph = None
    yield
    graph_module._graph = None
    get_settings.cache_clear()
    get_checkpointer.cache_clear()


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
    monkeypatch.setattr(graph_module, "get_vectorstore", lambda: FakeStore())
    return model


def _turn(graph, question, session_id):
    graph.invoke(
        {"question": question, "session_id": session_id, "attempt": 0},
        config={"configurable": {"thread_id": session_id}},
    )


def test_history_is_replayed_for_the_same_session(memory_env, fake_model):
    graph = graph_module.get_graph()
    _turn(graph, "What technologies?", "s1")
    _turn(graph, "And the second project?", "s1")

    assert len(fake_model.prompts) == 2
    second_prompt = [m.content for m in fake_model.prompts[1]]
    assert any("What technologies?" in content for content in second_prompt)
    assert any("answer-1" in content for content in second_prompt)


def test_sessions_are_isolated(memory_env, fake_model):
    graph = graph_module.get_graph()
    _turn(graph, "Question in session one", "s1")
    _turn(graph, "Question in session two", "s2")

    second_prompt = [m.content for m in fake_model.prompts[1]]
    assert not any("Question in session one" in content for content in second_prompt)


def test_history_survives_a_restart(memory_env, fake_model):
    graph = graph_module.get_graph()
    _turn(graph, "What technologies?", "s1")

    # Simulate a process restart: new connection to the same SQLite file.
    get_checkpointer.cache_clear()
    graph_module._graph = None
    graph = graph_module.get_graph()
    _turn(graph, "And the second project?", "s1")

    second_prompt = [m.content for m in fake_model.prompts[1]]
    assert any("answer-1" in content for content in second_prompt)


def test_history_is_trimmed_to_the_token_budget(memory_env, fake_model, monkeypatch):
    monkeypatch.setenv("MAX_HISTORY_TOKENS", "1")
    get_settings.cache_clear()

    graph = graph_module.get_graph()
    _turn(graph, "What technologies?", "s1")
    _turn(graph, "And the second project?", "s1")

    second_prompt = [m.content for m in fake_model.prompts[1]]
    assert not any("What technologies?" in content for content in second_prompt)


def test_system_prompt_describes_the_history_correctly(memory_env, fake_model):
    graph = graph_module.get_graph()
    _turn(graph, "What technologies?", "s1")

    system_prompt = fake_model.prompts[0][0].content
    assert "conversation so far" in system_prompt
    assert "Never treat earlier assistant answers as evidence" in system_prompt
    assert "conversation context" not in system_prompt
