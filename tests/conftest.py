"""Pytest configuration and fixtures for mem0 integration tests."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Configure environment before any imports
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_DB"] = "test_db"
os.environ["POSTGRES_USER"] = "test_user"
os.environ["POSTGRES_PASSWORD"] = "test_pass"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["OPENAI_API_KEY"] = "sk-test-123"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"

# Mock database service before importing app modules
mock_database_service = MagicMock()
sys.modules["app.services.database"] = MagicMock(database_service=mock_database_service)

from app.core.config import settings
from app.schemas.chat import Message


@pytest.fixture
def mock_mem0_client():
    """Mock Mem0 client for testing."""
    client = MagicMock()

    # Mock add method
    client.add = MagicMock(return_value={"id": "memory_123"})

    # Mock search method
    client.search = MagicMock(return_value=[
        {
            "id": "mem_1",
            "data": "User likes Python programming",
            "metadata": {"type": "preference"}
        },
        {
            "id": "mem_2",
            "data": "User is a software engineer",
            "metadata": {"type": "fact"}
        }
    ])

    # Mock get_all method
    client.get_all = MagicMock(return_value=[
        {
            "id": "mem_1",
            "data": "User likes Python programming",
            "metadata": {"type": "preference"}
        },
        {
            "id": "mem_2",
            "data": "User is a software engineer",
            "metadata": {"type": "fact"}
        },
        {
            "id": "mem_3",
            "data": "User is detail-oriented",
            "metadata": {"type": "trait"}
        }
    ])

    # Mock update method
    client.update = MagicMock(return_value=True)

    # Mock delete method
    client.delete = MagicMock(return_value=True)

    return client


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing."""
    llm = AsyncMock()
    llm.model_name = "gpt-4o-mini"
    return llm


@pytest.fixture
def test_client():
    """Create a test client for FastAPI app."""
    pytest.skip("Database not available in test environment")


@pytest.fixture
def test_user_id() -> int:
    """Test user ID."""
    return 12345


@pytest.fixture
def test_session_id() -> str:
    """Test session ID."""
    return "test-session-12345"


@pytest.fixture
def sample_messages() -> list[Message]:
    """Sample messages for testing."""
    return [
        Message(role="user", content="Hi, my name is John and I love Python"),
        Message(
            role="assistant",
            content="Nice to meet you, John! Python is a great language. What do you enjoy about it?"
        ),
        Message(role="user", content="I love how readable and expressive it is"),
        Message(
            role="assistant",
            content="Absolutely! Python's clean syntax makes it great for both beginners and experts."
        )
    ]


@pytest.fixture
def sample_conversation_dict() -> list[dict]:
    """Sample conversation as dictionaries."""
    return [
        {"role": "user", "content": "Hi, my name is John and I love Python"},
        {
            "role": "assistant",
            "content": "Nice to meet you, John! Python is a great language. What do you enjoy about it?"
        },
        {"role": "user", "content": "I love how readable and expressive it is"},
        {
            "role": "assistant",
            "content": "Absolutely! Python's clean syntax makes it great for both beginners and experts."
        }
    ]


@pytest.fixture
def mock_extracted_memories() -> dict:
    """Mock extracted memories from conversation."""
    return {
        "traits": ["Detail-oriented", "Thoughtful", "Curious about technology"],
        "preferences": [
            "Prefers readable code",
            "Likes Python programming",
            "Values clean syntax"
        ],
        "facts": [
            "Name is John",
            "Works as a software engineer",
            "Uses Python for development"
        ],
        "summary": "John discussed his passion for Python programming and preferences for readable code."
    }


@pytest.fixture
def env_vars(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("MEM0_ENABLED", "true")
    monkeypatch.setenv("MEM0_API_KEY", "test-key-123")
    monkeypatch.setenv("MEM0_ORG_ID", "test-org-456")
    monkeypatch.setenv("MEM0_PROJECT_ID", "test-project-789")
    monkeypatch.setenv("MEM0_CONTEXT_MAX_TOKENS", "500")
    monkeypatch.setenv("MEM0_CHECKPOINT_MESSAGE_LIMIT", "10")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
