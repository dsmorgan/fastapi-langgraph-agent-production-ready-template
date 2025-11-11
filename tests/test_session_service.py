"""Tests for session service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.session_service import SessionMemoryManager
from app.schemas.chat import Message


class TestSessionMemoryManager:
    """Test cases for SessionMemoryManager class."""

    @pytest.fixture
    def session_manager(self, mock_mem0_client):
        """Create SessionMemoryManager with mocked dependencies."""
        manager = SessionMemoryManager()
        return manager

    @pytest.mark.asyncio
    async def test_initialize_session_with_context(self, session_manager):
        """Test initializing session with user context."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            mock_mem0.get_user_context = AsyncMock(
                return_value="User Traits:\n- Friendly\n\nUser Preferences:\n- Likes Python"
            )

            result = await session_manager.initialize_session(
                session_id="session_123",
                user_id=12345
            )

            assert len(result) > 0
            assert isinstance(result[0], Message)
            assert result[0].role == "system"
            assert "Friendly" in result[0].content

    @pytest.mark.asyncio
    async def test_initialize_session_no_context(self, session_manager):
        """Test initializing session when user has no context."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            mock_mem0.get_user_context = AsyncMock(return_value="")

            result = await session_manager.initialize_session(
                session_id="session_123",
                user_id=12345
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_initialize_session_handles_exception(self, session_manager):
        """Test handling of exceptions during session initialization."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            mock_mem0.get_user_context = AsyncMock(side_effect=Exception("Error"))

            result = await session_manager.initialize_session(
                session_id="session_123",
                user_id=12345
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_finalize_session_no_messages(self, session_manager):
        """Test finalizing session with no messages."""
        result = await session_manager.finalize_session(
            session_id="session_123",
            user_id=12345,
            messages=[]
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_finalize_session_with_messages(self, session_manager, sample_messages):
        """Test finalizing session with messages."""
        with patch("app.services.session_service.memory_extractor") as mock_extractor, \
             patch("app.services.session_service.mem0_service") as mock_mem0:

            # Mock memory extraction
            mock_extractor.extract_memories_from_conversation = AsyncMock(
                return_value={
                    "traits": ["Friendly"],
                    "preferences": ["Likes Python"],
                    "facts": ["Name is John"],
                    "summary": "John loves Python"
                }
            )

            # Mock memory storage
            mock_mem0.add_memory = AsyncMock(return_value="mem_123")

            result = await session_manager.finalize_session(
                session_id="session_123",
                user_id=12345,
                messages=sample_messages
            )

            assert result is True
            # Verify that add_memory was called for traits, preferences, facts, and summary
            assert mock_mem0.add_memory.call_count == 4

    @pytest.mark.asyncio
    async def test_finalize_session_handles_exception(self, session_manager, sample_messages):
        """Test handling of exceptions during session finalization."""
        with patch("app.services.session_service.memory_extractor") as mock_extractor:
            mock_extractor.extract_memories_from_conversation = AsyncMock(
                side_effect=Exception("Extraction error")
            )

            result = await session_manager.finalize_session(
                session_id="session_123",
                user_id=12345,
                messages=sample_messages
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_get_search_result_context_with_results(self, session_manager):
        """Test getting search result context with matching memories."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            mock_mem0.search_memories = AsyncMock(
                return_value=[
                    {"id": "mem_1", "data": "User likes Python"},
                    {"id": "mem_2", "data": "User is a developer"}
                ]
            )

            result = await session_manager.get_search_result_context(
                session_id="session_123",
                user_id=12345,
                query="What do you like?"
            )

            assert isinstance(result, str)
            assert "Relevant user memories:" in result
            assert "Python" in result
            assert "developer" in result

    @pytest.mark.asyncio
    async def test_get_search_result_context_no_results(self, session_manager):
        """Test getting search result context with no matching memories."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            mock_mem0.search_memories = AsyncMock(return_value=[])

            result = await session_manager.get_search_result_context(
                session_id="session_123",
                user_id=12345,
                query="What do you like?"
            )

            assert result == ""

    @pytest.mark.asyncio
    async def test_get_search_result_context_handles_exception(self, session_manager):
        """Test handling of exceptions during search context retrieval."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            mock_mem0.search_memories = AsyncMock(side_effect=Exception("Search error"))

            result = await session_manager.get_search_result_context(
                session_id="session_123",
                user_id=12345,
                query="test"
            )

            assert result == ""

    @pytest.mark.asyncio
    async def test_get_search_result_context_various_formats(self, session_manager):
        """Test handling of various memory data formats."""
        with patch("app.services.session_service.mem0_service") as mock_mem0:
            # Mix of different response formats
            mock_mem0.search_memories = AsyncMock(
                return_value=[
                    {"id": "mem_1", "data": "User likes Python"},
                    {"id": "mem_2", "content": "User is a developer"},
                    {"id": "mem_3"}  # No data or content field
                ]
            )

            result = await session_manager.get_search_result_context(
                session_id="session_123",
                user_id=12345,
                query="test"
            )

            assert isinstance(result, str)
            assert "Python" in result
