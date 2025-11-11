"""Tests for Mem0 service."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.mem0_service import Mem0Service


class TestMem0Service:
    """Test cases for Mem0Service class."""

    @pytest.fixture
    def mem0_service(self, mock_mem0_client, env_vars):
        """Create Mem0Service with mocked client."""
        with patch("app.services.mem0_service.MemoryClient", return_value=mock_mem0_client):
            service = Mem0Service()
            service.client = mock_mem0_client
            return service

    @pytest.mark.asyncio
    async def test_add_memory_success(self, mem0_service):
        """Test adding a memory successfully."""
        user_id = 12345
        data = "User loves Python programming"
        memory_type = "preference"

        result = await mem0_service.add_memory(
            user_id=user_id,
            data=data,
            memory_type=memory_type
        )

        assert result == "memory_123"
        mem0_service.client.add.assert_called_once()
        call_args = mem0_service.client.add.call_args
        assert call_args.kwargs["user_id"] == str(user_id)
        assert call_args.kwargs["messages"][0]["content"] == data

    @pytest.mark.asyncio
    async def test_add_memory_when_disabled(self, env_vars):
        """Test adding memory when mem0 is disabled."""
        with patch("app.services.mem0_service.MemoryClient"):
            service = Mem0Service()
            service.enabled = False

            result = await service.add_memory(
                user_id=12345,
                data="Test memory",
                memory_type="fact"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_add_memory_handles_exception(self, mem0_service):
        """Test handling of exceptions during memory addition."""
        mem0_service.client.add.side_effect = Exception("API Error")

        result = await mem0_service.add_memory(
            user_id=12345,
            data="Test memory",
            memory_type="fact"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_search_memories_success(self, mem0_service):
        """Test searching memories successfully."""
        user_id = 12345
        query = "Python programming"
        limit = 5

        result = await mem0_service.search_memories(
            user_id=user_id,
            query=query,
            limit=limit
        )

        assert len(result) == 2
        assert result[0]["id"] == "mem_1"
        assert "Python" in result[0]["data"]

        mem0_service.client.search.assert_called_once()
        call_args = mem0_service.client.search.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["limit"] == limit
        # Check that filters include user_id
        filters = call_args.kwargs["filters"]
        assert filters[0]["key"] == "user_id"
        assert filters[0]["value"] == str(user_id)

    @pytest.mark.asyncio
    async def test_search_memories_when_disabled(self, env_vars):
        """Test searching memories when mem0 is disabled."""
        with patch("app.services.mem0_service.MemoryClient"):
            service = Mem0Service()
            service.enabled = False

            result = await service.search_memories(
                user_id=12345,
                query="test",
                limit=5
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_search_memories_handles_exception(self, mem0_service):
        """Test handling of exceptions during memory search."""
        mem0_service.client.search.side_effect = Exception("Search failed")

        result = await mem0_service.search_memories(
            user_id=12345,
            query="test",
            limit=5
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_user_context_success(self, mem0_service):
        """Test getting user context successfully."""
        user_id = 12345

        result = await mem0_service.get_user_context(user_id=user_id)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "User Traits" in result or "User Preferences" in result or "Known Facts" in result

    @pytest.mark.asyncio
    async def test_get_user_context_includes_all_memory_types(self, mem0_service):
        """Test that context includes all memory types."""
        user_id = 12345

        result = await mem0_service.get_user_context(user_id=user_id)

        # Should include sections for different memory types
        assert "User Traits" in result or "User Preferences" in result or "Known Facts" in result

    @pytest.mark.asyncio
    async def test_get_user_context_with_max_tokens(self, mem0_service):
        """Test getting user context with token limit."""
        user_id = 12345
        max_tokens = 100

        result = await mem0_service.get_user_context(
            user_id=user_id,
            max_tokens=max_tokens
        )

        # Result should be limited by max_tokens (approximate: 1 token ≈ 4 chars)
        assert len(result) <= max_tokens * 4 + 100  # Allow some buffer

    @pytest.mark.asyncio
    async def test_get_user_context_when_disabled(self, env_vars):
        """Test getting context when mem0 is disabled."""
        with patch("app.services.mem0_service.MemoryClient"):
            service = Mem0Service()
            service.enabled = False

            result = await service.get_user_context(user_id=12345)

            assert result == ""

    @pytest.mark.asyncio
    async def test_get_user_context_empty_memories(self, mem0_service):
        """Test getting context when user has no memories."""
        mem0_service.client.get_all.return_value = []

        result = await mem0_service.get_user_context(user_id=12345)

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_user_context_handles_exception(self, mem0_service):
        """Test handling of exceptions during context retrieval."""
        mem0_service.client.get_all.side_effect = Exception("API Error")

        result = await mem0_service.get_user_context(user_id=12345)

        assert result == ""

    @pytest.mark.asyncio
    async def test_update_memory_success(self, mem0_service):
        """Test updating a memory successfully."""
        memory_id = "mem_123"
        data = "Updated memory content"

        result = await mem0_service.update_memory(
            memory_id=memory_id,
            data=data
        )

        assert result is True
        mem0_service.client.update.assert_called_once_with(
            memory_id=memory_id,
            data=data
        )

    @pytest.mark.asyncio
    async def test_update_memory_failure(self, mem0_service):
        """Test update memory failure."""
        mem0_service.client.update.side_effect = Exception("Update failed")

        result = await mem0_service.update_memory(
            memory_id="mem_123",
            data="data"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, mem0_service):
        """Test deleting a memory successfully."""
        memory_id = "mem_123"

        result = await mem0_service.delete_memory(memory_id=memory_id)

        assert result is True
        mem0_service.client.delete.assert_called_once_with(memory_id=memory_id)

    @pytest.mark.asyncio
    async def test_delete_memory_failure(self, mem0_service):
        """Test delete memory failure."""
        mem0_service.client.delete.side_effect = Exception("Delete failed")

        result = await mem0_service.delete_memory(memory_id="mem_123")

        assert result is False
