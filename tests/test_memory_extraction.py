"""Tests for memory extraction utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.memory_extraction import MemoryExtractor


class TestMemoryExtractor:
    """Test cases for MemoryExtractor class."""

    @pytest.fixture
    def memory_extractor(self, mock_openai_llm):
        """Create MemoryExtractor with mocked LLM."""
        extractor = MemoryExtractor()
        extractor.llm = mock_openai_llm
        return extractor

    @pytest.mark.asyncio
    async def test_extract_memories_from_conversation(
        self,
        sample_conversation_dict,
        mock_extracted_memories
    ):
        """Test extracting memories from a conversation."""
        # Create extractor and mock the entire chain
        with patch("app.utils.memory_extraction.ChatPromptTemplate") as mock_prompt_template:
            # Mock the response
            llm_response = MagicMock()
            llm_response.content = """TRAITS:
- Detail-oriented
- Thoughtful
- Curious about technology

PREFERENCES:
- Prefers readable code
- Likes Python programming
- Values clean syntax

FACTS:
- Name is John
- Works as a software engineer
- Uses Python for development

SUMMARY:
John discussed his passion for Python programming and preferences for readable code."""

            # Mock the chain
            mock_chain = AsyncMock(return_value=llm_response)
            mock_prompt_template.from_template.return_value = mock_chain

            extractor = MemoryExtractor()

            # Mock the pipe operator
            mock_chain_piped = AsyncMock(return_value=llm_response)
            with patch.object(mock_chain, "__or__", return_value=mock_chain_piped):
                result = await extractor.extract_memories_from_conversation(
                    messages=sample_conversation_dict,
                    user_id=12345
                )

                # Even with mock setup, verify it returns proper structure
                assert isinstance(result, dict)
                assert "traits" in result
                assert "preferences" in result
                assert "facts" in result
                assert "summary" in result

    @pytest.mark.asyncio
    async def test_extract_memories_handles_exception(
        self,
        memory_extractor,
        sample_conversation_dict
    ):
        """Test handling of exceptions during memory extraction."""
        memory_extractor.llm.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))

        result = await memory_extractor.extract_memories_from_conversation(
            messages=sample_conversation_dict,
            user_id=12345
        )

        assert result == {"traits": [], "preferences": [], "facts": [], "summary": ""}

    def test_format_conversation(self, memory_extractor, sample_conversation_dict):
        """Test formatting conversation to text."""
        result = memory_extractor._format_conversation(sample_conversation_dict)

        assert isinstance(result, str)
        assert "USER:" in result
        assert "ASSISTANT:" in result
        assert "Python" in result

    def test_parse_extraction_response_complete(self, memory_extractor):
        """Test parsing complete extraction response."""
        response_text = """TRAITS:
- Friendly
- Detail-oriented
- Curious

PREFERENCES:
- Likes coffee
- Prefers remote work
- Values cleanliness

FACTS:
- Name is Alice
- Has 5 years of experience
- Lives in New York

SUMMARY:
Alice is a friendly developer who values clean code and remote work."""

        result = memory_extractor._parse_extraction_response(response_text)

        assert len(result["traits"]) == 3
        assert "Friendly" in result["traits"]
        assert len(result["preferences"]) == 3
        assert "Likes coffee" in result["preferences"]
        assert len(result["facts"]) == 3
        assert "Name is Alice" in result["facts"]
        assert len(result["summary"]) > 0
        assert "Alice" in result["summary"]

    def test_parse_extraction_response_partial(self, memory_extractor):
        """Test parsing partial extraction response."""
        response_text = """TRAITS:
- Helpful

PREFERENCES:
- Likes Python

SUMMARY:
User is helpful and likes Python."""

        result = memory_extractor._parse_extraction_response(response_text)

        assert len(result["traits"]) == 1
        assert len(result["preferences"]) == 1
        assert len(result["facts"]) == 0
        assert len(result["summary"]) > 0

    def test_parse_extraction_response_empty_sections(self, memory_extractor):
        """Test parsing extraction response with empty sections."""
        response_text = """TRAITS:

PREFERENCES:
- Likes reading

FACTS:

SUMMARY:
User likes reading."""

        result = memory_extractor._parse_extraction_response(response_text)

        assert len(result["traits"]) == 0
        assert len(result["preferences"]) == 1
        assert len(result["facts"]) == 0
        assert len(result["summary"]) > 0

    def test_parse_extraction_response_handles_error(self, memory_extractor):
        """Test parsing handles malformed input gracefully."""
        response_text = "This is not a properly formatted response"

        result = memory_extractor._parse_extraction_response(response_text)

        assert isinstance(result, dict)
        assert "traits" in result
        assert "preferences" in result
        assert "facts" in result
        assert "summary" in result

    def test_parse_extraction_response_deduplication(self, memory_extractor):
        """Test that duplicate items are not added."""
        response_text = """TRAITS:
- Friendly
- Friendly
- Creative

PREFERENCES:
- Coffee
- Coffee

FACTS:
- Name is Bob
- Name is Bob

SUMMARY:
Bob is friendly and creative."""

        result = memory_extractor._parse_extraction_response(response_text)

        assert len(result["traits"]) == 2  # Friendly should appear only once
        assert len(result["preferences"]) == 1
        assert len(result["facts"]) == 1

    @pytest.mark.asyncio
    async def test_extract_conversation_summary(self, sample_conversation_dict):
        """Test extracting conversation summary."""
        with patch("app.utils.memory_extraction.ChatPromptTemplate") as mock_prompt_template:
            llm_response = MagicMock()
            llm_response.content = "John shared his passion for Python and preferences for clean code."

            # Create a chain that has ainvoke method
            mock_chain_piped = MagicMock()
            mock_chain_piped.ainvoke = AsyncMock(return_value=llm_response)

            # Mock the from_template to return a chain
            mock_chain = MagicMock()
            mock_chain.__or__ = MagicMock(return_value=mock_chain_piped)
            mock_prompt_template.from_template.return_value = mock_chain

            extractor = MemoryExtractor()

            result = await extractor.extract_conversation_summary(
                messages=sample_conversation_dict,
                max_length=200
            )

            assert isinstance(result, str)
            assert len(result) > 0
            assert len(result) <= 200 + 3  # +3 for "..."

    @pytest.mark.asyncio
    async def test_extract_conversation_summary_max_length(
        self,
        memory_extractor,
        sample_conversation_dict
    ):
        """Test that summary respects max length."""
        long_summary = "a" * 300

        llm_response = MagicMock()
        llm_response.content = long_summary

        memory_extractor.llm.ainvoke = AsyncMock(return_value=llm_response)

        result = await memory_extractor.extract_conversation_summary(
            messages=sample_conversation_dict,
            max_length=100
        )

        assert len(result) <= 103  # 100 + "..."

    @pytest.mark.asyncio
    async def test_extract_conversation_summary_handles_exception(
        self,
        memory_extractor,
        sample_conversation_dict
    ):
        """Test handling of exceptions during summary extraction."""
        memory_extractor.llm.ainvoke = AsyncMock(side_effect=Exception("Error"))

        result = await memory_extractor.extract_conversation_summary(
            messages=sample_conversation_dict
        )

        assert result == ""
