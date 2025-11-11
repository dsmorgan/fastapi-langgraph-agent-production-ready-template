"""Utilities for extracting and categorizing memories from conversations."""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.logging import logger


class MemoryExtractor:
    """Extracts structured memories from conversations using LLM."""

    def __init__(self):
        """Initialize the memory extractor with LLM."""
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.DEFAULT_LLM_MODEL,
            temperature=0.1,  # Low temperature for consistent extraction
        )

    async def extract_memories_from_conversation(
        self,
        messages: list,
        user_id: int,
    ) -> dict:
        """Extract facts, preferences, and traits from a conversation.

        Args:
            messages: List of messages from the conversation
            user_id: User ID for context

        Returns:
            Dictionary with extracted memories by type:
            {
                "traits": [...],
                "preferences": [...],
                "facts": [...],
                "summary": "..."
            }
        """
        try:
            # Format messages for analysis
            conversation_text = self._format_conversation(messages)

            # Extract using structured prompts
            extraction_prompt = ChatPromptTemplate.from_template(
                """Analyze the following conversation and extract key information about the user.

Conversation:
{conversation}

Please identify and list:
1. **User Traits**: Personal characteristics, personality traits, communication style
2. **User Preferences**: Likes, dislikes, preferences mentioned
3. **Key Facts**: Important facts, background info, status updates
4. **Summary**: Brief summary of the conversation

Format your response as:
TRAITS:
- [trait 1]
- [trait 2]
...

PREFERENCES:
- [preference 1]
- [preference 2]
...

FACTS:
- [fact 1]
- [fact 2]
...

SUMMARY:
[One or two sentence summary]
"""
            )

            # Create chain and invoke
            chain = extraction_prompt | self.llm
            response = await chain.ainvoke({"conversation": conversation_text})

            # Parse the response
            extracted = self._parse_extraction_response(response.content)

            logger.info(
                "memories_extracted",
                user_id=user_id,
                traits_count=len(extracted.get("traits", [])),
                preferences_count=len(extracted.get("preferences", [])),
                facts_count=len(extracted.get("facts", [])),
            )

            return extracted
        except Exception as e:
            logger.error(
                "memory_extraction_failed",
                user_id=user_id,
                error=str(e),
            )
            return {"traits": [], "preferences": [], "facts": [], "summary": ""}

    def _format_conversation(self, messages: list) -> str:
        """Format messages into readable conversation text.

        Args:
            messages: List of message dicts

        Returns:
            Formatted conversation string
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.upper()}: {content}")
        return "\n".join(formatted)

    @staticmethod
    def _parse_extraction_response(response_text: str) -> dict:
        """Parse LLM response into structured memory categories.

        Args:
            response_text: Raw LLM response

        Returns:
            Dictionary with extracted memories by type
        """
        result = {
            "traits": [],
            "preferences": [],
            "facts": [],
            "summary": "",
        }

        try:
            sections = response_text.split("\n")
            current_section = None

            for line in sections:
                line = line.strip()

                if not line:
                    continue

                # Detect section headers
                if line.startswith("TRAITS:"):
                    current_section = "traits"
                    continue
                elif line.startswith("PREFERENCES:"):
                    current_section = "preferences"
                    continue
                elif line.startswith("FACTS:"):
                    current_section = "facts"
                    continue
                elif line.startswith("SUMMARY:"):
                    current_section = "summary"
                    continue

                # Extract bullet points for list sections
                if current_section in ["traits", "preferences", "facts"]:
                    if line.startswith("- "):
                        item = line[2:].strip()
                        if item and item not in result[current_section]:
                            result[current_section].append(item)
                elif current_section == "summary":
                    if result["summary"]:
                        result["summary"] += " " + line
                    else:
                        result["summary"] = line

            return result
        except Exception as e:
            logger.error("memory_parsing_failed", error=str(e))
            return result

    async def extract_conversation_summary(
        self,
        messages: list,
        max_length: int = 200,
    ) -> str:
        """Create a concise summary of the conversation.

        Args:
            messages: List of messages from the conversation
            max_length: Maximum length of summary in characters

        Returns:
            Conversation summary
        """
        try:
            conversation_text = self._format_conversation(messages)

            summary_prompt = ChatPromptTemplate.from_template(
                """Provide a brief, one or two sentence summary of this conversation:

{conversation}

Summary:"""
            )

            # Create chain and invoke
            chain = summary_prompt | self.llm
            response = await chain.ainvoke({"conversation": conversation_text})

            summary = response.content.strip()

            # Enforce max length
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."

            return summary
        except Exception as e:
            logger.error("summary_extraction_failed", error=str(e))
            return ""


# Create singleton instance
memory_extractor = MemoryExtractor()
