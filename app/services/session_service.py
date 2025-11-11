"""Service for managing chat sessions with mem0 integration."""

from typing import Optional

from app.core.config import settings
from app.core.logging import logger
from app.services.database import database_service
from app.services.mem0_service import mem0_service
from app.utils.memory_extraction import memory_extractor
from app.schemas.chat import Message


class SessionMemoryManager:
    """Manages session lifecycle with mem0 integration.

    Responsibilities:
    - Initialize sessions with user context from mem0
    - Extract and store memories at session end
    - Manage checkpoint message limits
    """

    async def initialize_session(
        self,
        session_id: str,
        user_id: int,
    ) -> list[Message]:
        """Initialize a session with user context from mem0.

        When a new session starts, this retrieves relevant memories from mem0
        to provide context to the agent.

        Args:
            session_id: The session ID
            user_id: The user ID

        Returns:
            List of initial context messages to prepend to the conversation
        """
        try:
            logger.info(
                "session_initialized",
                session_id=session_id,
                user_id=user_id,
            )

            # Get user context from mem0
            user_context = await mem0_service.get_user_context(user_id)

            if not user_context:
                logger.info(
                    "no_user_context_found",
                    session_id=session_id,
                    user_id=user_id,
                )
                return []

            # Create system message with user context
            context_message = Message(
                role="system",
                content=(
                    "The following is relevant context about the user from previous conversations:\n\n"
                    + user_context
                    + "\n\nUse this context to provide personalized responses."
                ),
            )

            logger.info(
                "user_context_injected",
                session_id=session_id,
                user_id=user_id,
                context_length=len(user_context),
            )

            return [context_message]
        except Exception as e:
            logger.error(
                "session_initialization_failed",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )
            return []

    async def finalize_session(
        self,
        session_id: str,
        user_id: int,
        messages: list[Message],
    ) -> bool:
        """Finalize a session by extracting and storing memories.

        When a session ends, this extracts new memories from the conversation
        and stores them in mem0 for future use.

        Args:
            session_id: The session ID
            user_id: The user ID
            messages: All messages from the session

        Returns:
            True if successful, False otherwise
        """
        try:
            if not messages:
                logger.info(
                    "session_finalized_no_messages",
                    session_id=session_id,
                    user_id=user_id,
                )
                return True

            logger.info(
                "session_finalizing",
                session_id=session_id,
                user_id=user_id,
                message_count=len(messages),
            )

            # Extract memories from the conversation
            extracted = await memory_extractor.extract_memories_from_conversation(
                [{"role": m.role, "content": m.content} for m in messages],
                user_id,
            )

            # Store extracted memories
            memory_count = 0

            # Store traits
            for trait in extracted.get("traits", []):
                memory_id = await mem0_service.add_memory(
                    user_id=user_id,
                    data=trait,
                    memory_type="trait",
                )
                if memory_id:
                    memory_count += 1

            # Store preferences
            for preference in extracted.get("preferences", []):
                memory_id = await mem0_service.add_memory(
                    user_id=user_id,
                    data=preference,
                    memory_type="preference",
                )
                if memory_id:
                    memory_count += 1

            # Store facts
            for fact in extracted.get("facts", []):
                memory_id = await mem0_service.add_memory(
                    user_id=user_id,
                    data=fact,
                    memory_type="fact",
                )
                if memory_id:
                    memory_count += 1

            # Store conversation summary
            summary = extracted.get("summary", "")
            if summary:
                summary_memory_id = await mem0_service.add_memory(
                    user_id=user_id,
                    data=summary,
                    memory_type="summary",
                )
                if summary_memory_id:
                    memory_count += 1

            logger.info(
                "session_finalized",
                session_id=session_id,
                user_id=user_id,
                memories_stored=memory_count,
            )

            return True
        except Exception as e:
            logger.error(
                "session_finalization_failed",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )
            return False

    async def get_search_result_context(
        self,
        session_id: str,
        user_id: int,
        query: str,
    ) -> str:
        """Search mem0 for relevant context based on a query.

        This is used during chat to find relevant memories that should be
        injected into the LLM context.

        Args:
            session_id: The session ID
            user_id: The user ID
            query: The search query (usually the current user message)

        Returns:
            Formatted context string from matching memories
        """
        try:
            # Search for relevant memories
            results = await mem0_service.search_memories(
                user_id=user_id,
                query=query,
                limit=3,
            )

            if not results:
                return ""

            # Format results into context
            context_parts = ["Relevant user memories:"]
            for result in results:
                # Handle different response formats
                if isinstance(result, dict):
                    content = result.get("data") or result.get("content") or str(result)
                else:
                    content = str(result)

                context_parts.append(f"- {content}")

            context = "\n".join(context_parts)

            logger.info(
                "search_context_retrieved",
                session_id=session_id,
                user_id=user_id,
                results_count=len(results),
            )

            return context
        except Exception as e:
            logger.error(
                "search_context_failed",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )
            return ""


# Create singleton instance
session_memory_manager = SessionMemoryManager()
