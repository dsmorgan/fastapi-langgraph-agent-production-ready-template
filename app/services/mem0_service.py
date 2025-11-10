"""Service for managing long-term memory using Mem0 SaaS."""

from typing import Optional
import os
from mem0 import MemoryClient

from app.core.config import settings
from app.core.logging import logger


class Mem0Service:
    """Service for managing user memories using Mem0 SaaS platform.

    This service handles:
    - Adding memories (facts, preferences, traits)
    - Searching for relevant memories
    - Updating existing memories
    - Retrieving user context for LLM interactions
    """

    def __init__(self):
        """Initialize Mem0 client with API credentials."""
        self.enabled = settings.MEM0_ENABLED
        self.client: Optional[MemoryClient] = None

        if self.enabled:
            try:
                # Initialize Mem0 client
                self.client = MemoryClient(
                    api_key=settings.MEM0_API_KEY,
                    org_id=settings.MEM0_ORG_ID,
                    project_id=settings.MEM0_PROJECT_ID,
                )
                logger.info(
                    "mem0_service_initialized",
                    org_id=settings.MEM0_ORG_ID,
                    project_id=settings.MEM0_PROJECT_ID,
                )
            except Exception as e:
                logger.error(
                    "mem0_initialization_failed",
                    error=str(e),
                    enabled=self.enabled,
                )
                self.enabled = False

    async def add_memory(
        self,
        user_id: int,
        data: str,
        memory_type: str = "fact",
    ) -> Optional[str]:
        """Add a memory to mem0.

        Args:
            user_id: The user ID for memory association
            data: The memory text/content to store
            memory_type: Type of memory (fact, preference, trait, summary)

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.enabled or not self.client:
            return None

        try:
            # Call Mem0 API to add memory
            # The API expects: messages parameter with list of dicts containing role and content
            response = self.client.add(
                messages=[{"role": "user", "content": data}],
                user_id=str(user_id),
            )

            logger.info(
                "memory_added_to_mem0",
                user_id=user_id,
                memory_type=memory_type,
                memory_id=response.get("id") if isinstance(response, dict) else None,
            )

            return response.get("id") if isinstance(response, dict) else str(response)
        except Exception as e:
            logger.error(
                "mem0_add_memory_failed",
                user_id=user_id,
                memory_type=memory_type,
                error=str(e),
            )
            return None

    async def search_memories(
        self,
        user_id: int,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search for relevant memories using semantic search.

        Args:
            user_id: The user ID to search memories for
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of relevant memory objects
        """
        if not self.enabled or not self.client:
            return []

        try:
            # Search using Mem0 with filters required by API
            # Filters are required by the Mem0 API
            filters = [{"key": "user_id", "value": str(user_id)}]

            results = self.client.search(
                query=query,
                filters=filters,
                limit=limit,
            )

            logger.info(
                "memories_searched",
                user_id=user_id,
                query=query,
                results_count=len(results) if results else 0,
            )

            return results if results else []
        except Exception as e:
            logger.error(
                "mem0_search_failed",
                user_id=user_id,
                query=query,
                error=str(e),
            )
            return []

    async def update_memory(
        self,
        memory_id: str,
        data: str,
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: The ID of the memory to update
            data: New data for the memory

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            self.client.update(memory_id=memory_id, data=data)
            logger.info("memory_updated", memory_id=memory_id)
            return True
        except Exception as e:
            logger.error(
                "mem0_update_failed",
                memory_id=memory_id,
                error=str(e),
            )
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            self.client.delete(memory_id=memory_id)
            logger.info("memory_deleted", memory_id=memory_id)
            return True
        except Exception as e:
            logger.error(
                "mem0_delete_failed",
                memory_id=memory_id,
                error=str(e),
            )
            return False

    async def get_user_context(
        self,
        user_id: int,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get formatted user context from all memories.

        Retrieves all user memories and formats them into a context string
        suitable for injection into LLM prompts.

        Args:
            user_id: The user ID to retrieve context for
            max_tokens: Maximum tokens to allocate for context

        Returns:
            Formatted context string ready for LLM injection
        """
        if not self.enabled or not self.client:
            return ""

        try:
            max_tokens = max_tokens or settings.MEM0_CONTEXT_MAX_TOKENS

            # Get all memories for the user
            # Note: get_all() expects filters as a dictionary, unlike search() which uses a list
            filters = {"user_id": str(user_id)}
            response = self.client.get_all(user_id=str(user_id), filters=filters)

            # Extract memories from the response - it can be a dict with 'results' key or a list
            if isinstance(response, dict) and "results" in response:
                memories = response.get("results", [])
            else:
                memories = response if response else []

            logger.info(
                "mem0_memories_retrieved",
                user_id=user_id,
                total_memory_count=len(memories) if memories else 0,
                raw_memories=memories,
            )

            if not memories:
                logger.info(
                    "mem0_no_memories_found",
                    user_id=user_id,
                )
                return ""

            # Format memories into context
            context_parts = []

            # Group by categories for better organization
            # Mem0 API returns memories with 'categories' field instead of metadata.type
            # Handle cases where categories might be None instead of a list
            traits = [
                m for m in memories
                if isinstance(m, dict) and (m.get("categories") or []) and "personal_details" in (m.get("categories") or [])
            ]
            preferences = [
                m for m in memories
                if isinstance(m, dict) and (m.get("categories") or []) and "user_preferences" in (m.get("categories") or [])
            ]
            facts = [
                m for m in memories
                if isinstance(m, dict) and (m.get("categories") or []) and "facts" in (m.get("categories") or [])
            ]

            # Get all memories that don't match our expected types (in case mem0 stores differently)
            categorized_ids = set()
            for m in traits + preferences + facts:
                if isinstance(m, dict) and "id" in m:
                    categorized_ids.add(m["id"])

            uncategorized = [m for m in memories if not (isinstance(m, dict) and m.get("id") in categorized_ids)]

            logger.info(
                "mem0_memories_categorized",
                user_id=user_id,
                traits_count=len(traits),
                preferences_count=len(preferences),
                facts_count=len(facts),
                uncategorized_count=len(uncategorized),
            )

            if traits:
                context_parts.append("## User Traits")
                for trait in traits[:3]:  # Limit to 3 traits
                    memory_text = trait.get('memory') or trait.get('data') or trait.get('content') or str(trait)
                    context_parts.append(f"- {memory_text}")

            if preferences:
                context_parts.append("\n## User Preferences")
                for pref in preferences[:3]:
                    memory_text = pref.get('memory') or pref.get('data') or pref.get('content') or str(pref)
                    context_parts.append(f"- {memory_text}")

            if facts:
                context_parts.append("\n## Known Facts")
                for fact in facts[:5]:  # Limit to 5 facts
                    memory_text = fact.get('memory') or fact.get('data') or fact.get('content') or str(fact)
                    context_parts.append(f"- {memory_text}")

            # Include uncategorized memories (they might be important)
            if uncategorized:
                context_parts.append("\n## Other Information")
                for item in uncategorized[:5]:  # Limit to 5 uncategorized
                    if isinstance(item, dict):
                        content = item.get("memory") or item.get("data") or item.get("content") or str(item)
                    else:
                        content = str(item)
                    context_parts.append(f"- {content}")

            context = "\n".join(context_parts)

            # Rough token limit (approximate: 1 token ≈ 4 chars)
            if len(context) > max_tokens * 4:
                context = context[: max_tokens * 4]

            logger.info(
                "user_context_formatted",
                user_id=user_id,
                memory_count=len(memories),
                context_length=len(context),
                context_preview=context[:200] if context else "empty",
            )

            return context
        except Exception as e:
            logger.error(
                "mem0_get_context_failed",
                user_id=user_id,
                error=str(e),
            )
            return ""

    @staticmethod
    def _format_messages_for_memory(messages: list) -> str:
        """Format messages into readable text for memory storage.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Formatted message string
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


# Create singleton instance
mem0_service = Mem0Service()
