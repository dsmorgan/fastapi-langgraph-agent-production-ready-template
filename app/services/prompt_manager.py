"""Service for managing prompts using Langfuse."""

from datetime import datetime, timedelta
from typing import Any, Optional

from langfuse import Langfuse

from app.core.config import settings
from app.core.logging import logger


class PromptData:
    """Data class for prompt information from Langfuse."""

    def __init__(self, template: str, config: dict, version: int, labels: list[str], name: str):
        """Initialize prompt data.

        Args:
            template: The prompt template string
            config: Configuration dict (model, temperature, etc.)
            version: Prompt version number
            labels: List of labels (e.g., "production", "staging")
            name: Prompt name in Langfuse
        """
        self.template = template
        self.config = config
        self.version = version
        self.labels = labels
        self.name = name
        self.fetched_at = datetime.now()

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cached prompt has expired.

        Args:
            ttl_seconds: Time-to-live in seconds (default 1 hour)

        Returns:
            True if expired, False otherwise
        """
        elapsed = (datetime.now() - self.fetched_at).total_seconds()
        return elapsed > ttl_seconds


class PromptManager:
    """Service for fetching and managing prompts from Langfuse.

    This service handles:
    - Fetching prompts from Langfuse with label/version support
    - Caching prompts to reduce API calls
    - Compiling templates with variable substitution
    - Tracking prompt metadata for tracing
    """

    def __init__(self):
        """Initialize the Prompt Manager with Langfuse client."""
        self.langfuse: Optional[Langfuse] = None
        self.prompt_cache: dict[str, PromptData] = {}
        self.cache_ttl_seconds = 3600  # 1 hour TTL for session-level caching

        try:
            # Initialize Langfuse client
            if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
                self.langfuse = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                logger.info(
                    "prompt_manager_initialized",
                    langfuse_host=settings.LANGFUSE_HOST,
                )
            else:
                logger.warning("prompt_manager_langfuse_credentials_missing")
        except Exception as e:
            logger.error(
                "prompt_manager_initialization_failed",
                error=str(e),
            )
            self.langfuse = None

    async def get_prompt(
        self,
        name: str = "system-prompt",
        label: Optional[str] = None,
        version: Optional[int] = None,
        use_cache: bool = True,
    ) -> PromptData:
        """Fetch a prompt from Langfuse with caching.

        Args:
            name: The prompt name in Langfuse (default: "system-prompt")
            label: Optional label like "production", "staging" (default: latest)
            version: Optional specific version number (takes precedence over label)
            use_cache: Whether to use cached prompt (default: True for session-level)

        Returns:
            PromptData object with template, config, version, and labels

        Raises:
            ValueError: If prompt cannot be fetched and no fallback available
        """
        if not self.langfuse:
            logger.warning("prompt_manager_langfuse_not_available_using_fallback")
            return await self._get_fallback_prompt()

        # Create cache key from prompt name, label, and version
        cache_key = f"{name}:{label or 'latest'}:{version or 'none'}"

        # Check cache first (if enabled and not expired)
        if use_cache and cache_key in self.prompt_cache:
            cached_prompt = self.prompt_cache[cache_key]
            if not cached_prompt.is_expired(self.cache_ttl_seconds):
                logger.info(
                    "prompt_loaded_from_cache",
                    prompt_name=name,
                    label=label,
                    version=version,
                    cache_age_seconds=(datetime.now() - cached_prompt.fetched_at).total_seconds(),
                )
                return cached_prompt

        # Fetch from Langfuse
        try:
            prompt_obj = self.langfuse.get_prompt(
                name=name,
                label=label,
                version=version,
            )

            prompt_data = PromptData(
                template=prompt_obj.prompt,
                config=prompt_obj.config or {},
                version=prompt_obj.version,
                labels=prompt_obj.labels or [],
                name=name,
            )

            # Cache the prompt
            self.prompt_cache[cache_key] = prompt_data

            logger.info(
                "prompt_fetched_from_langfuse",
                prompt_name=name,
                label=label,
                version=version,
                prompt_version=prompt_obj.version,
                prompt_labels=prompt_obj.labels,
            )

            return prompt_data

        except Exception as e:
            logger.error(
                "prompt_fetch_from_langfuse_failed",
                prompt_name=name,
                label=label,
                version=version,
                error=str(e),
            )
            # Fall back to default prompt
            return await self._get_fallback_prompt()

    async def _get_fallback_prompt(self) -> PromptData:
        """Get fallback prompt when Langfuse is unavailable.

        Returns:
            PromptData with default system prompt
        """
        from app.core.prompts import load_system_prompt

        template = load_system_prompt()
        return PromptData(
            template=template,
            config={"model": settings.DEFAULT_LLM_MODEL, "temperature": settings.DEFAULT_LLM_TEMPERATURE},
            version=0,
            labels=["fallback"],
            name="system-prompt-fallback",
        )

    def compile_prompt(
        self,
        prompt_data: PromptData,
        **variables: Any,
    ) -> str:
        """Compile a prompt template with variable substitution.

        Args:
            prompt_data: The PromptData object containing template
            **variables: Variables to substitute in the template

        Returns:
            Compiled prompt string with variables substituted

        Example:
            compiled = prompt_manager.compile_prompt(
                prompt_data,
                agent_name="Cicero",
                current_date="2025-11-10"
            )
        """
        template = prompt_data.template

        try:
            # Use Python format string substitution
            # Convert Langfuse {{var}} syntax to {var} if needed
            if "{{" in template:
                # Replace {{var}} with {var}
                template = template.replace("{{", "{").replace("}}", "}")

            compiled = template.format(**variables)

            logger.info(
                "prompt_compiled",
                prompt_name=prompt_data.name,
                variables_count=len(variables),
            )

            return compiled
        except KeyError as e:
            logger.error(
                "prompt_compilation_missing_variable",
                prompt_name=prompt_data.name,
                missing_variable=str(e),
            )
            # Return uncompiled template if variables are missing
            return template
        except Exception as e:
            logger.error(
                "prompt_compilation_failed",
                prompt_name=prompt_data.name,
                error=str(e),
            )
            return template

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cached prompts.

        Args:
            name: If provided, only clear cache entries for this prompt name.
                  If None, clear entire cache.
        """
        if name:
            # Clear specific prompt entries
            keys_to_remove = [key for key in self.prompt_cache.keys() if key.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self.prompt_cache[key]
            logger.info("prompt_cache_cleared_for_name", prompt_name=name, removed_count=len(keys_to_remove))
        else:
            # Clear entire cache
            self.prompt_cache.clear()
            logger.info("prompt_cache_cleared_all")


# Create singleton instance
prompt_manager = PromptManager()
