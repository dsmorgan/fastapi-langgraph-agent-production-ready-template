"""This file contains the graph schema for the application."""

import re
import uuid
from typing import (
    Annotated,
    Optional,
)

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    session_id: str = Field(..., description="The unique identifier for the conversation session")
    user_id: Optional[int] = Field(default=None, description="The user ID for mem0 context injection")
    prompt_label: Optional[str] = Field(
        default=None, description="The Langfuse prompt label to use (production, staging, etc.)"
    )
    prompt_template: Optional[str] = Field(default=None, description="The compiled prompt template for this session")
    prompt_config: Optional[dict] = Field(
        default=None, description="Configuration from the Langfuse prompt (model, temperature, etc.)"
    )
    prompt_version: Optional[int] = Field(default=None, description="The version of the prompt being used")
    prompt_name: Optional[str] = Field(default="system-prompt", description="The Langfuse prompt name")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate that the session ID is a valid UUID or follows safe pattern.

        Args:
            v: The thread ID to validate

        Returns:
            str: The validated session ID

        Raises:
            ValueError: If the session ID is not valid
        """
        # Try to validate as UUID
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            # If not a UUID, check for safe characters only
            if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
                raise ValueError("Session ID must contain only alphanumeric characters, underscores, and hyphens")
            return v
