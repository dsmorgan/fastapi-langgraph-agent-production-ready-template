# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A production-ready FastAPI template for building AI agent applications with LangGraph integration. The application includes:

- **FastAPI**: High-performance async API endpoints
- **LangGraph**: AI agent workflows with PostgreSQL checkpointing
- **Langfuse**: LLM observability and tracing
- **PostgreSQL**: Persistent data storage with connection pooling
- **Prometheus + Grafana**: Metrics and monitoring dashboards
- **Rate Limiting**: SlowAPI-based request throttling
- **Structured Logging**: Environment-specific log formatting
- **JWT Auth**: Token-based authentication with session management

## Architecture Overview

### Core Components

1. **LangGraph Agent** (`app/core/langgraph/graph.py`):
   - The main agent workflow that orchestrates chat state and LLM interactions
   - Manages PostgreSQL connection pools for stateful conversations
   - Processes tool calls from the LLM and routes to appropriate tools
   - Provides both streaming and non-streaming response modes
   - Stores conversation state in PostgreSQL checkpoint tables

2. **FastAPI Application** (`app/main.py`):
   - Sets up middleware (logging context, metrics, rate limiting, CORS)
   - Configures exception handlers for validation and rate limit errors
   - Includes health check and root endpoints
   - Integrates Langfuse for LLM tracing

3. **API Routes** (`app/api/v1/`):
   - `/auth`: Authentication endpoints (register, login)
   - `/chatbot`: Chat endpoints (non-streaming and streaming)
   - Each endpoint applies rate limiting based on configuration

4. **Database Layer** (`app/services/database.py`):
   - SQLModel ORM wrapper with connection pooling
   - User management (create, fetch, delete by email)
   - Chat session management (create, delete, fetch, list by user)
   - Health check endpoint

5. **Configuration** (`app/core/config.py`):
   - Environment-based settings (development, staging, production, test)
   - Loads from `.env.[ENVIRONMENT]` files with fallback priority
   - All settings accessible via the global `settings` object
   - Environment-specific defaults (logging, rate limits, pool sizes)

### State Management

The agent uses LangGraph's `StateGraph` with `GraphState` containing:
- `messages`: List of conversation messages (user, assistant, tool)
- `session_id`: Identifier for conversation thread

The graph has two nodes:
- `chat`: Calls LLM, routes to `tool_call` if tool calls present, else END
- `tool_call`: Executes tools, routes back to `chat`

### Database Schema

PostgreSQL tables created automatically via SQLModel:
- `user`: User accounts with email and hashed passwords
- `chatsession`: Chat sessions linked to users
- `checkpoint_blobs`, `checkpoint_writes`, `checkpoints`: LangGraph state persistence

## Common Development Tasks

### Setup

```bash
# Install dependencies
make install

# Create .env file for your environment
cp .env.example .env.development
# Update with your API keys and database credentials
```

### Development Server

```bash
# Start with hot reload
make dev

# Swagger API docs: http://localhost:8000/docs
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format
```

### Testing

```bash
# Run tests (test framework: pytest)
pytest

# Run specific test
pytest path/to/test_file.py::test_function

# Run with coverage
pytest --cov=app
```

### Running Evaluations

The project includes a model evaluation framework that fetches traces from Langfuse and applies evaluation metrics:

```bash
# Interactive mode with prompts
make eval ENV=development

# Quick mode with defaults
make eval-quick ENV=development

# No report generation
make eval-no-report ENV=development
```

Evaluation metrics are defined in `evals/metrics/prompts/` as markdown files. New metrics are automatically discovered.

### Docker

```bash
# Build and run entire stack (API, DB, Prometheus, Grafana)
make docker-compose-up ENV=development

# View logs
make docker-compose-logs ENV=development

# Stop everything
make docker-compose-down ENV=development

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## Key Configuration Files

- `.env.[development|staging|production]`: Environment-specific settings
- `pyproject.toml`: Dependencies (uv), pytest config, ruff rules
- `Makefile`: Common commands (dev, test, eval, docker)
- `docker-compose.yml`: Full stack services (app, db, prometheus, grafana)

## Configuration Details

### Environment Variables

Key settings in `.env.example`:

- `APP_ENV`: Controls which `.env.[APP_ENV]` file is loaded
- `OPENAI_API_KEY`: LLM API key
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: Observability
- `POSTGRES_*`: Database connection (host, port, credentials)
- `RATE_LIMIT_*`: Per-endpoint rate limits (e.g., `RATE_LIMIT_CHAT="30 per minute"`)
- `LOG_LEVEL`, `LOG_FORMAT`: Logging configuration
- `JWT_SECRET_KEY`: Authentication secret

### Environment-Specific Behavior

The `apply_environment_settings()` method in `config.py` applies defaults:

- **Development**: DEBUG=True, LOG_LEVEL=DEBUG, relaxed rate limits
- **Staging**: DEBUG=False, LOG_LEVEL=INFO, moderate rate limits
- **Production**: DEBUG=False, LOG_LEVEL=WARNING, strict rate limits, graceful degradation
- **Test**: DEBUG=True, LOG_LEVEL=DEBUG, very relaxed rate limits

In production, the application gracefully continues if the database or LangGraph initialization fails, logging warnings instead of crashing.

## Important Implementation Patterns

### LLM Interactions

- The `llm_service` binds tools to the model before chat operations
- LLM calls are wrapped with retry logic (3 retries by default)
- Metrics tracked: `llm_inference_duration_seconds` labeled by model name
- Langfuse callback handlers track all LLM interactions

### Database Operations

- All database operations use SQLModel sessions with the connection pool
- Sessions are context-managed (auto-close)
- Check `database_service.health_check()` before relying on DB state

### Streaming Responses

- `get_stream_response()` uses `graph.astream()` with `stream_mode="messages"`
- Individual token processing errors don't crash the stream (continue on error)
- Yielded tokens are message content strings

### Rate Limiting

- Uses SlowAPI with per-endpoint configuration
- Limits parsed from environment: `RATE_LIMIT_ENDPOINT="N per minute"`
- Default endpoints: root, health, chat, chat_stream, messages, register, login
- Exception handler returns 429 with descriptive error message

### Logging

- Structured logging via `structlog` with context middleware
- `LoggingContextMiddleware` captures user_id and session_id from requests
- Environment-specific formats: JSON (staging/prod), colored console (dev)
- All major operations logged: user creation, session management, LLM calls, errors

## Testing Notes

- Test files follow `test_*.py` or `*_test.py` pattern
- Tests use pytest with httpx for async client
- Configure test environment via `APP_ENV=test` (sets relaxed limits)
- Database tests need PostgreSQL running; consider using fixtures with temp DB

## Debugging Tips

- Enable debug mode via `DEBUG=true` in `.env`
- Check logs in `logs/` directory (if `LOG_DIR` configured)
- Use Langfuse dashboard to trace LLM interactions
- Prometheus metrics available at `/metrics` endpoint
- Grafana dashboards pre-configured in docker-compose

## Mem0 Integration for Cross-Session Memory

This application now integrates **Mem0** for persistent, cross-session memory management. This enables the AI agent to remember user preferences, traits, and facts across multiple conversation sessions.

### How It Works

**Architecture**: Hybrid approach combining LangGraph checkpoints (session state) with Mem0 (long-term memory)

1. **Session Initialization** (`session_service.py:initialize_session`):
   - When a user starts a new session, relevant memories are retrieved from Mem0
   - User context (traits, preferences, facts) is injected into the initial system prompt
   - Agent gains awareness of user history without loading entire conversation histories

2. **Chat-Time Memory Injection** (`chatbot.py:chat` and `chat/stream`):
   - Before each LLM call, search Mem0 for relevant memories based on the current message
   - Retrieved memories are prepended as system context
   - Provides adaptive, query-aware context (not just session-wide context)

3. **Memory Extraction** (`memory_extraction.py`):
   - After each conversation, the agent extracts:
     - **Traits**: User personality, communication style, characteristics
     - **Preferences**: Likes, dislikes, favorite things
     - **Facts**: Background info, status updates, important details
     - **Summary**: Conversation summary for future reference
   - Uses LLM to intelligently categorize and extract memories

4. **Memory Storage** (`session_service.py:finalize_session`):
   - Extracted memories are stored in Mem0 with appropriate categorization
   - Memories are indexed and made searchable for future sessions

### Configuration

Set these environment variables in your `.env` file:

```env
# Enable/disable mem0
MEM0_ENABLED=true

# Mem0 SaaS credentials (from your Mem0 account)
MEM0_API_KEY="your-api-key"
MEM0_ORG_ID="your-org-id"
MEM0_PROJECT_ID="your-project-id"

# Checkpoint settings
# Keep only recent N messages in session checkpoint (older ones archived to mem0)
MEM0_CHECKPOINT_MESSAGE_LIMIT=10

# Token budget for mem0 context injection per LLM call
MEM0_CONTEXT_MAX_TOKENS=500
```

### Service Architecture

- **`app/services/mem0_service.py`**: Core Mem0 API wrapper
  - `add_memory()`: Store new memories
  - `search_memories()`: Semantic search for relevant memories
  - `get_user_context()`: Retrieve all user memories formatted for LLM injection

- **`app/utils/memory_extraction.py`**: Conversation analysis
  - `extract_memories_from_conversation()`: LLM-powered extraction of traits, preferences, facts
  - `extract_conversation_summary()`: Generate concise conversation summaries

- **`app/services/session_service.py`**: Session lifecycle management
  - `initialize_session()`: Load user context from Mem0 at session start
  - `finalize_session()`: Extract and store memories at session end
  - `get_search_result_context()`: Search-based memory retrieval during chat

### Memory Flow

```
New Session Starts
    ↓
initialize_session() → Query Mem0 for user context
    ↓
Inject context into initial system prompt
    ↓
User sends message
    ↓
Chat endpoint calls session_memory_manager.get_search_result_context()
    ↓
Search Mem0 for memories relevant to user's message
    ↓
Inject search results as system context
    ↓
LLM responds with awareness of user history
    ↓
Session ends
    ↓
finalize_session() → Extract memories from conversation
    ↓
Store extracted traits, preferences, facts, summary in Mem0
    ↓
Next session has access to these memories
```

### Usage in Code

```python
# In your chatbot endpoints:
from app.services.session_service import session_memory_manager

# Initialize session with user context
context_messages = await session_memory_manager.initialize_session(session_id, user_id)

# During chat, inject relevant memories
search_context = await session_memory_manager.get_search_result_context(
    session_id=session_id,
    user_id=user_id,
    query=user_message  # Search based on current message
)

# Extract and store memories after conversation
await session_memory_manager.finalize_session(session_id, user_id, messages)
```

### Performance Considerations

- **Memory Search**: Semantic search in Mem0 is fast (~100ms) for typical use cases
- **Context Token Limit**: Configure `MEM0_CONTEXT_MAX_TOKENS` to balance context richness vs. token cost
- **Extraction Cost**: Memory extraction uses LLM (costs tokens) - runs at session end, not per-message
- **Fallback**: If Mem0 is disabled or unavailable, system continues to work without cross-session memory

### Disabling Mem0

To disable Mem0 and revert to single-session memory only:

```env
MEM0_ENABLED=false
```

The system will continue to work normally, but won't retain memories across sessions.

## Langfuse Prompt Management

This application integrates **Langfuse** for dynamic prompt management, enabling system prompts to be updated in real-time without restarting the application.

### How It Works

**Architecture**: Session-level caching approach where prompts are fetched once per session and remain consistent throughout the conversation.

1. **Prompt Fetching** (`app/services/prompt_manager.py:PromptManager.get_prompt`):
   - When a session starts, the system fetches the prompt from Langfuse based on the `prompt_label`
   - Prompts are cached in memory with a 1-hour TTL to reduce API calls
   - If Langfuse is unavailable, falls back to the default `SYSTEM_PROMPT`

2. **Prompt Compilation** (`app/services/prompt_manager.py:PromptManager.compile_prompt`):
   - Prompt templates use Langfuse syntax (`{{variable_name}}`)
   - Variables are substituted at runtime (e.g., `agent_name`, `current_date_and_time`)
   - The compiled prompt is used for all messages in the session

3. **Session-Level Consistency** (`app/core/langgraph/graph.py:_chat`):
   - The prompt is fetched once at the start of the chat (`_chat` node first call)
   - Prompt metadata (template, config, version) is stored in `GraphState`
   - All subsequent messages in the same session use the same prompt
   - Enables A/B testing and gradual rollouts per session

4. **Langfuse Tracing** (`app/api/v1/chatbot.py`):
   - The `prompt_label` is passed to Langfuse traces for analysis
   - Prompt version and metadata are logged for debugging
   - Correlate conversation quality with prompt versions in Langfuse dashboard

### Configuration

Set these environment variables in your `.env` file:

```env
# Langfuse credentials (from your Langfuse account)
LANGFUSE_PUBLIC_KEY="your-public-key"
LANGFUSE_SECRET_KEY="your-secret-key"
LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted instance

# Optional: Override default prompt name
# PROMPT_NAME="custom-prompt"  # defaults to "system-prompt"
```

### API Usage

The chat endpoints accept an optional `prompt_label` query parameter:

```bash
# Use production prompt (default)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Use staging prompt for testing
curl -X POST http://localhost:8000/api/v1/chat?prompt_label=staging \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Use development prompt for debugging
curl -X POST http://localhost:8000/api/v1/chat?prompt_label=development \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

Both `/chat` and `/chat/stream` endpoints support the `prompt_label` parameter.

### Workflow: Creating and Updating Prompts

1. **Create a prompt in Langfuse**:
   - Go to your Langfuse dashboard
   - Create a new prompt named `system-prompt` (or custom name via `PROMPT_NAME` env var)
   - Add template variables using `{{variable_name}}` syntax
   - Supported variables: `agent_name`, `current_date_and_time`
   - Example template:
     ```
     You are {{agent_name}}, a helpful AI assistant.
     Current date and time: {{current_date_and_time}}

     Your instructions go here...
     ```

2. **Create labels for different environments**:
   - `production`: Main prompt used by most users
   - `staging`: Test prompt for development teams
   - `development`: Debug prompt for local testing
   - Custom labels can be added as needed

3. **Version and rollout**:
   - Langfuse automatically versions prompts as you edit them
   - Use labels to control which version is active for different groups
   - Update a label to point to a new version (zero-downtime deployment)
   - Old sessions continue using their original prompt version

4. **Monitor effectiveness**:
   - View conversations in Langfuse dashboard filtered by `prompt_label`
   - Compare conversation quality metrics across prompt versions
   - Use evaluations to score responses by prompt version

### Service Architecture

- **`app/services/prompt_manager.py`**: Core Langfuse prompt management
  - `PromptManager`: Singleton service for fetching, caching, and compiling prompts
  - `PromptData`: Data class holding template, config, version, labels
  - Cache TTL: 1 hour per session (configurable via `cache_ttl_seconds`)
  - `get_prompt()`: Fetch with optional label and version
  - `compile_prompt()`: Substitute variables with runtime values
  - `clear_cache()`: Manual cache invalidation

- **`app/schemas/graph.py`**: Extended `GraphState` fields
  - `prompt_label`: Which Langfuse label was used
  - `prompt_template`: The compiled prompt for this session
  - `prompt_config`: LLM config from Langfuse (model, temperature)
  - `prompt_version`: Version number for tracing
  - `prompt_name`: Prompt name in Langfuse

### Prompt Flow Diagram

```
User sends message with prompt_label=production
    ↓
/chat or /chat/stream endpoint
    ↓
Call agent.get_response(prompt_label="production")
    ↓
Graph invokes _chat node
    ↓
PromptManager.get_prompt("system-prompt", label="production")
    ↓
Check cache → Not found or expired
    ↓
Fetch from Langfuse API
    ↓
Cache prompt for 1 hour
    ↓
Compile prompt with variables
    ↓
Store metadata in GraphState
    ↓
Prepare messages with compiled prompt
    ↓
Call LLM
    ↓
Log prompt metadata to Langfuse
    ↓
Return response to client
    ↓
Next message in same session → Reuse cached prompt (no API call)
```

### Performance Considerations

- **Cache Hit**: ~1ms per message (in-memory lookup)
- **Cache Miss**: ~100-200ms per session (Langfuse API call + compilation)
- **TTL**: 1 hour default - balance between freshness and API quota
- **Fallback**: If Langfuse unavailable, uses default SYSTEM_PROMPT (always available)

### Disabling Langfuse Prompt Management

To disable dynamic prompts and use the hardcoded default:

```env
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
```

The system will use the default `SYSTEM_PROMPT` from `app/core/prompts.py` for all sessions.

## Testing with the Browser Client

A full-featured HTML/JavaScript test client is included (`test_client.html`):

```bash
# Option 1: Serve from the same port as the API (no CORS issues)
python -m http.server 8000 --directory . &
# Then open: http://localhost:8000/test_client.html

# Option 2: Serve on a different port
python -m http.server 8001 &
# Then open: http://localhost:8001/test_client.html
# (The client will auto-detect CORS and expand allowed origins in dev mode)
```

Features:
- User registration and login
- Session management
- Real-time chat with streaming and normal modes
- Message history viewing and clearing
- Configurable server URL with connection testing
- Console logging for debugging

## Deployment Considerations

- Database migrations: ORM creates tables automatically on startup
- Checkpoint tables must exist for LangGraph state persistence
- Connection pooling configured per environment (pool_size, max_overflow)
- Graceful degradation in production if DB or graph init fails
- Structured JSON logging for log aggregation systems
- Health check endpoint: `GET /health` with component status
- CORS configuration: In development mode, localhost origins (3000, 3001, 8000, 8001, 8080) are automatically allowed. In production, set explicit `ALLOWED_ORIGINS` in `.env` file
