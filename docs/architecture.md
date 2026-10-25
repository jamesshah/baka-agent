# Agent architecture

```mermaid
flowchart TB
    User[iMessage user]
    Sendblue[Sendblue]
    Webhook["FastAPI webhook"]
    ChatAgent[ChatAgent]
    ContextBuilder[ContextBuilder]
    ToolLoop[Agent tool loop]
    ChatModel["llama.cpp chat server"]
    MemoryTool[ManageMemoryTool]
    SpawnTool[SpawnAgentTool]
    Executor[ExecutorAgent]
    Registry[ToolRegistry]
    LocalTools[Local tools]
    McpTools[MCP tools]

    subgraph memorySystem [Local memory system]
        Repository["SQLAlchemy repository"]
        SQLite["SQLite database"]
        FTS["FTS5 indexes"]
        VectorSearch["sqlite-vec or cosine fallback"]
        Consolidator["Async memory consolidator"]
        Summaries["Conversation summaries"]
        Semantic["Semantic facts and preferences"]
        Skills["Markdown skills"]
        SkillIndexer[Skill indexer]
        EmbeddingClient[Embedding client]
        EmbeddingModel["llama.cpp embedding server"]
        Alembic[Alembic migrations]
    end

    User --> Sendblue
    Sendblue --> Webhook
    Webhook --> ChatAgent
    ChatAgent --> ContextBuilder
    ContextBuilder --> Repository
    Repository --> FTS
    Repository --> VectorSearch
    FTS --> ContextBuilder
    VectorSearch --> ContextBuilder
    ContextBuilder --> ToolLoop
    ToolLoop <--> ChatModel
    ToolLoop --> MemoryTool
    MemoryTool --> Repository
    ToolLoop --> SpawnTool
    SpawnTool --> Executor
    Executor --> Registry
    Registry --> LocalTools
    Registry --> McpTools
    Executor <--> ChatModel
    ToolLoop --> ChatAgent
    ChatAgent --> Sendblue
    Sendblue --> User

    ChatAgent -->|"begin and complete turn"| Repository
    Repository --> SQLite
    SQLite -->|"recent complete turns"| ContextBuilder
    ChatAgent -->|"completed user turn"| Consolidator
    Consolidator <--> ChatModel
    Consolidator --> Semantic
    Consolidator --> Summaries
    Semantic --> Repository
    Summaries --> Repository
    Consolidator --> EmbeddingClient
    MemoryTool --> EmbeddingClient
    SkillIndexer --> EmbeddingClient
    EmbeddingClient <--> EmbeddingModel
    EmbeddingClient --> Repository
    Skills --> SkillIndexer
    SkillIndexer --> Repository
    Alembic --> SQLite
```

## Runtime flow

1. Sendblue delivers an inbound iMessage to the FastAPI webhook.
2. `ChatAgent` loads recent complete turns and retrieves relevant per-phone
   memories, summaries, and shared skills.
3. `ContextBuilder` combines FTS5 and vector results under bounded context
   budgets.
4. The chat model replies directly or delegates to `ExecutorAgent`, which gets
   only the selected context and tools needed for the task.
5. The complete turn—including tool calls and results—is committed atomically
   to SQLite.
6. The background consolidator extracts durable user facts, periodically
   summarizes conversations, and indexes both with local embeddings.

## Memory boundaries

- The normalized phone number identifies the private memory owner.
- One turn contains one inbound message and its full assistant/tool response.
- Conversation history, semantic facts, embeddings, summaries, and skill
  metadata persist in `.data/baka.db`.
- Shared procedural skills are authored in `skills/<name>/SKILL.md`.
- If the embedding server or `sqlite-vec` is unavailable, retrieval falls back
  to FTS5 or portable cosine ranking without interrupting chat.
