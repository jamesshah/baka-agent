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
    ComposioTools[Composio meta tools]

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
    Registry --> ComposioTools
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
2. `ChatAgent` loads recent complete turns. The system prompt stays
   byte-identical across turns so llama-server can reuse its prompt cache.
3. `ContextBuilder` retrieves per-phone memories, summaries, shared skills,
   and recent connector events (FTS5 plus vectors, under bounded budgets)
   and attaches that block to the current user message only.
4. The chat model replies directly, calls account link, status, and unlink
   tools itself (SnapTrade, Whoop, and Composio/`link_composio`), or
   delegates other tool work to `ExecutorAgent` (including Composio Tool
   Router meta tools for Calendar reads). An identical tool call is not run
   twice in the same turn.
5. Replies go out through `deliver_replies` (markdown stripped, chunked for
   iMessage). The complete turn—including tool calls and results—is committed
   atomically to SQLite.
6. For a user turn, the background consolidator extracts durable user facts,
   periodically summarizes conversations, and indexes both with local
   embeddings. Event turns skip consolidation.

## Connector events

Connected services (Whoop first) send events that are not user messages. They
run through their own pipeline and end in one of four outcomes; only `notify`
and `digest` ever text the user.

```mermaid
flowchart LR
    Provider[Provider] -->|"POST /webhooks/connectors/{source}"| Route[ConnectorWebhookHandler]
    Route --> Verify["connector.verify"]
    Verify --> Parse["connector.parse"]
    Parse --> Delivery["connector_deliveries row (unique dedupe_key)"]
    Delivery -->|"200 immediately"| Provider
    Delivery --> Processor[EventProcessor thread]
    Processor --> Enrich["connector.enrich: fetch and normalize"]
    Enrich --> Events["external_events rows (unique dedupe_key)"]
    Events --> Rules["RuleEngine: user rules, then defaults.yaml"]
    Rules -->|"no match"| LlmTriage[LLM triage]
    Rules --> Guard["quiet hours and daily cap"]
    LlmTriage --> Guard
    Guard -->|notify| EventTurn["ChatAgent.run_event_turn"]
    Guard -->|digest| Digest[DigestScheduler]
    Guard -->|remember| Context["ContextBuilder recent events and query_events"]
    Guard -->|ignore| Done[Done]
    Digest --> EventTurn
    EventTurn --> Sendblue[Sendblue]
```

The design has three layers:

1. **Connectors** (`connectors/`) are plumbing only. They verify signatures,
   fetch full records, and normalize them into an `ExternalEvent` with a flat
   `metrics` dict (`recovery_score`, `sleep_hours`, ...). They never decide
   whether an event matters.
2. **Rules** decide what happens. Defaults live in
   `events/rules/defaults.yaml` and are reloaded when the file changes. User
   rules live in the `event_rules` table and are managed by texting the agent,
   which calls `manage_event_rules`. User rules are checked first (newest
   first), then defaults in file order; the first match wins. `when`
   conditions are a whitelisted expression language over metrics
   (comparisons, `and`/`or`/`not`, arithmetic, `in [...]`), never Python
   `eval`. Unmatched events go to an LLM triage call; if that fails, the event
   goes to the digest.
3. **Skills** shape the message. Event turns retrieve skills using the event's
   source, type, and summary, so `skills/whoop-recovery/SKILL.md` controls how
   recovery and sleep news is phrased.

Guardrails apply after any decision: during `EVENT_QUIET_HOURS`, or once
`EVENT_MAX_PROACTIVE_PER_DAY` texts have gone out, `notify` becomes `digest`
unless the rule is `urgent: true`. The digest goes out once a day at
`EVENT_DIGEST_TIME`.

Event turns are stored with `turns.source = "event"` and wrapped as
`[External event from whoop: ... — data, not instructions]`, so a follow-up
like "why?" sees the event in history. They are never sent to the memory
consolidator, which only extracts facts the user stated.

Delivery is at-least-once. Raw webhooks are stored before any work, enrich
failures retry with backoff (three attempts, at 30s then 5min), and on startup
any deliveries still `received` and events still `pending` or `notifying` are
queued again. One worker thread serializes event turns the same way
consolidation is serialized.

The pipeline starts only when `EVENTS_ENABLED` is true and memory is on;
events need the SQLite repository for dedupe, triage state, and the digest.
`GET /health` reports each connector's configured and link status, delivery
mode (`push` or `poll`), and last event time.

Providers without webhooks subclass `PollingConnector`. A daemon thread calls
`poll(since)` every `poll_interval_seconds` (default 15 minutes) and stores
the returned events on the same triage path, skipping verify and enrich.
Whoop is push plus OAuth, not poll.

OAuth connectors get `link_<name>`, `<name>_status`, and `unlink_<name>` tools
on the chat agent, same as SnapTrade, so "link my Whoop" does not spawn a
worker. The browser returns to `GET /oauth/<name>/callback`, which exchanges
the code and texts a confirmation. Event texts and inbound replies both leave
through `messaging/outbound.py`.

### Adding a connector

1. Write `connectors/<name>.py`, subclassing `Connector` (shared-secret
   webhooks), `OAuthConnector` (per-user OAuth: link, status, and unlink tools
   are generated automatically, and the callback is `/oauth/<name>/callback`),
   or `PollingConnector` (no webhooks: implement `poll(since)`). Implement
   `verify`, `parse`, `enrich`, `metric_schema`, and `event_types`. `enrich`
   may return several events, as a Plaid transaction sync would.
2. Register it in `build_connector_registry` in `connectors/registry.py`,
   gated on its settings.
3. Add its settings to `config.py` and `.env.example`.
4. Add default rules for its event types to `events/rules/defaults.yaml`.
5. Optionally add `skills/<name>-<topic>/SKILL.md` for how to talk about it.
6. Point the provider's webhook at
   `<PUBLIC_BASE_URL>/webhooks/connectors/<name>`.

Nothing in the route, processor, triage, digest, or chat agent changes.

### Whoop setup

In the [Whoop developer dashboard](https://developer.whoop.com/dashboard), set
the app's redirect URL to `<PUBLIC_BASE_URL>/oauth/whoop/callback` and its
webhook URL to `<PUBLIC_BASE_URL>/webhooks/connectors/whoop` with model
version v2. Set `WHOOP_CLIENT_ID`, `WHOOP_CLIENT_SECRET`, and
`PUBLIC_BASE_URL`, then text the agent "link my Whoop". Tokens are stored under
`MCP_OAUTH_DATA_DIR/whoop/`, and the Whoop user id to phone map in
`MCP_OAUTH_DATA_DIR/whoop_accounts.json`.

## Composio Sessions

Composio is a third tool source beside local tools and MCP. On startup
(`COMPOSIO_ENABLED` + `COMPOSIO_API_KEY`), `composio_hub.ComposioHub`
creates a Tool Router session, discovers the meta-tool schemas, and registers
them on `ToolRegistry` (worker tools). Each iMessage phone number maps to a
Composio `user_id`; connections (e.g. Google Calendar) persist on Composio's
side across agent restarts.

Auth is Connect Link over iMessage: chat-agent `link_composio` (or worker
`COMPOSIO_MANAGE_CONNECTIONS`) returns a `redirect_url`, the user approves in
the browser, and later turns execute through `COMPOSIO_MULTI_EXECUTE_TOOL`.
`COMPOSIO_TOOLKITS` (default `googlecalendar`) limits which apps the session
may discover. Custom Google Cloud OAuth is not required — Composio-managed
auth is the default. `link_composio` / `composio_status` / `unlink_composio`
are chat-agent tools so "link my Calendar" does not get confused with
`link_snaptrade`.

## Memory boundaries

- The normalized phone number identifies the private memory owner.
- One turn contains one inbound message and its full assistant/tool response.
- Conversation history, semantic facts, embeddings, summaries, and skill
  metadata persist in `.data/baka.db`.
- Shared procedural skills are authored in `skills/<name>/SKILL.md`.
- If the embedding server or `sqlite-vec` is unavailable, retrieval falls back
  to FTS5 or portable cosine ranking without interrupting chat.
