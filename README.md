# baka-agent

Local iMessage agent: Sendblue webhooks in, a from-scratch agent loop, and llama.cpp (`llama-server`) for local inference. No agent/LLM SDK — just FastAPI, the official Sendblue Python SDK, and raw HTTP to the model.

```
iMessage → Sendblue → POST /webhooks/receive → agent loop → llama-server → Sendblue → iMessage
```

## Setup

```bash
cd baka-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your Sendblue credentials + from number
```

### `.env`

| Variable                                              | Description                                                                                                          |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `SENDBLUE_API_KEY` / `SENDBLUE_API_SECRET`            | From [dashboard.sendblue.com](https://dashboard.sendblue.com)                                                        |
| `SENDBLUE_FROM_NUMBER`                                | Your Sendblue line (E.164, e.g. `+15551234567`)                                                                      |
| `SENDBLUE_GLOBAL_WEBHOOK_SECRET`                      | Must match Sendblue's webhook `globalSecret` (or per-webhook `secret`). Verified via the `sb-signing-secret` header. |
| `SENDBLUE_WEBHOOK_URL`                                | Public `https://…/webhooks/receive` URL. On startup the agent checks Sendblue and registers this URL if missing.     |
| `LLAMA_BASE_URL`                                      | OpenAI-compatible base URL (default `http://127.0.0.1:8080/v1`)                                                      |
| `LLAMA_MODEL`                                         | Model id passed to `/chat/completions` (often just `local`)                                                          |
| `LLAMA_HF_REPO`                                       | GGUF repo for `./run.sh` → `llama-server -hf` (default `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL`)                     |
| `LLAMA_CACHE`                                         | llama.cpp HF download cache (default: `LLAMA_HF_REPO` without `:quant`; relative → `<project>/.cache/…`)             |
| `LLAMA_CACHE_RAM`                                     | Host-RAM prompt cache MiB (default `1024`)                                                                           |
| `LLAMA_CTX_SIZE` / `LLAMA_PARALLEL` / `LLAMA_THREADS` | Context, slots, CPU threads (defaults `16384` / `1` / `8`)                                                           |
| `LLAMA_BATCH` / `LLAMA_UBATCH`                        | Batch sizes (defaults `1024` / `256`)                                                                                |
| `LLAMA_N_GPU_LAYERS` / `LLAMA_FIT_TARGET`             | Metal layers + `--fit` free-memory margin MiB (defaults `99` / `4096`)                                               |
| `LLAMA_PORT` / `AGENT_HOST` / `AGENT_PORT`            | Ports/host used by `./run.sh` (defaults `8080` / `0.0.0.0` / `8000`)                                                 |
| `ALLOWED_NUMBERS`                                     | Optional comma-separated E.164 allowlist; empty = allow all                                                          |
| `MAX_HISTORY_MESSAGES` / `MAX_AGENT_ITERATIONS`       | History trim + tool-call loop cap                                                                                    |
| `MCP_ENABLED`                                         | Connect MCP servers on startup (default `true`)                                                                      |
| `MCP_CONFIG_PATH`                                     | Path to Cursor-style `mcp.json` (default `mcp.json`)                                                                 |
| `MCP_OAUTH_DATA_DIR`                                  | Where OAuth tokens are stored (default `.data/mcp-oauth`)                                                            |
| `MCP_OAUTH_OWNER_NUMBER`                              | Phone that owns SnapTrade tokens (defaults to sole `ALLOWED_NUMBERS` entry)                                          |

On free shared-line Sendblue plans, add the recipient as a contact and have them text your number once to complete verification before messaging works.

## Run

### Quick start (macOS, background)

```bash
chmod +x run.sh   # once
./run.sh start            # llama-server + agent
./run.sh restart agent    # only re-run the agent
./run.sh restart llama    # only re-run llama-server
./run.sh status
./run.sh logs             # Ctrl-C to stop tailing
./run.sh stop
```

Override the Hugging Face model or ports via `.env` (read by `./run.sh`) or the shell:

```bash
# .env — Gemma 12B on M4 24 GB
LLAMA_HF_REPO=unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL
LLAMA_CACHE=unsloth/gemma-4-12b-it-GGUF
LLAMA_CACHE_RAM=1024
LLAMA_CTX_SIZE=16384
LLAMA_PARALLEL=1
LLAMA_THREADS=8

# or one-shot:
LLAMA_HF_REPO=unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL ./run.sh start
```

`./run.sh` exports `LLAMA_CACHE`, uses one slot, quantized KV (`q8_0`), Flash Attention, and `--fit-target 4096` (~4 GiB free for other apps). Defaults assume Gemma 12B (~7.4 GB), which leaves more room than Muse 30B. `/health` → `checks.model.prompt_cache` reports cache hit ratio from llama.cpp `/slots`.

PIDs and logs live under `.run/` (gitignored).

### Manual: 1. Start llama.cpp

Install (macOS):

```bash
brew install llama.cpp
```

Serve a small instruct GGUF with Jinja chat templates (needed for tool calls):

```bash
llama-server -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --port 8080 --jinja
```

Any OpenAI-compatible server that exposes `/v1/chat/completions` works — point `LLAMA_BASE_URL` at it.

### Manual: 2. Start the agent server

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `curl http://127.0.0.1:8000/health`

`/health` probes tools, MCP servers, the model server, and whether a Sendblue receive webhook is registered. Overall `status` is `ok`, `degraded`, or `error` (HTTP 503 when `error`).

### 3. Expose the webhook (local tunnel)

Sendblue must reach your machine. With [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated `https://….trycloudflare.com` URL.

### 4. Register the receive webhook

Set `SENDBLUE_WEBHOOK_URL` to your tunnel URL + path, then start (or restart) the agent — it checks Sendblue on startup and registers the URL if missing:

```bash
# .env
SENDBLUE_WEBHOOK_URL=https://<your-tunnel-host>/webhooks/receive
SENDBLUE_GLOBAL_WEBHOOK_SECRET=your-shared-secret
```

Or register without restarting:

```bash
curl -X POST http://127.0.0.1:8000/webhooks/register \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://<your-tunnel-host>/webhooks/receive"}'
```

You can still set it in the dashboard (**Settings → Webhooks → Receive**) if you prefer.

Every `/webhooks/receive` request must include an `sb-signing-secret` header matching `SENDBLUE_GLOBAL_WEBHOOK_SECRET`, or it is rejected with `401`.

Outbound replies are stripped of Markdown and split into ≤2000-character chunks before Sendblue send.

### 5. Text your Sendblue number

Inbound messages hit `/webhooks/receive`, which returns `200` immediately and runs the agent in the background. The reply is sent back through Sendblue.

Photos, audio, and video attachments are downloaded from Sendblue and sent to the model when llama-server advertises that modality (`GET /v1/models`). Otherwise the agent replies: `I don't have multi-modal capabilities.`

Demo tool: ask _"what time is it?"_ — the model can call `get_current_time`.

## MCP tools

The agent can also call tools from one or more [MCP](https://modelcontextprotocol.io) servers. Config uses the same JSON shape as Cursor (`mcpServers`).

```bash
cp mcp.json.example mcp.json
# edit mcp.json — add as many servers as you want
```

Example with local + remote OAuth (SnapTrade):

```json
{
	"mcpServers": {
		"filesystem": {
			"command": "npx",
			"args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
		},
		"snaptrade": {
			"url": "https://mcp.snaptrade.com/mcp",
			"auth": "oauth",
			"scopes": ["read"]
		},
		"remote-api": {
			"url": "https://example.com/mcp",
			"headers": {
				"Authorization": "Bearer ${env:MY_TOKEN}"
			}
		}
	}
}
```

Supported transports:

| Config                                        | Transport                                                  |
| --------------------------------------------- | ---------------------------------------------------------- |
| `command` + optional `args` / `env` / `cwd`   | stdio (local process)                                      |
| `url` (default) or `"type": "streamableHttp"` | Streamable HTTP                                            |
| `url` + `"type": "sse"`                       | SSE                                                        |
| `"auth": "oauth"` on a `url` server           | OAuth device-code link (tokens under `MCP_OAUTH_DATA_DIR`) |

On startup the client connects to every non-OAuth server, discovers tools, and registers them as `mcp__<server>__<tool>`. OAuth servers (SnapTrade) stay `pending_auth` until linked; if tokens already exist on disk they reconnect automatically with silent refresh — no re-login.

### SnapTrade (iMessage OAuth)

SnapTrade MCP is read-only Personal OAuth ([docs](https://docs.snaptrade.com/docs/mcp-server)). Add the `snaptrade` entry from `mcp.json.example`, set `MCP_OAUTH_OWNER_NUMBER` (or a single `ALLOWED_NUMBERS` entry), then text:

1. **link snaptrade** — agent replies with a verification URL; open it, log in, approve **read**.
2. After approval you get a confirmation text; portfolio tools (`mcp__snaptrade__*`) are registered.
3. Ask about balances, positions, orders as usual. Tokens persist under `.data/mcp-oauth/` across restarts.
4. **unlink snaptrade** — revokes access and deletes local tokens (or revoke under SnapTrade **Settings → Connected apps**).

Local helper tools: `link_snaptrade`, `snaptrade_status`, `unlink_snaptrade`. Set `MCP_ENABLED=false` to skip MCP entirely.

## Project layout

| Path               | Role                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`           | FastAPI: `/health`, `/webhooks/register`, `/webhooks/receive`; wires adapters + MCP lifespan; ensures Sendblue webhook on start |
| `config.py`        | Settings from `.env`                                                                                                            |
| `agents/`          | `Agent` ABC + `ChatAgent` (owns `SYSTEM_PROMPT`)                                                                                |
| `tools/`           | `Tool` ABC, `ToolRegistry`, local tools (time, SnapTrade link/status/unlink)                                                    |
| `mcp_client/`      | Cursor-style `mcp.json` loader, multi-server hub, OAuth device flow + token store                                               |
| `llm/`             | `LLMClient` ABC + `LlamaServerAdapter`                                                                                          |
| `messaging/`       | `MessagingClient` ABC + `SendblueAdapter`                                                                                       |
| `webhooks/`        | `SendblueWebhookHandler` (verify + receive + reply)                                                                             |
| `mcp.json.example` | Sample MCP server config                                                                                                        |

## Notes

- The webhook handler acknowledges with HTTP 200 right away, then processes and replies asynchronously. If llama latency is high this avoids Sendblue retries.
- If the served model/template does not emit structured `tool_calls`, the loop treats the response as final text (still capped by `MAX_AGENT_ITERATIONS`).
- Conversation history is in-memory only; it resets when the process restarts.
- Optional `ALLOWED_NUMBERS` keeps strangers from driving your local model.
