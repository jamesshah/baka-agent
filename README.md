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

| Variable | Description |
|---|---|
| `SENDBLUE_API_KEY` / `SENDBLUE_API_SECRET` | From [dashboard.sendblue.com](https://dashboard.sendblue.com) |
| `SENDBLUE_FROM_NUMBER` | Your Sendblue line (E.164, e.g. `+15551234567`) |
| `SENDBLUE_GLOBAL_WEBHOOK_SECRET` | Must match Sendblue's webhook `globalSecret` (or per-webhook `secret`). Verified via the `sb-signing-secret` header. |
| `LLAMA_BASE_URL` | OpenAI-compatible base URL (default `http://127.0.0.1:8080/v1`) |
| `LLAMA_MODEL` | Model id passed to `/chat/completions` (often just `local`) |
| `ALLOWED_NUMBERS` | Optional comma-separated E.164 allowlist; empty = allow all |
| `SYSTEM_PROMPT` | System prompt for the agent |
| `MAX_HISTORY_MESSAGES` / `MAX_AGENT_ITERATIONS` | History trim + tool-call loop cap |

On free shared-line Sendblue plans, add the recipient as a contact and have them text your number once to complete verification before messaging works.

## Run

### 1. Start llama.cpp

Install (macOS):

```bash
brew install llama.cpp
```

Serve a small instruct GGUF with Jinja chat templates (needed for tool calls):

```bash
llama-server -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --port 8080 --jinja
```

Any OpenAI-compatible server that exposes `/v1/chat/completions` works — point `LLAMA_BASE_URL` at it.

### 2. Start the agent server

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `curl http://127.0.0.1:8000/health`

### 3. Expose the webhook (local tunnel)

Sendblue must reach your machine. With [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated `https://….trycloudflare.com` URL.

### 4. Register the receive webhook

Dashboard: **Settings → Webhooks → Receive** → set to:

```
https://<your-tunnel-host>/webhooks/receive
```

Or via the SDK / a one-off Python snippet:

```python
from sendblue_api import SendblueAPI
import os

client = SendblueAPI(
    api_key=os.environ["SENDBLUE_API_KEY"],
    api_secret=os.environ["SENDBLUE_API_SECRET"],
)
client.webhooks.create(
    webhooks=["https://<your-tunnel-host>/webhooks/receive"],
    type="receive",
    global_secret=os.environ["SENDBLUE_GLOBAL_WEBHOOK_SECRET"],
)
```

Every `/webhooks/receive` request must include an `sb-signing-secret` header matching `SENDBLUE_GLOBAL_WEBHOOK_SECRET`, or it is rejected with `401`.

### 5. Text your Sendblue number

Inbound messages hit `/webhooks/receive`, which returns `200` immediately and runs the agent in the background. The reply is sent back through Sendblue.

Demo tool: ask *"what time is it?"* — the model can call `get_current_time`.

## Project layout

| Path | Role |
|---|---|
| `app.py` | FastAPI: `/health`, `/webhooks/receive`; wires adapters |
| `config.py` | Settings from `.env` |
| `agents/` | `Agent` ABC + `ChatAgent` |
| `tools/` | `Tool` ABC, `ToolRegistry`, `GetCurrentTimeTool` |
| `llm/` | `LLMClient` ABC + `LlamaServerAdapter` |
| `messaging/` | `MessagingClient` ABC + `SendblueAdapter` |
| `webhooks/` | `SendblueWebhookHandler` (verify + receive + reply) |

## Notes

- The webhook handler acknowledges with HTTP 200 right away, then processes and replies asynchronously. If llama latency is high this avoids Sendblue retries.
- If the served model/template does not emit structured `tool_calls`, the loop treats the response as final text (still capped by `MAX_AGENT_ITERATIONS`).
- Conversation history is in-memory only; it resets when the process restarts.
- Optional `ALLOWED_NUMBERS` keeps strangers from driving your local model.
