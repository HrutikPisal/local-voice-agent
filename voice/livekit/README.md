# Self-hosted LiveKit (Apache 2.0)

LiveKit gives you **WebRTC transport** and **agent workers** under your control. This stack uses the **open-source server** (not LiveKit Cloud) with **local faster-whisper** STT — same transcription engine as `python -m voice`.

## What you get

| Piece | Role |
| --- | --- |
| [livekit-server](https://github.com/livekit/livekit) (Docker) | Signaling + media (Apache 2.0) |
| `voice/livekit/agent.py` | Agent worker: room audio → STT → confirm/edit in terminal |
| `voice/stt.py` | Shared Whisper transcription |

No LiveKit Cloud account required for self-hosting.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows)
- **Ollama running** locally (the agent worker initializes an Ollama LLM handle; it does not auto-reply in STT-only mode)
- [LiveKit CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli/) (optional but recommended): `winget install LiveKit.LiveKitCLI`

## 1. Start LiveKit server

```bash
cd voice/livekit
docker compose up -d
```

Dev credentials (built into `--dev` mode):

- URL: `ws://127.0.0.1:7880`
- API key: `devkey`
- API secret: `secret`

Check: open `http://127.0.0.1:7880` — you should get a LiveKit HTTP response (not a blank error).

## 2. Configure the agent worker

```bash
copy voice\livekit\.env.example voice\livekit\.env.local
pip install -r voice/requirements-livekit.txt
```

## 3. Run the STT agent

**Terminal testing (mic through LiveKit pipeline):**

```bash
python -m voice.livekit console
```

Or with the CLI (auto-reload):

```bash
cd voice/livekit
lk agent dev
```

**With Agents Playground:** point your project at the same `LIVEKIT_*` values in `.env.local` and dispatch agent `local-cli-voice-stt`.

## 4. Confirm / edit

When speech is transcribed, the worker prints the same review prompt as `python -m voice`:

- Enter → accept  
- Type corrections → Enter  
- `.cancel` → discard  

Confirmed text: `[CONFIRMED INPUT] ...` (also sent on room data topic `voice_stt` as JSON).

## CLI-only fallback

Push-to-talk without Docker/LiveKit:

```bash
python -m voice
```

Use `.talk` — same STT and confirm/edit, no WebRTC.

## Production self-hosting

Replace `--dev` with a config file, TLS, and your own `LIVEKIT_KEYS`. See [LiveKit self-hosting docs](https://docs.livekit.io/transport/self-hosting/).
