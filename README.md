# haruka-voice-bridge

Bridge between Telnyx Call Control and Haruka's voice models, so the agent drives
phone calls itself instead of handing the conversation to a vendor's voice runtime.

## What it does

- `POST /telnyx/events` — Telnyx webhook receiver. Verifies the Ed25519 signature over the
  **raw body** before anything else touches the payload, then handles call events and inbound
  SMS (`message.received`).
- `WS /telnyx/media` — the forked call audio. Telnyx sends base64 G.711 µ-law at 8kHz; we
  detect end of utterance, transcribe it, ask a model for a reply, synthesize with Grok TTS,
  and push the result back as base64 mp3 frames. `clear` is used for barge-in.
- `GET /healthz` — liveness only.

## Voice loop

```
Telnyx media stream ──> µ-law 8kHz ──> WAV(PCM16) ──> STT (OpenRouter)
                                                          │
                       base64 mp3 frames <── Grok TTS <── reply ◀── LLM
```

## Environment

| Variable | Purpose |
| --- | --- |
| `TELNYX_PUBLIC_KEY` | Account public key from Mission Control (API Keys → Public Key). Used to verify webhooks. |
| `TELNYX_API_KEY` | Only needed for outbound dialing / replies. |
| `OPENROUTER_API_KEY` | STT, LLM and TTS calls. |
| `STT_MODEL` / `TTS_MODEL` / `TTS_VOICE` / `LLM_MODEL` | Model selection. Defaults: `x-ai/grok-stt-1.0`, `x-ai/grok-voice-tts-1.0`, `ara`, `x-ai/grok-4.6`. |
| `STREAM_AUTH_TOKEN` | Passed to Telnyx as `stream_auth_token`; also checked as `?token=` on the media socket. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional. Forwards inbound SMS to a chat. |

## Security posture

- Only `443` is exposed upstream; TLS terminates at the proxy. No SIP, no RTP range.
- Unsigned or stale webhooks are rejected with 401 before any logic runs.
- The media socket only accepts a call ID this process dialed (`LIVE_CALLS`), plus the shared token.
- No audio, transcripts or credentials are persisted. Secrets come from env only.
- Caller speech is treated as untrusted input in the system prompt: it can never trigger a
  tool call, transfer, refund or spend. Actions require out-of-band confirmation.

## Notes

- Latency today is roughly 2-4s per turn because STT is a batch round trip. Moving to a
  streaming STT socket and `stream_bidirectional_mode: rtp` is the next optimization.
- Interrupting Haruka mid-sentence triggers `clear`, which flushes queued audio immediately.
