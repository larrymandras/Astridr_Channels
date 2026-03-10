# astridr-channels

Channel adapters and message router for the Astridr AI agent framework. Each channel normalizes platform-specific messaging into a unified `IncomingMessage`/`OutgoingMessage` interface, allowing the agent core to remain transport-agnostic.

## Channels

| Channel | Description | Required Env Vars / Dependencies |
|---------|-------------|----------------------------------|
| **Telegram** | Telegram bot via long-polling or webhooks | `TELEGRAM_BOT_TOKEN`; `python-telegram-bot` |
| **Slack** | Slack app using Socket Mode or Events API | `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`; `slack-bolt` |
| **Email** | IMAP listener + SMTP sender for email conversations | `EMAIL_IMAP_HOST`, `EMAIL_SMTP_HOST`, `EMAIL_USER`, `EMAIL_PASSWORD`; `aiosmtplib` |
| **Web** | FastAPI-based HTTP/WebSocket endpoint | `WEB_HOST` (optional), `WEB_PORT` (optional); `fastapi`, `uvicorn` |
| **Voice** | Always-on microphone with wake word detection and TTS playback | `sounddevice`, `soundfile`; optional: `openwakeword` or `pvporcupine` + `PICOVOICE_ACCESS_KEY` |
| **Router** | Central message dispatcher — resolves profiles, manages sessions, runs security pipeline, invokes the agent loop | No additional env vars (uses injected dependencies) |
