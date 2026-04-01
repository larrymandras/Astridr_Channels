"""Web chat channel — FastAPI + Server-Sent Events for real-time streaming.

Serves a minimal HTML chat interface at ``/``, accepts messages via
``POST /api/chat``, and pushes responses back through an SSE stream at
``GET /api/chat/{chat_id}/stream``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from astridr.channels.base import (
    BaseChannel,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
)
from astridr.channels.briefing_dashboard import register_briefing_dashboard

logger = structlog.get_logger()


# ── SSE Manager ────────────────────────────────────────────────────────


class SSEManager:
    """Manages per-chat SSE connections for real-time message delivery."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, chat_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Register a new SSE listener for a chat."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if chat_id not in self._subscribers:
            self._subscribers[chat_id] = []
        self._subscribers[chat_id].append(queue)
        logger.debug("sse.subscribed", chat_id=chat_id)
        return queue

    def unsubscribe(self, chat_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a listener from a chat."""
        if chat_id in self._subscribers:
            try:
                self._subscribers[chat_id].remove(queue)
            except ValueError:
                pass
            if not self._subscribers[chat_id]:
                del self._subscribers[chat_id]
        logger.debug("sse.unsubscribed", chat_id=chat_id)

    async def publish(self, chat_id: str, event: str, data: dict[str, Any]) -> None:
        """Push an event to all listeners on a given chat_id."""
        if chat_id not in self._subscribers:
            return
        payload = {"event": event, "data": data}
        for queue in self._subscribers[chat_id]:
            await queue.put(payload)
        logger.debug("sse.published", chat_id=chat_id, event_type=event)

    async def publish_typing(self, chat_id: str) -> None:
        """Send a typing indicator event."""
        await self.publish(chat_id, "typing", {"status": "typing"})


# ── HTML Chat Interface ────────────────────────────────────────────────

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Astridr Web Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; height: 100vh;
    display: flex; flex-direction: column;
  }
  header {
    background: #16213e; padding: 16px 24px;
    border-bottom: 1px solid #0f3460; text-align: center;
  }
  header h1 { font-size: 1.2rem; color: #e94560; }
  #messages {
    flex: 1; overflow-y: auto; padding: 16px 24px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; line-height: 1.4; }
  .msg.user { align-self: flex-end; background: #0f3460; }
  .msg.bot { align-self: flex-start; background: #16213e; border: 1px solid #0f3460; }
  .msg.typing { align-self: flex-start; background: #16213e; border: 1px solid #0f3460;
    font-style: italic; color: #888; }
  #input-area {
    display: flex; padding: 12px 24px; background: #16213e;
    border-top: 1px solid #0f3460; gap: 8px;
  }
  #input-area input {
    flex: 1; padding: 10px 14px; border: 1px solid #0f3460;
    border-radius: 8px; background: #1a1a2e; color: #e0e0e0;
    font-size: 0.95rem; outline: none;
  }
  #input-area input:focus { border-color: #e94560; }
  #input-area button {
    padding: 10px 20px; border: none; border-radius: 8px;
    background: #e94560; color: #fff; cursor: pointer;
    font-size: 0.95rem; font-weight: 600;
  }
  #input-area button:hover { background: #c73650; }
#mic-btn {
  padding: 10px 16px; border: none; border-radius: 8px;
  background: #0f3460; color: #e0e0e0; cursor: pointer;
  font-size: 0.95rem; font-weight: 600; transition: background 0.2s;
}
#mic-btn:hover { background: #1a4a8a; }
#mic-btn.recording { background: #e94560; color: #fff; }
#mic-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.waveform {
  display: none; align-items: center; gap: 3px; height: 20px;
}
.waveform.active { display: flex; }
.waveform span {
  display: block; width: 3px; background: #e94560; border-radius: 2px;
  animation: wave 0.8s ease-in-out infinite;
}
.waveform span:nth-child(2) { animation-delay: 0.15s; }
.waveform span:nth-child(3) { animation-delay: 0.3s; }
.waveform span:nth-child(4) { animation-delay: 0.45s; }
@keyframes wave {
  0%, 100% { height: 6px; }
  50% { height: 18px; }
}
</style>
</head>
<body>
<header><h1>Astridr Chat</h1></header>
<div id="messages"></div>
<div id="input-area">
  <input id="msg-input" type="text" placeholder="Type a message..." autocomplete="off" />
  <button id="mic-btn" title="Voice recording">Mic</button>
  <div class="waveform" id="waveform"><span></span><span></span><span></span><span></span></div>
  <button id="send-btn">Send</button>
</div>
<script>
(function() {
  const chatId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36);
  const senderId = "web_" + chatId.slice(0, 8);
  const msgs = document.getElementById("messages");
  const input = document.getElementById("msg-input");
  const btn = document.getElementById("send-btn");
  let typingEl = null;

  function addMsg(text, cls) {
    const div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function showTyping() {
    if (!typingEl) {
      typingEl = addMsg("typing...", "typing");
    }
  }

  function hideTyping() {
    if (typingEl) { typingEl.remove(); typingEl = null; }
  }

  // SSE
  const es = new EventSource("/api/chat/" + chatId + "/stream");
  es.addEventListener("message", function(e) {
    hideTyping();
    const d = JSON.parse(e.data);
    if (d.text) addMsg(d.text, "bot");
    if (d.audio_url) {
      var audio = new Audio(d.audio_url);
      audio.play().catch(function(err) { console.warn("Audio playback failed:", err); });
    }
  });
  es.addEventListener("typing", function(e) { showTyping(); });
  es.onerror = function() { console.warn("SSE connection lost, retrying..."); };

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    addMsg(text, "user");
    try {
      await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({chat_id: chatId, text: text, sender_id: senderId})
      });
    } catch(err) { console.error("Send failed:", err); }
  }

  btn.addEventListener("click", send);
  input.addEventListener("keydown", function(e) {
    if (e.key === "Enter") send();
  });

  // ── Voice Recording (D-01, D-02, D-14) ──
  const micBtn = document.getElementById("mic-btn");
  const waveformEl = document.getElementById("waveform");
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  function setRecordingState(recording) {
    isRecording = recording;
    micBtn.classList.toggle("recording", recording);
    micBtn.textContent = recording ? "Stop" : "Mic";
    waveformEl.classList.toggle("active", recording);
  }

  micBtn.addEventListener("click", async function() {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/ogg;codecs=opus";
        mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
        audioChunks = [];

        mediaRecorder.ondataavailable = function(e) {
          if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async function() {
          stream.getTracks().forEach(function(t) { t.stop(); });
          const blob = new Blob(audioChunks, { type: "audio/webm" });
          const form = new FormData();
          form.append("chat_id", chatId);
          form.append("sender_id", senderId);
          form.append("audio", blob, "recording.webm");

          micBtn.disabled = true;
          try {
            const resp = await fetch("/api/chat/voice", { method: "POST", body: form });
            if (resp.ok) {
              const data = await resp.json();
              if (data.transcript) addMsg(data.transcript, "user");
            } else {
              console.error("Voice upload failed:", resp.status);
            }
          } catch(err) {
            console.error("Voice upload error:", err);
          } finally {
            micBtn.disabled = false;
          }
        };

        mediaRecorder.start();
        setRecordingState(true);
      } catch(err) {
        console.error("Microphone access denied:", err);
      }
    } else {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      setRecordingState(false);
    }
  });
})();
</script>
</body>
</html>"""


# ── WebChannel ─────────────────────────────────────────────────────────


class WebChannel(BaseChannel):
    """Web chat channel using FastAPI with SSE for real-time streaming."""

    channel_id: str = "web"

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, api_key: str | None = None, pipe_manager: Any | None = None) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._pipe_manager = pipe_manager
        self._app: FastAPI | None = None
        self._server: Any = None
        self._on_message: MessageHandler | None = None
        self._sse_manager = SSEManager()
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self, on_message: MessageHandler) -> None:
        """Create the FastAPI app, mount routes, and start uvicorn."""
        self._on_message = on_message
        self._setup_app()
        self._running = True

        logger.info("web.starting", host=self._host, port=self._port)

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self) -> None:
        """Shut down the uvicorn server."""
        self._running = False
        if self._server is not None:
            self._server.should_exit = True
            logger.info("web.stopped")

    # ── Sending ──────────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Push a message to the SSE stream for the given chat_id."""
        data: dict[str, Any] = {"text": message.text}
        if message.reply_to_message_id:
            data["reply_to"] = message.reply_to_message_id
        if message.buttons:
            data["buttons"] = [
                {"text": b.text, "callback_data": b.callback_data}
                for b in message.buttons
            ]
        if message.attachments:
            data["attachments"] = [
                {
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "url": a.file_url,
                }
                for a in message.attachments
            ]
            # Include audio_url for voice replies (D-12)
            for att in message.attachments:
                if att.mime_type == "audio/mpeg" and att.file_path:
                    audio_name = Path(att.file_path).name
                    data["audio_url"] = f"/api/audio/{audio_name}"
                    break

        await self._sse_manager.publish(message.chat_id, "message", data)
        logger.debug("web.sent", chat_id=message.chat_id)

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator via SSE."""
        await self._sse_manager.publish_typing(chat_id)

    async def _ffmpeg_transcode(self, input_path: str, output_path: str) -> bool:
        """Transcode audio to 16kHz mono WAV for Whisper. Returns True on success."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            if proc.returncode != 0:
                logger.warning(
                    "web.voice_transcode_failed",
                    returncode=proc.returncode,
                    stderr=stderr.decode()[:500] if stderr else "",
                )
                return False
            return True
        except asyncio.TimeoutError:
            logger.warning("web.voice_transcode_timeout", input_path=input_path)
            return False
        except FileNotFoundError:
            logger.error("web.ffmpeg_not_found")
            return False

    async def _process_voice_upload(
        self, chat_id: str, sender_id: str, audio: UploadFile
    ) -> JSONResponse:
        """Shared voice upload handler used by root and profile-scoped voice routes."""
        if not chat_id or not sender_id:
            return JSONResponse(
                status_code=422,
                content={"detail": "Missing required fields: chat_id, sender_id"},
            )

        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_in_path = tmp_in.name
        wav_path = tmp_in_path + ".wav"
        try:
            content = await audio.read()
            tmp_in.write(content)
            tmp_in.close()

            ok = await self._ffmpeg_transcode(tmp_in_path, wav_path)
            if not ok:
                return JSONResponse(
                    status_code=422, content={"detail": "Audio transcode failed"}
                )

            from astridr.media.whisper import WhisperTranscriber

            transcriber = WhisperTranscriber()
            result = await transcriber.execute(audio_path=wav_path)
            transcript = result.output if result.success and result.output else ""
            if not transcript:
                return JSONResponse(
                    status_code=422, content={"detail": "Transcription failed"}
                )

            incoming = IncomingMessage(
                text=transcript,
                sender_id=sender_id,
                chat_id=chat_id,
                channel_id=self.channel_id,
                timestamp=time.time(),
                reply_to_message_id=None,
                attachments=[],
                raw={"source": "voice"},
            )
            if self._on_message is not None:
                asyncio.create_task(self._on_message(incoming))

            logger.info("web.voice_transcribed", chat_id=chat_id, chars=len(transcript))
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "transcript": transcript, "chat_id": chat_id},
            )
        finally:
            Path(tmp_in_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)

    # ── Internal: app setup ──────────────────────────────────────────────

    def _setup_app(self) -> None:
        """Create the FastAPI application with routes and middleware."""
        self._app = FastAPI(title="Astridr Web Chat", docs_url=None, redoc_url=None)
        self._add_security_middleware()
        # Dashboard routes must register BEFORE catch-all /{profile_path} routes
        register_briefing_dashboard(self._app)
        self._register_routes()

    def _add_security_middleware(self) -> None:
        """Add security headers, API key auth, and rate-limiting middleware."""
        from collections import defaultdict

        app = self._app
        assert app is not None

        # Rate-limit state
        _rate_limits: dict[str, list[float]] = defaultdict(list)
        _RATE_LIMIT = 30  # requests per minute

        @app.middleware("http")
        async def rate_limit(request: Request, call_next: Any) -> Any:
            if request.url.path.startswith("/api/chat") and request.method == "POST":
                client_ip = request.client.host if request.client else "unknown"
                now = time.time()
                _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if now - t < 60]
                if len(_rate_limits[client_ip]) >= _RATE_LIMIT:
                    return JSONResponse(
                        status_code=429, content={"detail": "Rate limit exceeded"}
                    )
                _rate_limits[client_ip].append(now)
            return await call_next(request)

        if self._api_key:
            @app.middleware("http")
            async def auth_check(request: Request, call_next: Any) -> Any:
                if (
                    request.url.path.startswith("/api/")
                    and request.url.path != "/api/health"
                ):
                    auth = request.headers.get("Authorization", "")
                    if auth != f"Bearer {self._api_key}":
                        return JSONResponse(
                            status_code=401, content={"detail": "Unauthorized"}
                        )
                return await call_next(request)

        @app.middleware("http")
        async def security_headers(request: Request, call_next: Any) -> Any:
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; "
                "img-src 'self' data: blob:; "
                "media-src 'self' blob:; "
                "font-src 'self'"
            )
            return response

    def _register_routes(self) -> None:
        """Mount all API and page routes."""
        app = self._app
        assert app is not None

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """Serve the minimal HTML chat interface."""
            return HTMLResponse(content=CHAT_HTML)

        @app.post("/api/chat")
        async def post_chat(request: Request) -> JSONResponse:
            """Accept a chat message and route it through the handler."""
            body = await request.json()

            chat_id = body.get("chat_id")
            text = body.get("text")
            sender_id = body.get("sender_id")

            if not chat_id or not text or not sender_id:
                return JSONResponse(
                    status_code=422,
                    content={"detail": "Missing required fields: chat_id, text, sender_id"},
                )

            incoming = IncomingMessage(
                text=text,
                sender_id=sender_id,
                chat_id=chat_id,
                channel_id=self.channel_id,
                timestamp=time.time(),
                reply_to_message_id=None,
                attachments=[],
                raw=body,
            )

            if self._on_message is not None:
                # Fire and forget so we don't block the HTTP response
                asyncio.create_task(self._on_message(incoming))

            logger.debug("web.message_received", chat_id=chat_id, sender_id=sender_id)
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "chat_id": chat_id},
            )

        @app.post("/api/chat/voice")
        async def post_voice(
            chat_id: str = Form(...),
            sender_id: str = Form(...),
            audio: UploadFile = Form(...),
        ) -> JSONResponse:
            """Accept voice audio, transcode, transcribe, dispatch as text message."""
            return await self._process_voice_upload(chat_id, sender_id, audio)

        @app.post("/{profile_path}/api/chat/voice")
        async def post_voice_profile(
            profile_path: str,
            chat_id: str = Form(...),
            sender_id: str = Form(...),
            audio: UploadFile = Form(...),
        ) -> JSONResponse:
            """Accept voice audio scoped to a profile path."""
            scoped_chat_id = f"/{profile_path}:{chat_id}"
            return await self._process_voice_upload(scoped_chat_id, sender_id, audio)

        @app.get("/api/audio/{filename}")
        async def get_audio(filename: str) -> Any:
            """Serve a TTS audio file by filename with path traversal protection."""
            safe_name = Path(filename).name
            audio_path = Path.home() / ".astridr" / "media" / "tts" / safe_name
            if not audio_path.exists() or not audio_path.is_file():
                return JSONResponse(status_code=404, content={"detail": "Audio not found"})
            return FileResponse(str(audio_path), media_type="audio/mpeg")

        @app.get("/{profile_path}/api/audio/{filename}")
        async def get_audio_profile(profile_path: str, filename: str) -> Any:
            """Serve a TTS audio file scoped to a profile path."""
            safe_name = Path(filename).name
            audio_path = Path.home() / ".astridr" / "media" / "tts" / safe_name
            if not audio_path.exists() or not audio_path.is_file():
                return JSONResponse(status_code=404, content={"detail": "Audio not found"})
            return FileResponse(str(audio_path), media_type="audio/mpeg")

        @app.get("/api/chat/{chat_id}/stream")
        async def sse_stream(chat_id: str) -> StreamingResponse:
            """SSE endpoint — yields events as ``data: {json}\\n\\n``."""
            queue = self._sse_manager.subscribe(chat_id)

            async def event_generator() -> Any:
                try:
                    while True:
                        payload = await queue.get()
                        event_name = payload.get("event", "message")
                        data = json.dumps(payload.get("data", {}))
                        yield f"event: {event_name}\ndata: {data}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    self._sse_manager.unsubscribe(chat_id, queue)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Pipe management endpoints ───────────────────────────────────────

        @app.get("/api/pipes")
        async def list_pipes() -> JSONResponse:
            if self._pipe_manager is None:
                return JSONResponse(status_code=501, content={"detail": "Pipes not configured"})
            pipes = [
                {
                    "name": p.name,
                    "schedule": p.schedule,
                    "persona": p.persona,
                    "profile": p.profile,
                    "channel": p.channel,
                    "enabled": p.enabled,
                    "tags": p.tags,
                }
                for p in self._pipe_manager.pipes.values()
            ]
            return JSONResponse(content={"pipes": pipes})

        @app.get("/api/pipes/{name}")
        async def get_pipe(name: str) -> JSONResponse:
            if self._pipe_manager is None:
                return JSONResponse(status_code=501, content={"detail": "Pipes not configured"})
            pipe = self._pipe_manager.pipes.get(name)
            if pipe is None:
                return JSONResponse(status_code=404, content={"detail": f"Pipe '{name}' not found"})
            return JSONResponse(content={
                "name": pipe.name,
                "schedule": pipe.schedule,
                "persona": pipe.persona,
                "profile": pipe.profile,
                "channel": pipe.channel,
                "chat_id": pipe.chat_id,
                "tools": pipe.tools,
                "timeout_seconds": pipe.timeout_seconds,
                "enabled": pipe.enabled,
                "tags": pipe.tags,
            })

        @app.post("/api/pipes/{name}/run")
        async def run_pipe(name: str) -> JSONResponse:
            if self._pipe_manager is None:
                return JSONResponse(status_code=501, content={"detail": "Pipes not configured"})
            if name not in self._pipe_manager.pipes:
                return JSONResponse(status_code=404, content={"detail": f"Pipe '{name}' not found"})
            task = asyncio.create_task(self._pipe_manager.execute(name, trigger="manual"))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return JSONResponse(content={"status": "triggered", "pipe": name})

        @app.get("/api/pipes/{name}/executions")
        async def pipe_executions(name: str) -> JSONResponse:
            if self._pipe_manager is None:
                return JSONResponse(status_code=501, content={"detail": "Pipes not configured"})
            if name not in self._pipe_manager.pipes:
                return JSONResponse(status_code=404, content={"detail": f"Pipe '{name}' not found"})
            jobs = await self._pipe_manager.get_executions(name)
            return JSONResponse(content={
                "pipe": name,
                "executions": [
                    {
                        "id": j.id,
                        "status": j.status,
                        "trigger": j.trigger,
                        "created_at": j.created_at,
                        "started_at": j.started_at,
                        "completed_at": j.completed_at,
                        "error": j.error,
                    }
                    for j in jobs
                ],
            })

        # ── Profile-scoped routes ────────────────────────────────────────────

        @app.post("/{profile_path}/api/chat")
        async def post_chat_profile(profile_path: str, request: Request) -> JSONResponse:
            """Accept a chat message scoped to a profile path."""
            body = await request.json()

            chat_id = body.get("chat_id")
            text = body.get("text")
            sender_id = body.get("sender_id")

            if not chat_id or not text or not sender_id:
                return JSONResponse(
                    status_code=422,
                    content={"detail": "Missing required fields: chat_id, text, sender_id"},
                )

            incoming = IncomingMessage(
                text=text,
                sender_id=sender_id,
                chat_id=f"/{profile_path}:{chat_id}",
                channel_id=self.channel_id,
                timestamp=time.time(),
                reply_to_message_id=None,
                attachments=[],
                raw=body,
            )

            if self._on_message is not None:
                asyncio.create_task(self._on_message(incoming))

            logger.debug(
                "web.message_received",
                chat_id=chat_id,
                sender_id=sender_id,
                profile_path=profile_path,
            )
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "chat_id": chat_id},
            )

        @app.get("/{profile_path}/api/chat/{chat_id}/stream")
        async def sse_stream_profile(profile_path: str, chat_id: str) -> StreamingResponse:
            """SSE endpoint for a profile-scoped chat."""
            scoped_chat_id = f"/{profile_path}:{chat_id}"
            queue = self._sse_manager.subscribe(scoped_chat_id)

            async def event_generator() -> Any:
                try:
                    while True:
                        payload = await queue.get()
                        event_name = payload.get("event", "message")
                        data = json.dumps(payload.get("data", {}))
                        yield f"event: {event_name}\ndata: {data}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    self._sse_manager.unsubscribe(scoped_chat_id, queue)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.get("/{profile_path}", response_class=HTMLResponse)
        async def index_profile(profile_path: str) -> HTMLResponse:
            """Serve the chat interface scoped to a profile path."""
            scoped_html = CHAT_HTML.replace(
                '"/api/chat"',
                f'"/{profile_path}/api/chat"',
            ).replace(
                '"/api/chat/" + chatId + "/stream"',
                f'"/{profile_path}/api/chat/" + chatId + "/stream"',
            ).replace(
                "<h1>Astridr Chat</h1>",
                f"<h1>Astridr Chat \u2014 {profile_path.title()}</h1>",
            ).replace(
                '"/api/chat/voice"',
                f'"/{profile_path}/api/chat/voice"',
            ).replace(
                '"/api/audio/',
                f'"/{profile_path}/api/audio/',
            )
            return HTMLResponse(content=scoped_html)

        @app.get("/api/health")
        async def health() -> JSONResponse:
            """Health check endpoint."""
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "channel": "web"},
            )
