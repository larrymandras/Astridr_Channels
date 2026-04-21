"""Always-on voice channel with wake word detection and TTS responses."""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from astridr.channels.base import (
    BaseChannel,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
)
from astridr.channels.vad import VADDetector, create_vad_detector
from astridr.channels.wake_word import WakeWordDetector, create_wake_word_detector
from astridr.core.secrets import get_snapshot
from astridr.media.tts import STREAM_SAMPLE_RATE

log = structlog.get_logger()

# Default sample rate for microphone capture
_DEFAULT_SAMPLE_RATE = 16000

# Audio is 16-bit signed PCM
_BYTES_PER_SAMPLE = 2

# Read audio in 512-sample chunks (~32 ms at 16 kHz)
_CHUNK_SAMPLES = 512


class ConversationState(Enum):
    """State machine states for multi-turn conversation mode (D-10)."""

    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    PROCESSING = "processing"


class VoiceChannel(BaseChannel):
    """Always-on voice channel with wake word detection.

    Listens to the microphone, detects a wake word, records the user's
    utterance until silence is detected, transcribes it via Whisper, and
    delivers the text to the message handler.  Outgoing responses are
    converted to speech via the configured TTS engine and played back.
    """

    channel_id: str = "voice"

    def __init__(
        self,
        wake_word: str = "astridr",
        sensitivity: float = 0.5,
        silence_threshold: float = 0.5,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        tts_tool: Any | None = None,
        transcriber: Any | None = None,
        wake_word_detector: WakeWordDetector | None = None,
        vad_detector: VADDetector | None = None,
        input_device: int | str | None = None,
        end_phrases: list[str] | None = None,
    ) -> None:
        self._wake_word = wake_word
        self._sensitivity = sensitivity
        self._silence_threshold = silence_threshold
        self._sample_rate = sample_rate
        self._running = False
        self._on_message: MessageHandler | None = None
        self._tts_tool = tts_tool
        self._transcriber = transcriber
        self._listen_task: asyncio.Task[None] | None = None
        self._input_device = input_device
        self._detector = wake_word_detector or create_wake_word_detector(
            wake_word=wake_word,
            sensitivity=sensitivity,
        )
        self._vad = vad_detector or create_vad_detector()
        self._default_voice_id = get_snapshot().get("ELEVENLABS_VOICE_ID") or ""
        # Session state attributes (conversation mode — D-10, D-11)
        self._state: ConversationState = ConversationState.IDLE
        self._session_id: str | None = None
        self._turn_count: int = 0
        self._session_start: float | None = None
        self._tts_complete: asyncio.Event = asyncio.Event()
        self._end_phrases: list[str] = end_phrases if end_phrases is not None else [
            "goodbye", "that's all", "thanks", "stop",
        ]

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, to: ConversationState) -> None:
        """Transition to a new state and emit a structured log event (D-11)."""
        from_state = self._state
        self._state = to
        elapsed = (
            round(time.perf_counter() - self._session_start, 2)
            if self._session_start is not None else 0.0
        )
        log.info(
            "voice.state_change",
            from_state=from_state.value,
            to_state=to.value,
            session_id=self._session_id,
            turn_count=self._turn_count,
            elapsed_seconds=elapsed,
        )

    async def _cleanup_session(self, reason: str) -> None:
        """Reset all session state and transition to IDLE (CONV-04)."""
        log.info(
            "voice.session_end",
            reason=reason,
            session_id=self._session_id,
            turn_count=self._turn_count,
        )
        self._session_id = None
        self._session_start = None
        self._turn_count = 0
        self._tts_complete.set()
        self._transition(ConversationState.IDLE)

    def _is_end_phrase(self, text: str) -> bool:
        """Return True if text (stripped, lowercased) exactly matches an end phrase (D-06)."""
        return text.lower().strip() in self._end_phrases

    async def _play_confirmation_tone(self) -> None:
        """Play a 440 Hz sine wave confirmation tone via sounddevice (D-08, D-09)."""
        import numpy as np
        import sounddevice as sd

        sample_rate = 22050
        duration = 0.1
        freq = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: (sd.play(tone, sample_rate), sd.wait()),
            )
        except Exception as exc:
            log.debug("voice.tone_failed", error=str(exc))

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    async def start(self, on_message: MessageHandler) -> None:
        """Start listening for the wake word on the default microphone."""
        self._on_message = on_message
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        log.info(
            "voice.started",
            wake_word=self._wake_word,
            sample_rate=self._sample_rate,
        )

    async def send(self, message: OutgoingMessage) -> None:
        """Convert response text to speech and play it back.

        Uses streaming TTS (per D-03) when stream_to_queue is available on the
        TTS tool. Falls back to batch generation on streaming failure (per D-06)
        or when streaming is not available (per D-08/STTS-04).

        Always sets _tts_complete in finally block to prevent state machine deadlock
        (Pitfall 1 — D-01/D-03).
        """
        try:
            if not message.text:
                return

            if self._tts_tool is None:
                log.debug("voice.no_tts", text=message.text[:80])
                return

            # Read voice identity from metadata (populated by router)
            meta = message.metadata if hasattr(message, "metadata") else {}
            voice_id = meta.get("voice_id", self._default_voice_id)
            stability = meta.get("stability", 0.5)
            similarity_boost = meta.get("similarity_boost", 0.75)

            # Try streaming path first (per D-01, D-03)
            if hasattr(self._tts_tool, "stream_to_queue"):
                try:
                    await self._stream_and_play(
                        message.text, voice_id, stability, similarity_boost,
                    )
                    return
                except Exception as exc:
                    log.warning(
                        "voice.streaming_failed",
                        error=str(exc),
                        chars=len(message.text),
                        method="streaming",
                        model="eleven_flash_v2_5",
                    )
                    log.info(
                        "voice.fallback_to_batch",
                        chars=len(message.text),
                    )

            # Batch fallback (per D-06, D-08)
            try:
                result = await self._tts_tool.execute(
                    text=message.text,
                    voice_id=voice_id,
                    stability=stability,
                    similarity_boost=similarity_boost,
                )
                if result.success and result.data.get("path"):
                    await self._play_audio(Path(result.data["path"]))
                else:
                    log.warning("voice.tts_failed", error=result.error)
            except Exception as exc:
                log.error("voice.send_failed", error=str(exc))
        finally:
            self._tts_complete.set()

    async def _stream_and_play(
        self,
        text: str,
        voice_id: str,
        stability: float,
        similarity_boost: float,
    ) -> None:
        """Stream TTS audio from ElevenLabs and play chunks as they arrive.

        Producer: tts_tool.stream_to_queue() pushes PCM bytes into queue.
        Consumer: Drains queue, converts to numpy int16, writes to OutputStream.
        Both run concurrently via asyncio.gather (per D-03).

        TTFB is measured from method entry to first stream.write() call
        and logged as voice.ttfb_ms (per research Pitfall 6).
        """
        import numpy as np
        import sounddevice as sd

        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=20)
        start_time = time.perf_counter()
        first_write_logged = False

        async def consumer() -> None:
            nonlocal first_write_logged
            with sd.OutputStream(
                samplerate=STREAM_SAMPLE_RATE, channels=1, dtype="int16",
            ) as stream:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    await asyncio.to_thread(stream.write, pcm)
                    if not first_write_logged:
                        ttfb_ms = round((time.perf_counter() - start_time) * 1000)
                        log.info("voice.ttfb_ms", ttfb_ms=ttfb_ms)
                        first_write_logged = True

        await asyncio.gather(
            self._tts_tool.stream_to_queue(
                text, voice_id, queue,
                stability=stability,
                similarity_boost=similarity_boost,
            ),
            consumer(),
        )

    async def send_typing(self, chat_id: str) -> None:
        """Play a brief thinking tone, or do nothing."""
        # Voice channel has no visual typing indicator; this is a no-op.
        pass

    async def stop(self) -> None:
        """Stop listening and clean up resources."""
        self._running = False
        if self._state != ConversationState.IDLE:
            await self._cleanup_session("stopped")
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        self._detector.cleanup()
        self._vad.cleanup()
        self._on_message = None
        log.info("voice.stopped")

    # ------------------------------------------------------------------
    # Listening loop and state machine handlers
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Main loop: match/case state machine dispatching to state handlers."""
        try:
            while self._running:
                match self._state:
                    case ConversationState.IDLE:
                        await self._handle_idle()
                    case ConversationState.WAKE_DETECTED:
                        await self._handle_wake_detected()
                    case ConversationState.LISTENING:
                        await self._handle_listening()
                    case ConversationState.PROCESSING:
                        await self._handle_processing()
        except asyncio.CancelledError:
            await self._cleanup_session("cancelled")
            return
        except Exception as exc:
            log.error("voice.listen_loop_error", error=str(exc))
            await self._cleanup_session("error")

    async def _handle_idle(self) -> None:
        """IDLE: wait for wake word, then transition to WAKE_DETECTED."""
        detected = await self._wait_for_wake_word()
        if detected and self._running:
            self._transition(ConversationState.WAKE_DETECTED)

    async def _handle_wake_detected(self) -> None:
        """WAKE_DETECTED: initialize session and play confirmation tone, then transition to LISTENING."""
        self._session_id = uuid.uuid4().hex[:12]
        self._session_start = time.perf_counter()
        self._turn_count = 0
        self._tts_complete.set()
        log.info("voice.wake_word_detected", wake_word=self._wake_word, session_id=self._session_id)
        await self._play_confirmation_tone()
        self._transition(ConversationState.LISTENING)

    async def _handle_listening(self) -> None:
        """LISTENING: record audio, transcribe, check end phrases, and dispatch message."""
        # D-04: 5-minute max duration cap
        if self._session_start is not None and time.perf_counter() - self._session_start > 300.0:
            await self._cleanup_session("max_duration")
            return

        # D-07: 30-second silence timeout
        try:
            audio_bytes = await asyncio.wait_for(
                self._record_until_silence(),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            log.info("voice.session_timeout", session_id=self._session_id)
            await self._cleanup_session("timeout")
            return

        if not audio_bytes:
            return  # Stay in LISTENING

        text = await self._transcribe(audio_bytes)
        if not text:
            return  # Stay in LISTENING

        # D-06: Check for end phrases
        if self._is_end_phrase(text):
            log.info("voice.end_phrase", text=text, session_id=self._session_id)
            await self._cleanup_session("end_phrase")
            return

        # Wake word filtering only on first turn (turn_count == 0)
        if self._turn_count == 0:
            text_lower = text.lower().strip()
            wake = self._wake_word.lower()
            if text_lower.startswith(wake):
                text = text[len(wake):].strip()
                if not text:
                    text = text_lower  # Wake word only — treat as greeting
            elif wake not in text_lower:
                log.debug("voice.no_wake_word", transcript=text[:50])
                await self._cleanup_session("no_wake_word")
                return

        # Dispatch to handler
        if self._on_message is not None:
            msg = IncomingMessage(
                text=text,
                sender_id="voice_user",
                chat_id="voice",
                channel_id=self.channel_id,
                timestamp=time.time(),
                raw={
                    "source": "microphone",
                    "sample_rate": self._sample_rate,
                    "session_id": self._session_id,
                    "turn_count": self._turn_count,
                },
            )
            self._transition(ConversationState.PROCESSING)
            await self._on_message(msg)

    async def _handle_processing(self) -> None:
        """PROCESSING: wait for TTS to complete, then transition back to LISTENING."""
        # D-01/D-03: Wait for TTS to complete before re-listening (mic muted during TTS)
        await self._tts_complete.wait()
        self._tts_complete.clear()
        self._turn_count += 1
        self._transition(ConversationState.LISTENING)

    async def _wait_for_wake_word(self) -> bool:
        """Listen for the wake word using the configured detector.

        Continuously reads mic chunks and passes them to the
        wake word detector until a detection occurs.
        """
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        device = self._input_device

        while self._running:
            try:
                raw = await loop.run_in_executor(
                    None,
                    lambda: sd.rec(
                        _CHUNK_SAMPLES,
                        samplerate=self._sample_rate,
                        channels=1,
                        dtype="int16",
                        blocking=True,
                        device=device,
                    ).tobytes(),
                )
                if self._detector.process_audio(raw):
                    return True  # Wake word detected
            except Exception as exc:
                log.warning("voice.mic_read_error", error=str(exc))
                await asyncio.sleep(1)
        return False

    async def _record_until_silence(self) -> bytes:
        """Record audio from the microphone until silence is detected.

        Returns raw PCM bytes (16-bit signed, mono).
        """
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        recorded: list[bytes] = []
        silence_start: float | None = None
        device = self._input_device

        while self._running:
            raw = await loop.run_in_executor(
                None,
                lambda: sd.rec(
                    _CHUNK_SAMPLES,
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="int16",
                    blocking=True,
                    device=device,
                ).tobytes(),
            )
            recorded.append(raw)

            if not self._vad.is_speech(raw):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= self._silence_threshold:
                    break  # Enough silence — done recording
            else:
                silence_start = None

            # Safety cap: max 30 seconds of recording
            total_samples = sum(len(c) for c in recorded) // _BYTES_PER_SAMPLE
            if total_samples / self._sample_rate > 30:
                break

        self._vad.reset()
        return b"".join(recorded)

    async def _transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe recorded audio bytes to text."""
        if self._transcriber is None:
            log.warning("voice.no_transcriber")
            return ""

        # Save audio to a temporary file for the transcriber
        tmp_dir = Path.home() / ".astridr" / "media" / "voice_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"recording_{uuid.uuid4().hex[:8]}.wav"

        try:
            self._write_wav(tmp_path, audio_bytes)
            result = await self._transcriber.execute(audio_path=str(tmp_path))
            if result.success:
                return result.output
            log.warning("voice.transcription_failed", error=result.error)
            return ""
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    async def _play_audio(self, path: Path) -> None:
        """Play an audio file through the default output device."""
        import sounddevice as sd
        import soundfile as sf

        loop = asyncio.get_event_loop()
        try:
            data, samplerate = await loop.run_in_executor(
                None, sf.read, str(path),
            )
            await loop.run_in_executor(
                None,
                lambda: (sd.play(data, samplerate), sd.wait()),
            )
        except Exception as exc:
            log.warning("voice.playback_error", error=str(exc))

    def _write_wav(self, path: Path, pcm_data: bytes) -> None:
        """Write raw PCM data as a WAV file.

        Args:
            path: Output file path.
            pcm_data: Raw 16-bit signed mono PCM data.
        """
        import wave

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(_BYTES_PER_SAMPLE)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_data)
