"""Telegram channel — async long-polling via python-telegram-bot.

Supports message splitting for the 4096-char limit, typing indicators,
file/photo/document handling, inline keyboard buttons, Markdown parse mode,
and allowed-user-ID filtering for security.
"""

from __future__ import annotations

from typing import Any

import structlog

from astridr.channels.base import (
    Attachment,
    BaseChannel,
    IncomingMessage,
    InlineButton,
    MessageHandler,
    OutgoingMessage,
)

logger = structlog.get_logger()

# Telegram hard limit for a single message
TELEGRAM_MAX_LENGTH = 4096


class TelegramChannel(BaseChannel):
    """Telegram channel implementation using python-telegram-bot (async)."""

    channel_id: str = "telegram"

    def __init__(self, token: str, allowed_user_ids: list[int] | None = None) -> None:
        self._token = token
        self._allowed_user_ids: set[int] = set(allowed_user_ids or [])
        self._application: Any = None
        self._on_message: MessageHandler | None = None
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self, on_message: MessageHandler) -> None:
        """Build the Telegram Application, register handlers, and start polling."""
        from telegram import Update
        from telegram.ext import (
            ApplicationBuilder,
            MessageHandler as TGMessageHandler,
            filters,
        )

        self._on_message = on_message

        self._application = (
            ApplicationBuilder()
            .token(self._token)
            .build()
        )

        self._application.add_handler(
            TGMessageHandler(
                filters.TEXT | filters.PHOTO | filters.Document.ALL
                | filters.VOICE | filters.AUDIO,
                self._handle_update,
            )
        )

        self._running = True
        logger.info("telegram.starting")

        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

        logger.info("telegram.started")

    async def stop(self) -> None:
        """Gracefully shut down the Telegram polling loop."""
        self._running = False
        if self._application is not None:
            if self._application.updater and self._application.updater.running:
                await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
            logger.info("telegram.stopped")

    # ── Sending ──────────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Send a message to Telegram, splitting if necessary and adding buttons."""
        if self._application is None:
            raise RuntimeError("TelegramChannel has not been started")

        bot = self._application.bot
        chat_id = int(message.chat_id)

        chunks = split_message(message.text)

        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            kwargs: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
            }

            if message.parse_mode:
                kwargs["parse_mode"] = message.parse_mode

            if message.reply_to_message_id:
                kwargs["reply_to_message_id"] = int(message.reply_to_message_id)

            # Attach inline keyboard only to the last chunk
            if is_last and message.buttons:
                keyboard = self._build_inline_keyboard(message.buttons)
                kwargs["reply_markup"] = keyboard

            await bot.send_message(**kwargs)

        # Send attachments
        for attachment in message.attachments:
            await self._send_attachment(bot, chat_id, attachment)

        logger.debug("telegram.sent", chat_id=chat_id, chunks=len(chunks))

    async def send_typing(self, chat_id: str) -> None:
        """Send a 'typing' chat action to the user."""
        if self._application is None:
            return

        from telegram.constants import ChatAction

        await self._application.bot.send_chat_action(
            chat_id=int(chat_id),
            action=ChatAction.TYPING,
        )

    # ── Internal: update handling ────────────────────────────────────────

    async def _handle_update(self, update: Any, context: Any) -> None:  # noqa: ARG002
        """Process an incoming Telegram update."""
        if self._on_message is None:
            return

        message = update.effective_message
        if message is None:
            return

        user = update.effective_user
        if user is None:
            return

        # Security: filter by allowed user IDs
        if self._allowed_user_ids and user.id not in self._allowed_user_ids:
            logger.warning("telegram.unauthorized_user", user_id=user.id)
            return

        text = message.text or message.caption or ""
        attachments = self._extract_attachments(message)

        # Transcribe voice/audio messages that have no text
        if not text and (message.voice or message.audio):
            text = await self._transcribe_voice(message)

        incoming = IncomingMessage(
            text=text,
            sender_id=str(user.id),
            chat_id=str(message.chat_id),
            channel_id=self.channel_id,
            timestamp=float(message.date.timestamp()) if message.date else 0.0,
            reply_to_message_id=str(message.message_id),
            attachments=attachments,
            raw=update.to_dict() if hasattr(update, "to_dict") else {},
        )

        await self._on_message(incoming)

    async def _transcribe_voice(self, message: Any) -> str:
        """Download a voice/audio message from Telegram and transcribe it."""
        try:
            from astridr.media.whisper import WhisperTranscriber

            voice_obj = message.voice or message.audio
            if voice_obj is None:
                return ""

            tg_file = await self._application.bot.get_file(voice_obj.file_id)

            import tempfile
            from pathlib import Path

            suffix = ".ogg" if message.voice else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name

            await tg_file.download_to_drive(tmp_path)

            transcriber = WhisperTranscriber()
            result = await transcriber.execute(audio_path=tmp_path)

            Path(tmp_path).unlink(missing_ok=True)

            if result.success and result.output:
                logger.info("telegram.voice_transcribed", chars=len(result.output))
                return result.output

            logger.warning("telegram.voice_transcription_failed", error=result.error)
            return "[Voice message \u2014 transcription unavailable]"
        except Exception as exc:
            logger.warning("telegram.voice_transcription_error", error=str(exc))
            return "[Voice message \u2014 transcription unavailable]"

    # ── Internal: attachment helpers ───────────────────────────────────────

    @staticmethod
    def _extract_attachments(message: Any) -> list[Attachment]:
        """Extract attachments from a Telegram message."""
        attachments: list[Attachment] = []

        if message.photo:
            # Telegram sends multiple sizes; take the largest
            photo = message.photo[-1]
            attachments.append(
                Attachment(
                    file_url=f"tg://file/{photo.file_id}",
                    mime_type="image/jpeg",
                    filename="photo.jpg",
                )
            )

        if message.document:
            doc = message.document
            attachments.append(
                Attachment(
                    file_url=f"tg://file/{doc.file_id}",
                    mime_type=doc.mime_type or "application/octet-stream",
                    filename=doc.file_name or "document",
                )
            )

        if message.audio:
            audio = message.audio
            attachments.append(
                Attachment(
                    file_url=f"tg://file/{audio.file_id}",
                    mime_type=audio.mime_type or "audio/mpeg",
                    filename=audio.file_name or "audio.mp3",
                )
            )

        if message.voice:
            voice = message.voice
            attachments.append(
                Attachment(
                    file_url=f"tg://file/{voice.file_id}",
                    mime_type=voice.mime_type or "audio/ogg",
                    filename="voice.ogg",
                )
            )

        return attachments

    async def _send_attachment(self, bot: Any, chat_id: int, attachment: Attachment) -> None:
        """Send a single attachment to Telegram.

        Audio attachments (``audio/*``) are sent as voice notes via
        ``send_voice``; everything else goes through ``send_document``.
        """
        is_audio = attachment.mime_type.startswith("audio/")

        if attachment.file_path:
            with open(attachment.file_path, "rb") as f:
                if is_audio:
                    await bot.send_voice(chat_id=chat_id, voice=f)
                else:
                    await bot.send_document(chat_id=chat_id, document=f)
        elif attachment.file_bytes:
            from io import BytesIO

            bio = BytesIO(attachment.file_bytes)
            bio.name = attachment.filename
            if is_audio:
                await bot.send_voice(chat_id=chat_id, voice=bio)
            else:
                await bot.send_document(chat_id=chat_id, document=bio)

    @staticmethod
    def _build_inline_keyboard(buttons: list[InlineButton]) -> Any:
        """Build a Telegram InlineKeyboardMarkup from our InlineButton model."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = [
            [InlineKeyboardButton(text=btn.text, callback_data=btn.callback_data)]
            for btn in buttons
        ]
        return InlineKeyboardMarkup(keyboard)


# ── Utility: message splitting ───────────────────────────────────────


def split_message(text: str, max_length: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    """Split a long message into chunks respecting the Telegram limit.

    Tries to split on paragraph boundaries, then sentence boundaries,
    then word boundaries, and finally hard-cuts as a last resort.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try to find a good split point
        split_at = _find_split_point(remaining, max_length)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    return chunks


def _find_split_point(text: str, max_length: int) -> int:
    """Find the best position to split text at, up to max_length."""
    window = text[:max_length]

    # 1) Try paragraph break (double newline)
    idx = window.rfind("\n\n")
    if idx > max_length // 4:
        return idx + 2

    # 2) Try single newline
    idx = window.rfind("\n")
    if idx > max_length // 4:
        return idx + 1

    # 3) Try sentence end (. ! ?)
    for punct in (". ", "! ", "? "):
        idx = window.rfind(punct)
        if idx > max_length // 4:
            return idx + len(punct)

    # 4) Try word boundary (space)
    idx = window.rfind(" ")
    if idx > max_length // 4:
        return idx + 1

    # 5) Hard cut
    return max_length
