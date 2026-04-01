"""Slack channel — async Socket Mode via slack-bolt.

Listens for DMs and mentions, handles threads, files,
emoji reactions, and Block Kit formatting.
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


class SlackChannel(BaseChannel):
    """Slack channel implementation using slack-bolt in async Socket Mode."""

    channel_id: str = "slack"

    def __init__(self, bot_token: str, app_token: str) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._app: Any = None
        self._socket_handler: Any = None
        self._on_message: MessageHandler | None = None
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self, on_message: MessageHandler) -> None:
        """Register Slack event handlers and start async Socket Mode."""
        from slack_bolt.adapter.socket_mode.async_handler import (
            AsyncSocketModeHandler,
        )
        from slack_bolt.async_app import AsyncApp

        self._on_message = on_message

        self._app = AsyncApp(token=self._bot_token)
        self._register_handlers()

        self._socket_handler = AsyncSocketModeHandler(self._app, self._app_token)
        self._running = True

        logger.info("slack.starting")
        # Use connect_async() instead of start_async() — the latter blocks
        # forever as the main event loop; connect_async() establishes the
        # WebSocket connection and returns control to our own event loop.
        await self._socket_handler.connect_async()
        logger.info("slack.started")

    async def stop(self) -> None:
        """Gracefully shut down Socket Mode."""
        self._running = False
        if self._socket_handler is not None:
            await self._socket_handler.close_async()
            logger.info("slack.stopped")

    # ── Sending ──────────────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Post a message to Slack with optional Block Kit formatting and buttons."""
        if self._app is None:
            raise RuntimeError("SlackChannel has not been started")

        kwargs: dict[str, Any] = {
            "channel": message.chat_id,
            "text": message.text,
        }

        if message.reply_to_message_id:
            kwargs["thread_ts"] = message.reply_to_message_id

        # Build Block Kit blocks for rich formatting
        blocks = self._build_blocks(message)
        if blocks:
            kwargs["blocks"] = blocks

        await self._app.client.chat_postMessage(**kwargs)

        # Upload attachments if present (best-effort — don't block text delivery)
        for attachment in message.attachments:
            try:
                await self._upload_attachment(message.chat_id, attachment, message.reply_to_message_id)
            except Exception:
                logger.warning(
                    "slack.attachment_upload_failed",
                    chat_id=message.chat_id,
                    filename=attachment.filename,
                )

        logger.debug(
            "slack.sent",
            chat_id=message.chat_id,
            thread_ts=message.reply_to_message_id,
        )

    async def send_typing(self, chat_id: str) -> None:
        """Show a typing indicator by posting an ephemeral typing status.

        Slack does not have a native 'typing' API for bots in the same way
        as Telegram. We approximate by briefly posting a status or using
        a workaround via the Web API.
        """
        # Slack doesn't have a direct typing indicator for bots.
        # We log the intent; callers can use this as a hook for a "thinking" message.
        logger.debug("slack.typing", chat_id=chat_id)

    # ── Internal: handler registration ───────────────────────────────────

    def _register_handlers(self) -> None:
        """Wire up Slack event listeners on the bolt app."""
        if self._app is None:
            return

        @self._app.event("message")
        async def handle_message(event: dict[str, Any], say: Any) -> None:  # noqa: ARG001
            await self._process_message_event(event)

        @self._app.event("app_mention")
        async def handle_mention(event: dict[str, Any], say: Any) -> None:  # noqa: ARG001
            await self._process_message_event(event)

    async def _process_message_event(self, event: dict[str, Any]) -> None:
        """Convert a raw Slack event dict into an IncomingMessage."""
        if self._on_message is None:
            return

        # Ignore bot messages to avoid loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        text = event.get("text", "")
        sender_id = event.get("user", "")
        channel = event.get("channel", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        # Extract attachments (files shared in the message)
        attachments = self._extract_attachments(event)

        # Acknowledge with eyes emoji
        if self._app is not None:
            try:
                await self._app.client.reactions_add(
                    channel=channel,
                    name="eyes",
                    timestamp=ts,
                )
            except Exception:
                logger.debug("slack.reaction_failed", channel=channel, ts=ts)

        incoming = IncomingMessage(
            text=text,
            sender_id=sender_id,
            chat_id=channel,
            channel_id=self.channel_id,
            timestamp=float(ts) if ts else 0.0,
            reply_to_message_id=thread_ts or ts,
            attachments=attachments,
            raw=event,
        )

        await self._on_message(incoming)

    # ── Internal: attachments ────────────────────────────────────────────

    def _extract_attachments(self, event: dict[str, Any]) -> list[Attachment]:
        """Pull file metadata from a Slack message event."""
        attachments: list[Attachment] = []
        for file_info in event.get("files", []):
            attachments.append(
                Attachment(
                    file_url=file_info.get("url_private_download") or file_info.get("url_private"),
                    mime_type=file_info.get("mimetype", "application/octet-stream"),
                    filename=file_info.get("name", "file"),
                )
            )
        return attachments

    async def _upload_attachment(
        self,
        channel: str,
        attachment: Attachment,
        thread_ts: str | None,
    ) -> None:
        """Upload a single attachment to a Slack channel."""
        if self._app is None:
            return

        kwargs: dict[str, Any] = {
            "channels": channel,
            "filename": attachment.filename,
            "title": attachment.filename,
        }
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        if attachment.file_bytes:
            kwargs["content"] = attachment.file_bytes
        elif attachment.file_path:
            kwargs["file"] = attachment.file_path

        if "content" in kwargs or "file" in kwargs:
            await self._app.client.files_upload_v2(**kwargs)

    # ── Internal: Block Kit ──────────────────────────────────────────────

    def _build_blocks(self, message: OutgoingMessage) -> list[dict[str, Any]]:
        """Build Slack Block Kit blocks from an OutgoingMessage."""
        blocks: list[dict[str, Any]] = []

        # Main text as a section block with mrkdwn
        if message.text:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message.text,
                    },
                }
            )

        # Inline buttons as an actions block
        if message.buttons:
            elements = []
            for btn in message.buttons:
                elements.append(
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": btn.text},
                        "action_id": btn.callback_data,
                        "value": btn.callback_data,
                    }
                )
            blocks.append({"type": "actions", "elements": elements})

        return blocks
