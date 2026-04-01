"""Email channel — async IMAP polling + SMTP sending.

Uses aioimaplib for non-blocking IMAP (with IDLE push fallback to polling)
and aiosmtplib for async email dispatch.  Handles HTML-to-text conversion,
attachment extraction, and reply-in-thread via References/In-Reply-To headers.
"""

from __future__ import annotations

import asyncio
import email as email_lib
import email.encoders
import email.mime.base
import email.mime.multipart
import email.mime.text
import email.utils
import html
import re
import tempfile
import time
from email.message import Message as EmailMessage
from pathlib import Path
from typing import Any

import structlog

from astridr.channels.base import (
    Attachment,
    BaseChannel,
    IncomingMessage,
    MessageHandler,
    OutgoingMessage,
)

logger = structlog.get_logger()

# Headers that indicate newsletters / promotions
_NEWSLETTER_HEADERS = (
    "list-unsubscribe",
    "list-id",
    "x-mailer",
    "x-campaign",
    "x-mailchimp-id",
    "precedence",
)


class EmailChannel(BaseChannel):
    """Email channel: IMAP for receiving, SMTP for sending."""

    channel_id: str = "email"

    def __init__(
        self,
        imap_host: str,
        smtp_host: str,
        username: str,
        password: str,
        *,
        poll_interval: int = 30,
        imap_port: int = 993,
        smtp_port: int = 587,
        filter_newsletters: bool = True,
        avatar_path: Path | None = None,
        personal_avatar_path: Path | None = None,
    ) -> None:
        self._imap_host = imap_host
        self._smtp_host = smtp_host
        self._username = username
        self._password = password
        self._poll_interval = poll_interval
        self._imap_port = imap_port
        self._smtp_port = smtp_port
        self._filter_newsletters = filter_newsletters
        self._avatar_path = avatar_path
        self._personal_avatar_path = personal_avatar_path

        self._imap_client: Any = None
        self._on_message: MessageHandler | None = None
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None

        # Track message-IDs for threading
        self._thread_map: dict[str, dict[str, str]] = {}
        # key = chat_id (sender email), value = {message_id, references}

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self, on_message: MessageHandler) -> None:
        """Connect to IMAP and begin the polling / IDLE loop."""
        import aioimaplib

        self._on_message = on_message
        self._running = True

        self._imap_client = aioimaplib.IMAP4_SSL(
            host=self._imap_host,
            port=self._imap_port,
        )
        await self._imap_client.wait_hello_from_server()
        await self._imap_client.login(self._username, self._password)
        await self._imap_client.select("INBOX")

        logger.info("email.started", host=self._imap_host, user=self._username)

        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Disconnect IMAP and cancel the polling task."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._imap_client is not None:
            try:
                await self._imap_client.logout()
            except Exception:
                logger.debug("email.imap_logout_error")

        logger.info("email.stopped")

    # ── Sending ──────────────────────────────────────────────────

    async def send(self, message: OutgoingMessage) -> None:
        """Send an email via SMTP, preserving reply threading."""
        import aiosmtplib

        msg = self._build_email(message)

        await aiosmtplib.send(
            msg,
            hostname=self._smtp_host,
            port=self._smtp_port,
            username=self._username,
            password=self._password,
            start_tls=True,
        )

        logger.debug("email.sent", to=message.chat_id)

    async def send_typing(self, chat_id: str) -> None:
        """Email has no typing indicator — this is a no-op."""
        logger.debug("email.typing_noop", chat_id=chat_id)

    # ── Internal: polling loop ───────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages.  Attempts IDLE first, falls back to interval polling."""
        idle_supported = await self._try_idle_capability()

        while self._running:
            try:
                if idle_supported:
                    await self._idle_wait()
                else:
                    await asyncio.sleep(self._poll_interval)

                await self._fetch_new_messages()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("email.poll_error")
                await asyncio.sleep(self._poll_interval)

    async def _try_idle_capability(self) -> bool:
        """Check if the IMAP server advertises IDLE."""
        if self._imap_client is None:
            return False
        try:
            result, data = await self._imap_client.capability()
            caps = " ".join(d for d in data if isinstance(d, str)).upper()
            return "IDLE" in caps
        except Exception:
            return False

    async def _idle_wait(self) -> None:
        """Use IMAP IDLE to wait for new mail (with a timeout to re-check)."""
        if self._imap_client is None:
            return
        try:
            idle_task = await self._imap_client.idle_start(timeout=self._poll_interval)
            await asyncio.wait_for(idle_task, timeout=self._poll_interval + 5)
        except (asyncio.TimeoutError, Exception):
            pass
        finally:
            try:
                self._imap_client.idle_done()
            except Exception:
                pass

    async def _fetch_new_messages(self) -> None:
        """Search for UNSEEN messages and process them."""
        if self._imap_client is None or self._on_message is None:
            return

        result, data = await self._imap_client.search("UNSEEN")
        if result != "OK":
            return

        # data is a list; first element is a space-separated list of UIDs
        # aioimaplib may return bytes (b'10 11 12') — decode to str
        raw_uids = data[0] if data else ""
        if isinstance(raw_uids, bytes):
            raw_uids = raw_uids.decode("ascii", errors="replace")
        if not raw_uids or not raw_uids.strip():
            return
        uids_str = raw_uids

        for uid in uids_str.strip().split():
            try:
                await self._process_uid(uid)
            except Exception:
                logger.exception("email.process_uid_error", uid=uid)

    async def _process_uid(self, uid: str) -> None:
        """Fetch and process a single email by UID."""
        if self._imap_client is None or self._on_message is None:
            return

        result, data = await self._imap_client.fetch(uid, "(RFC822)")
        if result != "OK" or not data:
            return

        raw_bytes: bytes = data[1] if len(data) > 1 else data[0]
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8", errors="replace")

        msg = email_lib.message_from_bytes(raw_bytes)

        # Newsletter filtering
        if self._filter_newsletters and self._is_newsletter(msg):
            logger.debug("email.filtered_newsletter", subject=msg.get("Subject"))
            return

        incoming = self._parse_email(msg, uid)
        await self._on_message(incoming)

        # Mark as read to prevent re-processing
        try:
            await self._imap_client.store(uid, "+FLAGS", "\\Seen")
        except Exception:
            logger.debug("email.mark_seen_failed", uid=uid)

    # ── Internal: email parsing ──────────────────────────────────

    def _parse_email(self, msg: EmailMessage, uid: str) -> IncomingMessage:
        """Convert a stdlib EmailMessage into an IncomingMessage."""
        sender = msg.get("From", "")
        subject = msg.get("Subject", "(no subject)")
        message_id = msg.get("Message-ID", "")
        references = msg.get("References", "")

        text = self._extract_text(msg)
        attachments = self._extract_attachments(msg)

        # Store threading info
        sender_email = self._extract_email_address(sender)
        self._thread_map[sender_email] = {
            "message_id": message_id,
            "references": references,
            "subject": subject,
        }

        return IncomingMessage(
            text=f"Subject: {subject}\n\n{text}",
            sender_id=sender_email,
            chat_id=sender_email,
            channel_id=self.channel_id,
            timestamp=time.time(),
            reply_to_message_id=message_id,
            attachments=attachments,
            raw={
                "uid": uid,
                "message_id": message_id,
                "references": references,
                "subject": subject,
                "from": sender,
            },
        )

    @staticmethod
    def _extract_text(msg: EmailMessage) -> str:
        """Extract plain text from an email, converting HTML if needed."""
        plain_parts: list[str] = []
        html_parts: list[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in disposition:
                    continue
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
                if content_type == "text/plain":
                    plain_parts.append(decoded)
                elif content_type == "text/html":
                    html_parts.append(decoded)
        else:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
                if msg.get_content_type() == "text/html":
                    html_parts.append(decoded)
                else:
                    plain_parts.append(decoded)

        if plain_parts:
            return "\n".join(plain_parts)
        if html_parts:
            return html_to_text("\n".join(html_parts))
        return ""

    @staticmethod
    def _extract_attachments(msg: EmailMessage) -> list[Attachment]:
        """Save attachments to temp files and return Attachment objects."""
        attachments: list[Attachment] = []
        if not msg.is_multipart():
            return attachments

        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" not in disposition:
                continue
            filename = part.get_filename() or "attachment"
            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            # Write to a temp file
            suffix = Path(filename).suffix or ".bin"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="astridr_")
            tmp.write(payload)
            tmp.close()

            attachments.append(
                Attachment(
                    file_path=tmp.name,
                    mime_type=part.get_content_type() or "application/octet-stream",
                    filename=filename,
                )
            )
        return attachments

    @staticmethod
    def _is_newsletter(msg: EmailMessage) -> bool:
        """Heuristic check for newsletters / bulk email."""
        for header in _NEWSLETTER_HEADERS:
            if msg.get(header):
                return True
        precedence = (msg.get("Precedence") or "").lower()
        if precedence in ("bulk", "list", "junk"):
            return True
        return False

    @staticmethod
    def _extract_email_address(from_header: str) -> str:
        """Pull the bare email from a From header like 'Name <a@b.com>'."""
        _, addr = email_lib.utils.parseaddr(from_header)
        return addr or from_header

    # ── Internal: email building ─────────────────────────────────

    _SIGNATURE_HTML = """\
<br><br>
<table cellpadding="0" cellspacing="0" style="font-family:Arial,sans-serif;font-size:13px;color:#555;">
  <tr>
    <td style="padding-right:12px;vertical-align:middle;">
      <img src="cid:astridr-avatar" alt="\u00c1str\u00ed\u00f0r" width="48" height="48"
           style="border-radius:50%;display:block;" />
    </td>
    <td style="vertical-align:middle;">
      <strong style="color:#222;">\u00c1str\u00ed\u00f0r</strong> \u26a1<br>
      <span style="font-size:11px;color:#888;">AI Assistant</span>
    </td>
  </tr>
</table>"""

    _SIGNATURE_PLAIN = "\n\n\u2014 \u00c1str\u00ed\u00f0r \u26a1"

    def _build_email(self, message: OutgoingMessage) -> email_lib.mime.multipart.MIMEMultipart:
        """Build a MIME email with HTML signature, inline avatar, and reply threading."""
        msg = email_lib.mime.multipart.MIMEMultipart("mixed")
        msg["From"] = self._username
        msg["To"] = message.chat_id
        msg["Date"] = email_lib.utils.formatdate(localtime=True)
        thread_info = self._thread_map.get(message.chat_id)
        original_subject = thread_info.get("subject", "") if thread_info else ""
        if original_subject and not original_subject.lower().startswith("re:"):
            msg["Subject"] = f"Re: {original_subject}"
        elif original_subject:
            msg["Subject"] = original_subject
        else:
            msg["Subject"] = "Re: (conversation)"

        # Thread headers
        if thread_info and message.reply_to_message_id:
            msg["In-Reply-To"] = message.reply_to_message_id
            refs = thread_info.get("references", "")
            if refs:
                msg["References"] = f"{refs} {message.reply_to_message_id}"
            else:
                msg["References"] = message.reply_to_message_id

        # Build multipart/alternative (plain + HTML)
        alt = email_lib.mime.multipart.MIMEMultipart("alternative")

        plain_text = message.text + self._SIGNATURE_PLAIN
        alt.attach(email_lib.mime.text.MIMEText(plain_text, "plain", "utf-8"))

        body_html = html.escape(message.text).replace("\n", "<br>\n")
        html_content = (
            f'<div style="font-family:Arial,sans-serif;font-size:14px;color:#222;">'
            f"{body_html}"
            f"{self._SIGNATURE_HTML}"
            f"</div>"
        )
        alt.attach(email_lib.mime.text.MIMEText(html_content, "html", "utf-8"))

        # Wrap in multipart/related so inline avatars are embedded, not attached
        import email.mime.image

        inline_images: list[tuple[bytes, str, str, str]] = []  # (data, subtype, cid, filename)

        if self._avatar_path and self._avatar_path.exists():
            inline_images.append((
                self._avatar_path.read_bytes(), "png", "astridr-avatar", "avatar.png",
            ))

        if self._personal_avatar_path and self._personal_avatar_path.exists():
            suffix = self._personal_avatar_path.suffix.lower()
            subtype = "jpeg" if suffix in (".jpg", ".jpeg") else "png"
            inline_images.append((
                self._personal_avatar_path.read_bytes(), subtype, "avatar", f"personal-avatar{suffix}",
            ))

        if inline_images:
            related = email_lib.mime.multipart.MIMEMultipart("related")
            related.attach(alt)

            for img_data, subtype, cid, filename in inline_images:
                img_part = email.mime.image.MIMEImage(img_data, _subtype=subtype)
                img_part.add_header("Content-ID", f"<{cid}>")
                img_part.add_header("Content-Disposition", "inline", filename=filename)
                related.attach(img_part)

            msg.attach(related)
        else:
            msg.attach(alt)

        # File attachments
        for att in message.attachments:
            part = email_lib.mime.base.MIMEBase(
                *(att.mime_type.split("/", 1) if "/" in att.mime_type else ("application", "octet-stream")),
            )
            if att.file_bytes:
                part.set_payload(att.file_bytes)
            elif att.file_path:
                part.set_payload(Path(att.file_path).read_bytes())
            else:
                continue
            email_lib.encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=att.filename)
            msg.attach(part)

        return msg


# ── Utility: HTML → plain text ───────────────────────────────────


def html_to_text(raw_html: str) -> str:
    """Lightweight HTML to plain-text conversion for LLM consumption.

    Strips tags, decodes entities, normalises whitespace.
    """
    # Remove <style> and <script> blocks entirely
    text = re.sub(r"<(style|script)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    # Replace <br>, <p>, <div>, <li> with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalise whitespace (preserve newlines)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
