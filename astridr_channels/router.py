"""Message Router — routes incoming messages to the correct profile and session.

The router is the central dispatcher: every incoming message from any channel
passes through here.  It resolves the sender to a profile, manages sessions,
runs the security pipeline, invokes the agent loop, and sends the response
back through the originating channel.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from astridr.channels.base import (
    Attachment,
    BaseChannel,
    IncomingMessage,
    OutgoingMessage,
)
from astridr.engine.config import ProfileConfig
from astridr.engine.telemetry import ConvexHandler

# Optional import — profiles module may not be available yet during early bootstrap
try:
    from astridr.agent.profiles import AgentProfile, ProfileManager
except ImportError:  # pragma: no cover
    ProfileManager = None  # type: ignore[assignment,misc]
    AgentProfile = None  # type: ignore[assignment,misc]

logger = structlog.get_logger()


@dataclass
class Session:
    """An active conversation session between a user and a profile."""

    id: str
    chat_id: str
    profile_id: str
    channel_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class MessageRouter:
    """Routes incoming messages from channels to profiles, sessions, and the agent loop.

    Pipeline per message:
      1. Check emergency stop
      2. Resolve profile from sender + channel
      3. Get or create session
      4. Security pipeline (inbound)
      5. Agent loop processing
      6. Security pipeline (outbound)
      7. Send response via originating channel
    """

    def __init__(
        self,
        profiles: list[ProfileConfig],
        telemetry: ConvexHandler,
        *,
        agent_loop: Any | None = None,
        security_pipeline: Any | None = None,
        emergency_stop: bool = False,
        persistence: Any | None = None,
        profile_manager: Any | None = None,
        system_prompt_builder: Any | None = None,
        all_tool_defs: list[Any] | None = None,
        episodic: Any | None = None,
        message_queue: Any | None = None,
        connectivity_monitor: Any | None = None,
    ) -> None:
        self._profiles = {p.id: p for p in profiles}
        self._telemetry = telemetry
        self._emergency_stop = emergency_stop
        self._persistence = persistence
        self._episodic = episodic

        # Sessions keyed by (channel_id, chat_id)
        self._sessions: dict[tuple[str, str], Session] = {}

        # Agent loop (injected from bootstrap)
        self._agent_loop_instance = agent_loop

        # Agent sessions keyed by router session id -> agent Session
        self._agent_sessions: dict[str, Any] = {}

        # Security pipeline (injected from bootstrap)
        self._security_pipeline: Any = security_pipeline
        if self._security_pipeline is None:
            self._try_load_security_fallback()

        # TTS tool (lazy-initialized on first use)
        self._tts_tool: Any | None = None

        # Profile switching support
        self._profile_manager: Any | None = profile_manager
        self._system_prompt_builder: Any | None = system_prompt_builder
        self._all_tool_defs: list[Any] = all_tool_defs or []

        # Offline mode: queue + connectivity monitor
        self._message_queue = message_queue
        self._connectivity_monitor = connectivity_monitor

    # -- Episodic helpers ------------------------------------------------------

    def _record_event(self, agent_id: str, event_type: str, summary: str, detail: dict | None = None) -> None:
        if self._episodic is not None:
            asyncio.create_task(self._safe_record(agent_id, event_type, summary, detail))

    async def _safe_record(self, agent_id: str, event_type: str, summary: str, detail: dict | None = None) -> None:
        try:
            await self._episodic.record(agent_id, event_type, summary, detail)
        except Exception:
            logger.debug("episodic.record_failed", event_type=event_type, exc_info=True)

    # -- Queue-aware send -----------------------------------------------------

    async def _send_or_queue(self, channel: BaseChannel, outgoing: OutgoingMessage) -> None:
        """Send a message via *channel*, falling back to the offline queue on transient errors."""
        try:
            await channel.send(outgoing)
        except Exception as exc:
            from astridr.engine.offline import is_transient_send_error

            if self._message_queue is not None and is_transient_send_error(exc):
                await self._message_queue.enqueue(
                    channel.channel_id,
                    {
                        "text": outgoing.text,
                        "chat_id": outgoing.chat_id,
                        "reply_to_message_id": outgoing.reply_to_message_id,
                    },
                )
                logger.warning(
                    "router.send_queued",
                    channel=channel.channel_id,
                    chat_id=outgoing.chat_id,
                    error=str(exc),
                )
            else:
                raise

    # -- Public API -----------------------------------------------------------

    async def route(self, message: IncomingMessage, channel: BaseChannel) -> None:
        """Full routing pipeline for an incoming message."""

        # 1. Emergency stop -- drop everything
        if self._emergency_stop:
            logger.warning("router.emergency_stop", sender=message.sender_id)
            await self._telemetry.send(
                "security_event",
                {
                    "layer": "emergency_stop",
                    "sender": message.sender_id,
                    "channel": message.channel_id,
                },
            )
            return

        # 2. Resolve profile
        try:
            profile = self.resolve_profile(
                message.sender_id, channel, chat_id=message.chat_id
            )
        except ValueError:
            logger.warning(
                "router.no_profile",
                sender=message.sender_id,
                channel=channel.channel_id,
            )
            return

        # 3. Get or create session
        session = self.get_or_create_session(message.chat_id, profile, channel.channel_id)
        session.last_activity = time.time()
        session.messages.append(
            {
                "role": "user",
                "content": message.text,
                "timestamp": message.timestamp,
            }
        )

        # 4. Security pipeline (inbound)
        inbound_text = message.text
        if self._security_pipeline is not None:
            try:
                inbound_result = await self._run_security_inbound(
                    message, session, profile
                )
                if inbound_result is None:
                    # Message was blocked
                    return
                # Use the (possibly redacted) text
                inbound_text = inbound_result
            except Exception:
                logger.exception("router.security_inbound_error")

        # 4b. Slash-command interception: /profile
        if inbound_text.strip().startswith("/profile"):
            response_text = await self._handle_profile_command(
                inbound_text.strip(), session, profile
            )
            session.messages.append(
                {"role": "assistant", "content": response_text, "timestamp": time.time()}
            )
            outgoing = OutgoingMessage(
                text=response_text,
                chat_id=message.chat_id,
                reply_to_message_id=message.reply_to_message_id,
            )
            await self._send_or_queue(channel, outgoing)
            return

        # 5. Agent loop
        response_text = await self._process_agent(inbound_text, message, session, profile)
        session.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "timestamp": time.time(),
            }
        )

        # 6. Security pipeline (outbound)
        final_text = response_text
        if self._security_pipeline is not None:
            try:
                outbound_result = await self._run_security_outbound(
                    response_text, message, session, profile
                )
                if outbound_result is None:
                    # Outbound was blocked -- send a safe generic response
                    final_text = "I'm sorry, I can't provide that response. Please try rephrasing your request."
                    session.messages[-1]["content"] = final_text
                else:
                    final_text = outbound_result
                    session.messages[-1]["content"] = final_text
            except Exception:
                logger.exception("router.security_outbound_error")

        # 7. TTS generation (if enabled on the profile)
        tts_attachments: list[Attachment] = []
        if profile.tts_enabled:
            tts_attachment = await self._generate_tts(final_text)
            if tts_attachment is not None:
                tts_attachments.append(tts_attachment)

        # 8. Send response back via channel
        outgoing = OutgoingMessage(
            text=final_text,
            chat_id=message.chat_id,
            reply_to_message_id=message.reply_to_message_id,
            attachments=tts_attachments,
        )

        await self._send_or_queue(channel, outgoing)

        # Persist session snapshot (fire-and-forget)
        if self._persistence is not None:
            turn_count = len([m for m in session.messages if m.get("role") == "user"])
            self._persistence.upsert_session_bg(
                session_id=session.id,
                profile_id=profile.id,
                channel_id=channel.channel_id,
                messages=session.messages,
                turn_count=turn_count,
            )

        # Telemetry
        await self._telemetry.send(
            "message_routed",
            {
                "profile": profile.id,
                "channel": channel.channel_id,
                "session_id": session.id,
                "sender": message.sender_id,
            },
        )

    def resolve_profile(
        self, sender_id: str, channel: BaseChannel, *, chat_id: str = ""
    ) -> ProfileConfig:
        """Map a sender + channel + chat_id to a ProfileConfig.

        Resolution order (3-tier):
          1. channel_mappings — exact match of (channel_type, chat_id)
          2. default_for — fallback profile for unmapped traffic on this channel
          3. channels — broad channel-type match (backward compat)
        """
        channel_type = channel.channel_id

        # Tier 1: channel_mappings — exact chat_id match
        if chat_id:
            for profile in self._profiles.values():
                mapped_ids = profile.channel_mappings.get(channel_type, [])
                if channel_type == "web":
                    # Web uses startswith for path prefixes
                    for prefix in mapped_ids:
                        if chat_id.startswith(prefix):
                            return profile
                else:
                    if chat_id in mapped_ids:
                        return profile

        # Tier 2: default_for — catch-all for unmapped traffic on this channel
        for profile in self._profiles.values():
            if channel_type in profile.default_for:
                return profile

        # Tier 3: channels list — backward compat broad match
        for profile in self._profiles.values():
            if channel_type in profile.channels or "*" in profile.channels:
                return profile

        raise ValueError(
            f"No profile found for sender={sender_id} channel={channel_type} chat_id={chat_id}"
        )

    def get_or_create_session(
        self,
        chat_id: str,
        profile: ProfileConfig,
        channel_id: str = "",
    ) -> Session:
        """Retrieve an existing session or create a new one.

        Sessions are keyed by (channel_id, chat_id) so the same user
        on different channels gets separate sessions.
        """
        key = (channel_id, chat_id)

        if key in self._sessions:
            return self._sessions[key]

        session = Session(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            profile_id=profile.id,
            channel_id=channel_id,
        )
        self._sessions[key] = session

        logger.info(
            "router.session_created",
            session_id=session.id,
            chat_id=chat_id,
            profile=profile.id,
            channel=channel_id,
        )

        self._record_event("astridr", "session_start",
            f"Session started on {channel_id}",
            {"session_id": session.id, "profile_id": profile.id, "channel_id": channel_id})

        return session

    def set_emergency_stop(self, active: bool) -> None:
        """Toggle emergency stop. When active, all messages are dropped."""
        self._emergency_stop = active
        logger.warning("router.emergency_stop_toggled", active=active)

    @property
    def sessions(self) -> dict[tuple[str, str], Session]:
        """Read-only access to active sessions (for diagnostics)."""
        return dict(self._sessions)

    # -- Internal -------------------------------------------------------------

    async def _run_security_inbound(
        self,
        message: IncomingMessage,
        session: Session,
        profile: ProfileConfig,
    ) -> str | None:
        """Run security pipeline on an inbound message.

        Returns the (possibly redacted) text if allowed, or None if blocked.
        Supports both the legacy .check() interface and the new SecurityPipeline
        with process_inbound/process_outbound.
        """
        pipeline = self._security_pipeline

        # Detect whether this is the real SecurityPipeline or a legacy/mock object.
        # The real SecurityPipeline is identified by being an instance of the class.
        is_real_pipeline = False
        try:
            from astridr.security.pipeline import SecurityPipeline

            is_real_pipeline = isinstance(pipeline, SecurityPipeline)
        except (ImportError, ModuleNotFoundError):
            pass

        if is_real_pipeline:
            # New interface: SecurityPipeline with process_inbound/process_outbound
            from astridr.security.pipeline import SecurityContext

            ctx = SecurityContext(
                profile_id=profile.id,
                channel_id=session.channel_id,
                sender_id=message.sender_id,
                session_id=session.id,
            )
            result = await pipeline.process_inbound(message.text, ctx)
            if not result.allowed:
                logger.warning(
                    "router.security_blocked",
                    sender=message.sender_id,
                    profile=profile.id,
                    reason=result.blocked_reason,
                )
                await self._telemetry.send(
                    "security_event",
                    {
                        "layer": "security_pipeline",
                        "action": "blocked",
                        "direction": "inbound",
                        "sender": message.sender_id,
                        "profile": profile.id,
                        "reason": result.blocked_reason,
                    },
                )
                return None
            return result.message

        # Legacy interface: .check(message, profile) -> bool
        if hasattr(pipeline, "check"):
            allowed = await pipeline.check(message, profile)
            if not allowed:
                logger.warning(
                    "router.security_blocked",
                    sender=message.sender_id,
                    profile=profile.id,
                )
                await self._telemetry.send(
                    "security_event",
                    {
                        "layer": "security_pipeline",
                        "action": "blocked",
                        "sender": message.sender_id,
                        "profile": profile.id,
                    },
                )
                return None
            return message.text

        return message.text

    async def _run_security_outbound(
        self,
        response_text: str,
        message: IncomingMessage,
        session: Session,
        profile: ProfileConfig,
    ) -> str | None:
        """Run security pipeline on an outbound response.

        Returns the (possibly redacted) text if allowed, or None if blocked.
        Only applicable for the real SecurityPipeline interface.
        """
        pipeline = self._security_pipeline

        # Only the real SecurityPipeline supports process_outbound
        is_real_pipeline = False
        try:
            from astridr.security.pipeline import SecurityPipeline

            is_real_pipeline = isinstance(pipeline, SecurityPipeline)
        except (ImportError, ModuleNotFoundError):
            pass

        if not is_real_pipeline:
            return response_text

        from astridr.security.pipeline import SecurityContext

        ctx = SecurityContext(
            profile_id=profile.id,
            channel_id=session.channel_id,
            sender_id=message.sender_id,
            session_id=session.id,
        )
        result = await pipeline.process_outbound(response_text, ctx)
        if not result.allowed:
            logger.warning(
                "router.security_outbound_blocked",
                profile=profile.id,
                reason=result.blocked_reason,
            )
            await self._telemetry.send(
                "security_event",
                {
                    "layer": "security_pipeline",
                    "action": "blocked",
                    "direction": "outbound",
                    "profile": profile.id,
                    "reason": result.blocked_reason,
                },
            )
            return None
        return result.message

    async def _process_agent(
        self,
        text: str,
        message: IncomingMessage,
        session: Session,
        profile: ProfileConfig,
    ) -> str:
        """Process a message through the agent loop.

        If an agent loop is configured, creates an agent session and processes
        the message. Otherwise falls back to the echo stub.
        """
        if self._agent_loop_instance is None:
            # Fallback stub -- echo with profile context
            return f"[{profile.name}] Received: {message.text}"

        try:
            from astridr.agent.loop import Session as AgentSession
            from astridr.providers.base import Message

            # Get or create an agent session for this router session
            agent_session = self._get_or_create_agent_session(session)

            user_msg = Message(role="user", content=text)
            response_text = await self._agent_loop_instance.process(user_msg, agent_session)
            return response_text
        except Exception:
            logger.exception("router.agent_loop_error")
            self._record_event("astridr", "error",
                f"Agent loop error for profile {profile.id}",
                {"session_id": session.id, "profile_id": profile.id})
            return f"[{profile.name}] I encountered an error processing your request. Please try again."

    def _get_or_create_agent_session(self, router_session: Session) -> Any:
        """Get or create an agent Session keyed by the router session id."""
        if router_session.id in self._agent_sessions:
            return self._agent_sessions[router_session.id]

        from astridr.agent.loop import Session as AgentSession

        agent_session = AgentSession(id=router_session.id)
        self._agent_sessions[router_session.id] = agent_session

        logger.debug(
            "router.agent_session_created",
            session_id=router_session.id,
        )
        return agent_session

    async def _handle_profile_command(
        self,
        text: str,
        session: Session,
        profile: ProfileConfig,
    ) -> str:
        """Handle /profile slash commands.

        Supported forms:
          /profile list      — show available profiles
          /profile use <id>  — switch to the given profile
          /profile current   — show active profile
          /profile           — same as /profile current
        """
        if self._profile_manager is None:
            return "Agent profiles are not configured."

        parts = text.split(maxsplit=2)
        sub_cmd = parts[1] if len(parts) > 1 else "current"

        if sub_cmd == "list":
            profiles = self._profile_manager.list_profiles()
            lines = ["**Available Profiles:**"]
            for p in profiles:
                marker = " (active)" if p.id == self._get_active_profile_id(session) else ""
                desc = f" — {p.soul_override}" if p.soul_override else ""
                lines.append(f"  • `{p.id}` — {p.name}{desc}{marker}")
            return "\n".join(lines)

        elif sub_cmd == "use":
            profile_id = parts[2].strip() if len(parts) > 2 else ""
            if not profile_id:
                return "Usage: `/profile use <id>`. Use `/profile list` to see options."
            try:
                agent_profile = self._profile_manager.get(profile_id)
            except KeyError:
                available = [p.id for p in self._profile_manager.list_profiles()]
                return f"Profile `{profile_id}` not found. Available: {', '.join(available)}"

            self._apply_profile_to_session(session, agent_profile)
            logger.info(
                "router.profile_switched",
                session_id=session.id,
                profile=agent_profile.id,
            )
            self._record_event("astridr", "profile_switch",
                f"Switched to {agent_profile.id} ({agent_profile.name})",
                {"session_id": session.id, "to_profile": agent_profile.id})
            return f"Switched to **{agent_profile.name}** (`{agent_profile.id}`)."

        elif sub_cmd == "current":
            current_id = self._get_active_profile_id(session)
            try:
                p = self._profile_manager.get(current_id)
                return f"Active profile: **{p.name}** (`{p.id}`)"
            except KeyError:
                return f"Active profile: `{current_id}`"

        else:
            return (
                "Unknown sub-command. Usage:\n"
                "  `/profile list` — show profiles\n"
                "  `/profile use <id>` — switch profile\n"
                "  `/profile current` — show active profile"
            )

    def _apply_profile_to_session(self, session: Session, agent_profile: Any) -> None:
        """Apply an AgentProfile to the agent session (prompt rebuild + tool filtering)."""
        agent_session = self._get_or_create_agent_session(session)
        agent_session.active_profile = agent_profile.id

        # Rebuild system prompt with the profile's soul_override and rules_override
        if self._system_prompt_builder is not None:
            new_prompt = self._system_prompt_builder(agent_profile)
            agent_session._override_system_prompt = new_prompt

        # Filter tools per profile
        if self._all_tool_defs and ProfileManager is not None:
            filtered = ProfileManager.filter_tools(self._all_tool_defs, agent_profile)
            agent_session._override_tools = filtered

        # Override max_rounds and temperature
        agent_session._override_max_rounds = agent_profile.max_rounds
        agent_session._override_temperature = agent_profile.temperature

    def _get_active_profile_id(self, session: Session) -> str:
        """Return the active agent profile ID for a router session."""
        if session.id in self._agent_sessions:
            return getattr(self._agent_sessions[session.id], "active_profile", "default")
        return "default"

    def _get_tts_tool(self) -> Any:
        """Lazily initialise and return the TTS tool."""
        if self._tts_tool is None:
            try:
                from astridr.media.tts import TextToSpeechTool

                self._tts_tool = TextToSpeechTool()
                logger.debug("router.tts_tool_loaded")
            except Exception:
                logger.debug("router.tts_tool_not_available")
        return self._tts_tool

    async def _generate_tts(self, text: str) -> Attachment | None:
        """Generate a TTS voice note for *text*.

        Returns an ``Attachment`` on success, or ``None`` if TTS is
        unavailable or the generation fails (non-blocking).
        """
        tool = self._get_tts_tool()
        if tool is None:
            return None

        try:
            import os

            voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "default")
            result = await tool.execute(text=text, voice_id=voice_id)
            if result.success and result.data and result.data.get("path"):
                return Attachment(
                    file_path=result.data["path"],
                    mime_type="audio/mpeg",
                    filename="reply.mp3",
                )
            logger.warning("router.tts_failed", error=result.error)
        except Exception:
            logger.exception("router.tts_error")
        return None

    def _try_load_security_fallback(self) -> None:
        """Try to import and build the SecurityPipeline as a fallback.

        Only used when no security_pipeline is explicitly injected.
        """
        try:
            from astridr.security.pipeline import SecurityPipeline

            self._security_pipeline = SecurityPipeline(layers=[])
            logger.debug("router.security_pipeline_loaded")
        except (ImportError, ModuleNotFoundError, Exception):
            self._security_pipeline = None
            logger.debug("router.security_pipeline_not_available")
