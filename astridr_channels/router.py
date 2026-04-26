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

from astridr.core.secrets import get_snapshot

from astridr.channels.base import (
    Attachment,
    BaseChannel,
    IncomingMessage,
    OutgoingMessage,
)
from astridr.engine.config import PersonaConfig, ProfileConfig
from astridr.engine.estop import EmergencyStop
from astridr.engine.session_keys import SessionKeyComposer
from astridr.engine.voice_identity import VoiceIdentityResolver
from astridr.engine.telemetry import ConvexHandler
from astridr.memory.recent_cache import RecentContextCache

# Phase 44 hook imports (lazy to avoid circular)
try:
    from astridr.engine.hooks import HookContext, HookPoint
except ImportError:  # pragma: no cover
    HookContext = None  # type: ignore[assignment,misc]
    HookPoint = None  # type: ignore[assignment,misc]

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
        personas: list[PersonaConfig] | None = None,
        persistence: Any | None = None,
        profile_manager: Any | None = None,
        system_prompt_builder: Any | None = None,
        all_tool_defs: list[Any] | None = None,
        episodic: Any | None = None,
        message_queue: Any | None = None,
        connectivity_monitor: Any | None = None,
        flow_registry: Any | None = None,
        stall_detector: Any | None = None,
        key_composer: SessionKeyComposer | None = None,
        estop: EmergencyStop | None = None,
        hook_registry: Any | None = None,
        snapshot_manager: Any | None = None,
        memory_store: Any | None = None,
        tool_registry: Any | None = None,
    ) -> None:
        self._profiles = {p.id: p for p in profiles}
        self._telemetry = telemetry
        self._emergency_stop = emergency_stop
        self._estop = estop

        # Pre-built profile resolution indexes (avoid O(n) loops per message)
        self._mapping_index: dict[str, dict[str, str]] = {}   # {channel_type: {chat_id: profile_id}}
        self._web_prefix_index: dict[str, list[tuple[str, str]]] = {}  # {channel_type: [(prefix, profile_id)]}
        self._default_for_index: dict[str, str] = {}           # {channel_type: profile_id}
        self._channels_index: dict[str, str] = {}              # {channel_type: profile_id}
        self._wildcard_profile_id: str | None = None
        self._build_profile_index(profiles)
        self._persistence = persistence
        self._episodic = episodic
        self._stall_detector = stall_detector
        self._key_composer = key_composer

        # Sessions keyed by composite key (str) when key_composer is set,
        # or by (channel_id, chat_id) tuple for legacy backward compatibility.
        self._sessions: dict[Any, Session] = {}

        # Agent loop (injected from bootstrap)
        self._agent_loop_instance = agent_loop

        # Agent sessions keyed by router session id -> agent Session
        self._agent_sessions: dict[str, Any] = {}

        # Security pipeline (injected from bootstrap)
        self._security_pipeline: Any = security_pipeline
        if self._security_pipeline is None:
            self._try_load_security_fallback()

        # Voice identity resolver (constructed once from persona config)
        self._voice_resolver = VoiceIdentityResolver(personas or [])

        # TTS tool (lazy-initialized on first use)
        self._tts_tool: Any | None = None

        # Profile switching support
        self._profile_manager: Any | None = profile_manager
        self._system_prompt_builder: Any | None = system_prompt_builder
        self._all_tool_defs: list[Any] = all_tool_defs or []

        # Offline mode: queue + connectivity monitor
        self._message_queue = message_queue
        self._connectivity_monitor = connectivity_monitor

        # Hot cache for recent conversation context (Pattern 6)
        self._recent_cache = RecentContextCache()

        # Plugin channel registry — keyed by channel_id (CHAN-03)
        self._plugin_channels: dict[str, BaseChannel] = {}

        # Flow registry for /tasks command (ORCH-07)
        self._flow_registry: Any | None = flow_registry

        # ModelRouter reference for /model command session overrides (ROUTE-05)
        self._model_router: Any | None = None

        # Turn lock per chat_id — prevents heartbeat alert + user message interleaving (D-10)
        self._turn_locks: dict[str, asyncio.Lock] = {}

        # Per-channel message counters for infrastructure telemetry
        self._channel_msg_counts: dict[str, int] = {}
        self._channel_last_latency_ms: dict[str, float] = {}

        # Phase 44 operations components — all optional
        self._hook_registry = hook_registry
        self._snapshot_manager = snapshot_manager
        self._memory_store = memory_store
        self._tool_registry = tool_registry

    def _build_profile_index(self, profiles: list[ProfileConfig]) -> None:
        """Pre-build O(1) lookup indexes for profile resolution."""
        for profile in profiles:
            for ch_type, chat_ids in profile.channel_mappings.items():
                if ch_type == "web":
                    self._web_prefix_index.setdefault(ch_type, []).extend(
                        (prefix, profile.id) for prefix in chat_ids
                    )
                else:
                    mapping = self._mapping_index.setdefault(ch_type, {})
                    for cid in chat_ids:
                        mapping[cid] = profile.id
            for ch_type in profile.default_for:
                self._default_for_index.setdefault(ch_type, profile.id)
            for ch_type in profile.channels:
                if ch_type == "*":
                    self._wildcard_profile_id = self._wildcard_profile_id or profile.id
                else:
                    self._channels_index.setdefault(ch_type, profile.id)

    # -- Setters for post-construction injection --------------------------------

    def set_model_router(self, router: Any) -> None:
        """Wire the ModelRouter for /model command session override management (ROUTE-05)."""
        self._model_router = router

    def channel_stats(self) -> list[dict[str, Any]]:
        """Return per-channel message count and last latency for infra telemetry."""
        all_ids = set(self._channel_msg_counts) | set(self._channel_last_latency_ms)
        return [
            {
                "channel": ch_id,
                "messageCount": self._channel_msg_counts.get(ch_id, 0),
                "lastLatencyMs": self._channel_last_latency_ms.get(ch_id),
            }
            for ch_id in sorted(all_ids)
        ]

    def get_turn_lock(self, chat_id: str) -> asyncio.Lock:
        """Get or create the turn lock for a chat_id (D-10).

        Each chat_id gets its own asyncio.Lock so concurrent heartbeat alerts
        and user messages on the same chat are serialized without blocking
        unrelated chats.
        """
        if chat_id not in self._turn_locks:
            self._turn_locks[chat_id] = asyncio.Lock()
        return self._turn_locks[chat_id]

    # -- Episodic helpers ------------------------------------------------------

    def _record_event(self, agent_id: str, event_type: str, summary: str, detail: dict | None = None) -> None:
        if self._episodic is not None:
            asyncio.create_task(self._safe_record(agent_id, event_type, summary, detail))

    async def _safe_record(self, agent_id: str, event_type: str, summary: str, detail: dict | None = None) -> None:
        try:
            await self._episodic.record(agent_id, event_type, summary, detail)
        except Exception:
            logger.debug("episodic.record_failed", event_type=event_type, exc_info=True)
        # Mirror to Convex so CodePulse Memory Browser shows data
        try:
            await self._telemetry.send(
                "episodic_event",
                {
                    "agentId": agent_id,
                    "eventType": event_type,
                    "summary": summary,
                    "detail": detail,
                    "occurredAt": time.time(),
                },
            )
        except Exception:
            logger.debug("episodic.telemetry_failed", event_type=event_type, exc_info=True)

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
        _ch_id = message.channel_id
        self._channel_msg_counts[_ch_id] = self._channel_msg_counts.get(_ch_id, 0) + 1

        # 1. Emergency stop — queue (global estop) or drop (legacy bool)
        if self._estop and self._estop.is_active:
            self._estop.queue_message(
                {
                    "sender_id": message.sender_id,
                    "channel_id": message.channel_id,
                    "chat_id": message.chat_id,
                    "content": message.text,
                    "queued_at": time.time(),
                }
            )
            logger.warning(
                "router.message_queued_estop",
                sender=message.sender_id,
                channel=message.channel_id,
            )
            return
        elif self._emergency_stop:
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

        lock = self.get_turn_lock(message.chat_id)
        _t0 = time.monotonic()
        async with lock:
            # 2. Resolve profile
            await self._route_locked(message, channel)
        self._channel_last_latency_ms[_ch_id] = round((time.monotonic() - _t0) * 1000, 1)

    async def _route_locked(self, message: IncomingMessage, channel: BaseChannel) -> None:
        """Execute routing pipeline steps 2-8 under the chat_id turn lock."""
        # 2. Resolve profile
        try:
            profile = self.resolve_profile(
                message.sender_id, channel, chat_id=message.chat_id,
                raw=message.raw,
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

        # Populate hot cache with user message
        self._recent_cache.append(session.id, "user", message.text)

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

        # 4b. Slash-command interception: /tasks (ORCH-07)
        if inbound_text.strip().lower() == "/tasks":
            response = await self._handle_tasks_command(message)
            session.messages.append(
                {"role": "assistant", "content": response.text, "timestamp": time.time()}
            )
            self._recent_cache.append(session.id, "assistant", response.text)
            await self._send_or_queue(channel, response)
            return

        # 4c. Slash-command interception: /profile
        if inbound_text.strip().startswith("/profile"):
            response_text = await self._handle_profile_command(
                inbound_text.strip(), session, profile
            )
            session.messages.append(
                {"role": "assistant", "content": response_text, "timestamp": time.time()}
            )
            self._recent_cache.append(session.id, "assistant", response_text)
            outgoing = OutgoingMessage(
                text=response_text,
                chat_id=message.chat_id,
                reply_to_message_id=message.reply_to_message_id,
            )
            await self._send_or_queue(channel, outgoing)
            return

        # 4d. Slash-command interception: /model (ROUTE-05)
        if inbound_text.strip().lower().startswith("/model"):
            response_text = await self._handle_model_command(
                inbound_text.strip(), session
            )
            session.messages.append(
                {"role": "assistant", "content": response_text, "timestamp": time.time()}
            )
            self._recent_cache.append(session.id, "assistant", response_text)
            outgoing = OutgoingMessage(
                text=response_text,
                chat_id=message.chat_id,
                reply_to_message_id=message.reply_to_message_id,
            )
            await self._send_or_queue(channel, outgoing)
            return

        # 4e. Slash-command interception: /autonomy (AP-09, D-05)
        if inbound_text.strip().lower().startswith("/autonomy"):
            response_text = await self._handle_autonomy_command(
                inbound_text.strip(), session
            )
            session.messages.append(
                {"role": "assistant", "content": response_text, "timestamp": time.time()}
            )
            self._recent_cache.append(session.id, "assistant", response_text)
            outgoing = OutgoingMessage(
                text=response_text,
                chat_id=message.chat_id,
                reply_to_message_id=message.reply_to_message_id,
            )
            await self._send_or_queue(channel, outgoing)
            return

        # 4f. Slash-command interception: /pair-whatsapp (Phase 68, D-10)
        if inbound_text.strip().lower() == "/pair-whatsapp":
            whatsapp_ch = self._plugin_channels.get("whatsapp")
            if whatsapp_ch and hasattr(whatsapp_ch, "request_pairing"):
                try:
                    await whatsapp_ch.request_pairing()
                    response_text = "WhatsApp pairing initiated. Check Telegram for the QR code."
                except Exception as exc:
                    response_text = f"Failed to start WhatsApp pairing: {exc}"
            else:
                response_text = "WhatsApp channel is not configured."
            response = OutgoingMessage(
                text=response_text,
                chat_id=message.chat_id,
            )
            session.messages.append(
                {"role": "assistant", "content": response.text, "timestamp": time.time()}
            )
            await self._send_or_queue(channel, response)
            return

        # 4g. Slash-command interception: /ingest (FIRE-01, Phase 73)
        if inbound_text.strip().lower().startswith("/ingest"):
            response = await self._handle_ingest_command(message)
            session.messages.append(
                {"role": "assistant", "content": response.text, "timestamp": time.time()}
            )
            self._recent_cache.append(session.id, "assistant", response.text)
            await self._send_or_queue(channel, response)
            return

        # 5. Agent loop
        response_text = await self._process_agent(inbound_text, message, session, profile)
        if self._stall_detector is not None:
            self._stall_detector.update_message_processed()
        session.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "timestamp": time.time(),
            }
        )

        # Populate hot cache with assistant response
        self._recent_cache.append(session.id, "assistant", response_text)

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

        # 7. Send text response immediately (don't block on TTS)
        voice_meta: dict[str, Any] = {}
        if profile.tts_enabled and channel.channel_id in ("voice", "web"):
            persona_id = getattr(profile, "persona_id", None)
            try:
                voice_id, stability, similarity_boost = self._voice_resolver.resolve(persona_id)
                voice_meta = {
                    "voice_id": voice_id,
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                }
            except Exception:
                logger.warning("router.voice_resolve_failed", persona_id=persona_id)

        outgoing = OutgoingMessage(
            text=final_text,
            chat_id=message.chat_id,
            reply_to_message_id=message.reply_to_message_id,
            metadata=voice_meta,
        )
        await self._send_or_queue(channel, outgoing)

        # 7b. TTS generation as non-blocking follow-up
        if profile.tts_enabled:
            persona_id = getattr(profile, "persona_id", None)
            asyncio.create_task(self._send_tts_followup(
                channel, message.chat_id, final_text, persona_id,
                message.reply_to_message_id,
            ))

        # Persist session snapshot (fire-and-forget with snapshot copy)
        if self._persistence is not None:
            msg_snapshot = list(session.messages)  # snapshot to avoid race with mutations
            turn_count = sum(1 for m in msg_snapshot if m.get("role") == "user")
            self._persistence.upsert_session_bg(
                session_id=session.id,
                profile_id=profile.id,
                channel_id=channel.channel_id,
                messages=msg_snapshot,
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
        self, sender_id: str, channel: BaseChannel, *, chat_id: str = "",
        raw: dict[str, Any] | None = None,
    ) -> ProfileConfig:
        """Map a sender + channel + chat_id to a ProfileConfig.

        Resolution order (4-tier):
          0. raw.profile_id — explicit override (e.g. from pipe execution)
          1. channel_mappings — exact match of (channel_type, chat_id)
          2. default_for — fallback profile for unmapped traffic on this channel
          3. channels — broad channel-type match (backward compat)
        """
        # Tier 0: explicit profile_id override (e.g. from pipe execution)
        if raw and "profile_id" in raw:
            override = self._profiles.get(raw["profile_id"])
            if override is not None:
                return override

        channel_type = channel.channel_id

        # Tier 1: channel_mappings — O(1) indexed lookup
        if chat_id:
            if channel_type == "web":
                for prefix, pid in self._web_prefix_index.get(channel_type, []):
                    if chat_id.startswith(prefix):
                        return self._profiles[pid]
            else:
                pid = self._mapping_index.get(channel_type, {}).get(chat_id)
                if pid is not None:
                    return self._profiles[pid]

        # Tier 2: default_for — O(1) indexed lookup
        pid = self._default_for_index.get(channel_type)
        if pid is not None:
            return self._profiles[pid]

        # Tier 3: channels list — O(1) indexed lookup
        pid = self._channels_index.get(channel_type) or self._wildcard_profile_id
        if pid is not None:
            return self._profiles[pid]

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

        When a ``key_composer`` was provided at construction time, sessions are
        keyed by the deterministic composite key (F-06).  Otherwise the legacy
        ``(channel_id, chat_id)`` tuple key is used for backward compatibility.
        """
        if self._key_composer:
            composite_key = self._key_composer.compose(profile.id, channel_id, chat_id)

            # Check in-memory by composite key
            if composite_key in self._sessions:
                return self._sessions[composite_key]

            session = Session(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                profile_id=profile.id,
                channel_id=channel_id,
            )
            self._sessions[composite_key] = session

            # Persist the composite key to Supabase (fire-and-forget)
            if self._persistence:
                self._persistence.persist_session_key(session.id, composite_key)

            logger.info(
                "router.session_created",
                session_id=session.id,
                chat_id=chat_id,
                profile=profile.id,
                channel=channel_id,
                session_key=composite_key,
            )
            self._record_event(
                "astridr", "session_start",
                f"Session started on {channel_id}",
                {
                    "session_id": session.id,
                    "profile_id": profile.id,
                    "channel_id": channel_id,
                    "session_key": composite_key,
                },
            )

            # SESSION_START hook (AP-20)
            if self._hook_registry is not None:
                asyncio.create_task(
                    self._fire_session_hook(HookPoint.SESSION_START, session.id)
                )

            # Snapshot restore: re-queue pending_tool_calls, flush memory_cache (AP-19, D-17)
            if self._snapshot_manager is not None:
                asyncio.create_task(
                    self._restore_session_snapshot(session, composite_key)
                )

            return session

        # Legacy path: tuple key (backward compat — no key_composer)
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
        self._record_event(
            "astridr", "session_start",
            f"Session started on {channel_id}",
            {"session_id": session.id, "profile_id": profile.id, "channel_id": channel_id},
        )

        # SESSION_START hook (AP-20)
        if self._hook_registry is not None:
            asyncio.create_task(
                self._fire_session_hook(HookPoint.SESSION_START, session.id)
            )

        # Snapshot restore: re-queue pending_tool_calls, flush memory_cache (AP-19, D-17)
        if self._snapshot_manager is not None:
            asyncio.create_task(
                self._restore_session_snapshot(session, session.id)
            )

        return session

    # -- Phase 44 hook + snapshot helpers ------------------------------------

    async def _fire_session_hook(self, point: Any, session_key: str) -> None:
        """Fire a SESSION_START or SESSION_END hook (AP-20)."""
        if self._hook_registry is None:
            return
        try:
            ctx = HookContext(hook_point=point, session_key=session_key)
            await self._hook_registry.fire(point, ctx)
        except Exception:
            logger.warning("router.session_hook_failed", point=str(point), exc_info=True)

    async def _restore_session_snapshot(self, session: Any, session_key: str) -> None:
        """Restore snapshot state onto session (AP-19, D-17).

        Re-queues pending_tool_calls onto session.pending_tool_calls so
        AgentLoop picks them up at process() start. Flushes memory_cache
        entries to memory store.
        """
        if self._snapshot_manager is None:
            return
        try:
            snapshot = await self._snapshot_manager.restore(session_key)
            if snapshot is None:
                return

            logger.info(
                "session.snapshot_restored",
                session_key=session_key,
                pending_tool_calls=len(snapshot.pending_tool_calls),
                memory_cache_entries=len(snapshot.memory_cache),
            )

            # Prepend restored messages so agent has prior context
            if snapshot.messages:
                session.messages = snapshot.messages + list(session.messages)

            # Store pending_tool_calls on session for AgentLoop to pick up (D-17)
            if snapshot.pending_tool_calls:
                session.pending_tool_calls = snapshot.pending_tool_calls

            # Flush memory_cache entries to memory store (D-17)
            if snapshot.memory_cache and self._memory_store is not None:
                for entry in snapshot.memory_cache:
                    try:
                        await self._memory_store.add(entry)
                    except Exception as e:
                        logger.warning(
                            "session.memory_flush_failed",
                            entry=str(entry)[:80],
                            error=str(e),
                        )

            # Restore autonomy_override from Supabase (D-06, Phase 46.3)
            if self._persistence is not None:
                try:
                    stored = await self._persistence.get_session(session.id)
                    if stored and stored.get("autonomy_override"):
                        from astridr.automation.autonomy import AutonomyLevel
                        try:
                            level = AutonomyLevel(stored["autonomy_override"])
                            agent_session = self._get_or_create_agent_session(session)
                            agent_session._autonomy_override = level
                            logger.info(
                                "autonomy_override.restored",
                                session_id=session.id,
                                level=level.value,
                            )
                        except (ValueError, KeyError):
                            pass  # Invalid stored value, skip
                except Exception:
                    logger.warning(
                        "session.autonomy_override_restore_failed",
                        session_id=session.id,
                        exc_info=True,
                    )

        except Exception:
            logger.warning(
                "session.snapshot_restore_failed",
                session_key=session_key,
                exc_info=True,
            )

    async def shutdown_snapshot_all(self) -> None:
        """Capture snapshots for all active sessions on graceful shutdown (D-14).

        Fires checkpoint_bg (fire-and-forget) for each active session.
        Called from bootstrap shutdown handler.
        """
        if self._snapshot_manager is None:
            return

        active_sessions = list(self._sessions.values())
        logger.info("router.shutdown_snapshot", session_count=len(active_sessions))

        for session in active_sessions:
            try:
                messages = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    if isinstance(m, dict) else
                    {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "") or ""}
                    for m in session.messages
                ]
                self._snapshot_manager.checkpoint_bg(
                    session_key=session.id,
                    agent_type="unknown",
                    messages=messages,
                    pending_tool_calls=[],
                    memory_cache=[],
                )
                # SESSION_END hook on shutdown
                if self._hook_registry is not None:
                    asyncio.create_task(
                        self._fire_session_hook(HookPoint.SESSION_END, session.id)
                    )
            except Exception:
                logger.warning(
                    "router.shutdown_snapshot_failed",
                    session_id=session.id,
                    exc_info=True,
                )

    def set_emergency_stop(self, active: bool, reason: str = "manual", initiator: str = "system") -> None:
        """Toggle emergency stop. When active, new messages are queued (global estop) or dropped (legacy).

        If an EmergencyStop instance is wired in, delegates to it so all active agent
        loops are cancelled and the audit log is written.  Also keeps the legacy bool
        flag in sync for backward compatibility.
        """
        self._emergency_stop = active
        if self._estop:
            if active:
                asyncio.create_task(self._estop.activate(reason, initiator))
            else:
                asyncio.create_task(self._estop.deactivate(initiator))
        logger.warning("router.emergency_stop_toggled", active=active)

    def register_channel(self, channel: BaseChannel) -> None:
        """Register a plugin channel at runtime.

        The channel must already have start() called before registering.
        Raises ValueError if channel_id is already registered.

        Per CHAN-03: registers without touching internal channel code.
        """
        cid = channel.channel_id
        if cid in self._plugin_channels:
            raise ValueError(
                f"Channel '{cid}' is already registered. "
                "Call deregister_channel() first."
            )
        self._plugin_channels[cid] = channel
        logger.info("router.plugin_channel_registered", channel_id=cid)

    def deregister_channel(self, channel_id: str) -> None:
        """Remove a plugin channel from the runtime registry."""
        removed = self._plugin_channels.pop(channel_id, None)
        if removed is not None:
            logger.info("router.plugin_channel_deregistered", channel_id=channel_id)

    @property
    def sessions(self) -> dict[Any, Session]:
        """Read-only access to active sessions (for diagnostics).

        Keys are composite strings (e.g. ``"larry::telegram::12345"``) when
        a ``key_composer`` is configured, or ``(channel_id, chat_id)`` tuples
        in legacy mode.
        """
        return dict(self._sessions)

    @property
    def plugin_channels(self) -> dict[str, BaseChannel]:
        """Read-only access to registered plugin channels (for diagnostics)."""
        return dict(self._plugin_channels)

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

            # HOOK-02: Plugin short-circuit before LLM (per D-04, D-05, D-06)
            originating_plugin = self._plugin_channels.get(message.channel_id)
            if originating_plugin is not None:
                try:
                    synthetic = await originating_plugin.before_agent_reply(message)
                    if synthetic is not None:
                        logger.info(
                            "router.before_agent_reply_short_circuit",
                            channel_id=originating_plugin.channel_id,
                        )
                        return synthetic
                except Exception:
                    logger.exception(
                        "router.before_agent_reply_error",
                        channel_id=message.channel_id,
                    )
                    # Fall through to LLM on hook error

            # HOOK-01: Per-pipe tool allowlist enforcement (per D-01, D-02, D-03)
            # Filter tool definitions BEFORE the LLM sees them (prompt-level filtering).
            # Empty/missing pipe_tools = unrestricted (all profile tools available).
            pipe_tools: list[str] = message.raw.get("pipe_tools", [])
            _pipe_override_applied = False
            if pipe_tools and message.raw.get("source") == "pipe":
                allowed = set(pipe_tools)
                filtered = [
                    t for t in (self._all_tool_defs or [])
                    if getattr(t, "name", None) in allowed
                ]
                agent_session._override_tools = filtered
                _pipe_override_applied = True
                logger.debug(
                    "router.pipe_tool_allowlist_applied",
                    pipe_name=message.raw.get("pipe_name"),
                    allowed=pipe_tools,
                    matched=len(filtered),
                )

            user_msg = Message(role="user", content=text)
            agent_response = await self._agent_loop_instance.process(user_msg, agent_session)
            response_text = str(agent_response)

            # Clear pipe tool override to prevent bleed to subsequent messages
            # on the same session (defensive — pipe sessions are typically fresh)
            if _pipe_override_applied:
                agent_session._override_tools = None

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

    async def _handle_tasks_command(self, message: IncomingMessage) -> OutgoingMessage:
        """Handle /tasks command — show active agent work inline (ORCH-07).

        Shows running flows, recent failures, and a summary count.
        Works in any channel by querying FlowRegistry.
        """
        import datetime

        if self._flow_registry is None:
            return OutgoingMessage(
                text="Task tracking not available (flow registry not initialized).",
                chat_id=message.chat_id,
            )

        running = await self._flow_registry.list_flows(status="running", limit=20)
        failed = await self._flow_registry.list_flows(status="failed", limit=5)

        parts: list[str] = []

        # Active tasks
        if running:
            parts.append(f"**Active Tasks ({len(running)})**")
            for f in running:
                started = datetime.datetime.fromtimestamp(f.started_at).strftime("%H:%M:%S")
                task_snippet = f.task[:80] if len(f.task) <= 80 else f.task[:80]
                parts.append(f"- `{f.agent_type_id}` — {task_snippet} (since {started})")
        else:
            parts.append("No active tasks.")

        # Recent failures
        if failed:
            parts.append("")
            parts.append(f"**Recent Failures ({len(failed)})**")
            for f in failed:
                err = f.error[:60] if f.error else "unknown"
                parts.append(f"- `{f.agent_type_id}` — {err}")

        # Summary line
        total_running = len(running)
        total_failed = len(failed)
        parts.append("")
        parts.append(f"_Flows: {total_running} active, {total_failed} recent failures_")

        return OutgoingMessage(
            text="\n".join(parts),
            chat_id=message.chat_id,
        )

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

    async def _handle_model_command(self, text: str, session: Session) -> str:
        """Handle /model slash command (ROUTE-05, D-07, D-08).

        Supported forms:
          /model           -- show current model
          /model <name>    -- set model for this session
          /model reset     -- clear session override, return to routing rules
        """
        parts = text.split(maxsplit=1)
        # Get the agent loop session to read/write _model_override
        agent_session = self._get_or_create_agent_session(session)

        if len(parts) == 1:
            # Show current model
            current = getattr(agent_session, "_model_override", None) or "auto (routing rules)"
            return f"Current model: `{current}`"

        model_name = parts[1].strip()

        if model_name.lower() == "reset":
            agent_session._model_override = None
            # Also clear on ModelRouter if available
            if self._model_router is not None:
                self._model_router.clear_session_override(agent_session.id)
            return "Model reset to auto (routing rules)."

        # Set session override — takes effect next turn (D-07)
        agent_session._model_override = model_name
        # Also set on ModelRouter for routing resolution
        if self._model_router is not None:
            self._model_router.set_session_override(agent_session.id, model_name)
        return f"Model set to `{model_name}` -- takes effect on your next message."

    async def _handle_autonomy_command(self, text: str, session: Session) -> str:
        """Handle /autonomy slash command (AP-09, D-05, D-06).

        Supported forms:
          /autonomy           -- show current override
          /autonomy <level>   -- set override (silent|draft_approval|always_ask|blocked)
          /autonomy reset     -- clear override, return to dynamic rules

        Override persists to Supabase session_history per D-06.
        """
        import structlog
        from astridr.automation.autonomy import AutonomyLevel

        log = structlog.get_logger("router.autonomy_command")
        parts = text.split(maxsplit=1)
        agent_session = self._get_or_create_agent_session(session)

        if len(parts) == 1:
            current = getattr(agent_session, "_autonomy_override", None)
            display = current.value if current else "none (dynamic rules apply)"
            return f"Current autonomy override: `{display}`"

        value = parts[1].strip().lower()

        if value == "reset":
            agent_session._autonomy_override = None
            # Write-through to Supabase (D-06)
            if self._persistence is not None:
                self._persistence.update_session_autonomy_override_bg(
                    session.id, None
                )
            log.info("autonomy_override.cleared", session_id=session.id)
            return "Autonomy override cleared -- dynamic rules apply."

        try:
            level = AutonomyLevel(value)
        except ValueError:
            valid = ", ".join(lv.value for lv in AutonomyLevel)
            return f"Unknown level `{value}`. Valid: {valid}"

        agent_session._autonomy_override = level
        # Write-through to Supabase (D-06)
        if self._persistence is not None:
            self._persistence.update_session_autonomy_override_bg(
                session.id, level.value
            )
        log.info("autonomy_override.set", session_id=session.id, level=level.value)
        return f"Autonomy override set to `{level.value}` for this session."

    async def _handle_ingest_command(self, message: IncomingMessage) -> OutgoingMessage:
        """Handle /ingest <url> -- scrape URL to markdown via FirecrawlTool (FIRE-01)."""
        parts = message.text.strip().split(maxsplit=1)
        if len(parts) < 2:
            return OutgoingMessage(
                text="Usage: /ingest <url>",
                chat_id=message.chat_id,
            )
        url = parts[1].strip()

        # Resolve FirecrawlTool from registry
        if self._tool_registry is None:
            return OutgoingMessage(
                text="Tool registry not available -- cannot run /ingest.",
                chat_id=message.chat_id,
            )
        tool = self._tool_registry.get("firecrawl_ingest")
        if tool is None:
            return OutgoingMessage(
                text="FirecrawlTool is not registered. Check config.firecrawl.enabled.",
                chat_id=message.chat_id,
            )

        result = await tool.execute(url=url)
        if not result.success:
            return OutgoingMessage(
                text=f"Ingest failed: {result.error}",
                chat_id=message.chat_id,
            )

        output = result.output or ""
        # Truncate if very long (markdown can be large)
        if len(output) > 4000:
            output = output[:4000] + "\n\n... [truncated -- full content available in tool output]"

        return OutgoingMessage(
            text=output,
            chat_id=message.chat_id,
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

    async def _generate_tts(self, text: str, persona_id: str | None = None) -> Attachment | None:
        """Generate a TTS voice note for *text*.

        Returns an ``Attachment`` on success, or ``None`` if TTS is
        unavailable or the generation fails (non-blocking).

        Args:
            text: The text to convert to speech.
            persona_id: Optional persona ID to resolve voice config from.
                Falls back to ELEVENLABS_VOICE_ID env var when None or unknown.
        """
        tool = self._get_tts_tool()
        if tool is None:
            return None

        try:
            voice_id, stability, similarity_boost = self._voice_resolver.resolve(persona_id)
            result = await tool.execute(
                text=text,
                voice_id=voice_id,
                stability=stability,
                similarity_boost=similarity_boost,
            )
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

    async def _send_tts_followup(
        self,
        channel: BaseChannel,
        chat_id: str,
        text: str,
        persona_id: str | None,
        reply_to_message_id: str | None,
    ) -> None:
        """Generate TTS and send as a follow-up message (non-blocking)."""
        if not text or not text.strip():
            return
        try:
            tts_attachment = await self._generate_tts(text, persona_id=persona_id)
            if tts_attachment is not None:
                followup = OutgoingMessage(
                    text="\u200b",
                    chat_id=chat_id,
                    reply_to_message_id=reply_to_message_id,
                    attachments=[tts_attachment],
                )
                await self._send_or_queue(channel, followup)
        except Exception:
            logger.exception("router.tts_followup_error")

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
