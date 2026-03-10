"""Wake word detection backends for the voice channel.

Supports multiple backends with automatic fallback:
1. OpenWakeWord — open-source, no API key needed
2. Porcupine — Picovoice, needs PICOVOICE_ACCESS_KEY
3. Energy fallback — simple energy-threshold VAD (current behavior)
"""

from __future__ import annotations

import os
import struct
from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger()


class WakeWordDetector(ABC):
    """Abstract base class for wake word detectors.

    Implementations process PCM audio chunks and return True
    when the wake word is detected.
    """

    @abstractmethod
    def process_audio(self, pcm_chunk: bytes) -> bool:
        """Process a chunk of 16-bit signed PCM audio.

        Args:
            pcm_chunk: Raw 16-bit signed little-endian mono PCM bytes.

        Returns:
            True if the wake word was detected in this chunk.
        """
        ...

    def cleanup(self) -> None:
        """Release resources held by the detector."""
        pass


class EnergyFallbackDetector(WakeWordDetector):
    """Energy-based voice activity detector (fallback).

    Detects speech onset when RMS energy exceeds a threshold.
    This does NOT detect a specific wake word — it triggers on
    any loud audio.

    Args:
        energy_threshold: RMS energy level above which speech is detected.
    """

    def __init__(self, energy_threshold: float = 500.0) -> None:
        self._threshold = energy_threshold

    def process_audio(self, pcm_chunk: bytes) -> bool:
        """Return True if chunk energy exceeds threshold (speech detected)."""
        if not pcm_chunk:
            return False
        num_samples = len(pcm_chunk) // 2
        if num_samples == 0:
            return False
        samples = struct.unpack(f"<{num_samples}h", pcm_chunk[: num_samples * 2])
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / num_samples) ** 0.5
        return rms >= self._threshold


class OpenWakeWordDetector(WakeWordDetector):
    """Wake word detector using the openwakeword library.

    Args:
        wake_word: Wake word model name (e.g. "hey_jarvis").
        sensitivity: Detection sensitivity (0.0–1.0).
        custom_model_path: Optional path to a custom .onnx model.
    """

    def __init__(
        self,
        wake_word: str = "hey_jarvis",
        sensitivity: float = 0.5,
        custom_model_path: str | None = None,
    ) -> None:
        import openwakeword  # noqa: F401
        from openwakeword.model import Model

        model_kwargs: dict = {"inference_framework": "onnx"}
        if custom_model_path:
            model_kwargs["wakeword_models"] = [custom_model_path]

        self._model = Model(**model_kwargs)
        self._wake_word = wake_word
        self._sensitivity = sensitivity

        logger.info(
            "wake_word.openwakeword_loaded",
            wake_word=wake_word,
            sensitivity=sensitivity,
        )

    def process_audio(self, pcm_chunk: bytes) -> bool:
        """Process audio through openwakeword and check for detection."""
        import numpy as np

        num_samples = len(pcm_chunk) // 2
        if num_samples == 0:
            return False

        audio_array = np.frombuffer(pcm_chunk[:num_samples * 2], dtype=np.int16)
        predictions = self._model.predict(audio_array)

        for key, score in predictions.items():
            if score >= self._sensitivity:
                logger.debug(
                    "wake_word.detected",
                    model=key,
                    score=score,
                )
                return True
        return False

    def cleanup(self) -> None:
        """Reset the openwakeword model."""
        if hasattr(self, "_model"):
            self._model.reset()


class PorcupineDetector(WakeWordDetector):
    """Wake word detector using Picovoice Porcupine.

    Requires the PICOVOICE_ACCESS_KEY environment variable.

    Args:
        wake_word: Built-in keyword name (e.g. "jarvis", "computer").
        sensitivity: Detection sensitivity (0.0–1.0).
        custom_model_path: Path to a custom .ppn keyword file trained
            via Picovoice Console (overrides wake_word).
    """

    def __init__(
        self,
        wake_word: str = "jarvis",
        sensitivity: float = 0.5,
        custom_model_path: str | None = None,
    ) -> None:
        import pvporcupine

        access_key = os.environ["PICOVOICE_ACCESS_KEY"]

        create_kwargs: dict = {
            "access_key": access_key,
            "sensitivities": [sensitivity],
        }
        if custom_model_path:
            create_kwargs["keyword_paths"] = [custom_model_path]
        else:
            create_kwargs["keywords"] = [wake_word]

        self._porcupine = pvporcupine.create(**create_kwargs)
        self._frame_length = self._porcupine.frame_length

        logger.info(
            "wake_word.porcupine_loaded",
            wake_word=wake_word,
            sensitivity=sensitivity,
        )

    def process_audio(self, pcm_chunk: bytes) -> bool:
        """Process audio through Porcupine and check for detection."""
        num_samples = len(pcm_chunk) // 2
        if num_samples < self._frame_length:
            return False

        samples = struct.unpack(f"<{self._frame_length}h", pcm_chunk[: self._frame_length * 2])
        keyword_index = self._porcupine.process(list(samples))
        if keyword_index >= 0:
            logger.debug("wake_word.porcupine_detected", keyword_index=keyword_index)
            return True
        return False

    def cleanup(self) -> None:
        """Release Porcupine resources."""
        if hasattr(self, "_porcupine"):
            self._porcupine.delete()


def create_wake_word_detector(
    wake_word: str = "astridr",
    sensitivity: float = 0.5,
    backend: str = "auto",
    custom_model_path: str | None = None,
    energy_threshold: float = 500.0,
) -> WakeWordDetector:
    """Create a wake word detector with automatic fallback.

    Tries backends in order: openwakeword -> porcupine -> energy fallback.
    The ``backend`` parameter can force a specific backend.

    Args:
        wake_word: Wake word to detect.
        sensitivity: Detection sensitivity (0.0-1.0).
        backend: "auto", "openwakeword", "porcupine", or "energy".
        custom_model_path: Optional path to a custom model file.
        energy_threshold: Threshold for the energy fallback detector.

    Returns:
        A :class:`WakeWordDetector` instance.
    """
    if backend == "energy":
        logger.info("wake_word.using_energy_fallback", reason="explicit")
        return EnergyFallbackDetector(energy_threshold=energy_threshold)

    if backend in ("auto", "openwakeword"):
        try:
            return OpenWakeWordDetector(
                wake_word=wake_word,
                sensitivity=sensitivity,
                custom_model_path=custom_model_path,
            )
        except ImportError:
            logger.debug("wake_word.openwakeword_unavailable")
        except Exception as exc:
            logger.warning("wake_word.openwakeword_failed", error=str(exc))

    if backend in ("auto", "porcupine"):
        try:
            return PorcupineDetector(
                wake_word=wake_word,
                sensitivity=sensitivity,
                custom_model_path=custom_model_path,
            )
        except ImportError:
            logger.debug("wake_word.porcupine_unavailable")
        except KeyError:
            logger.debug("wake_word.porcupine_no_access_key")
        except Exception as exc:
            logger.warning("wake_word.porcupine_failed", error=str(exc))

    logger.info("wake_word.using_energy_fallback", reason="no_backend_available")
    return EnergyFallbackDetector(energy_threshold=energy_threshold)
