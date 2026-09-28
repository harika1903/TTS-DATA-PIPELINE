"""Client for the team's custom STT server (approach B).

The server API (confirmed by testing):
    POST http://<host>:<port>/transcribe?language=<lang>
    body: multipart/form-data with `file` = the audio file
    returns JSON:
      {
        "language": "hindi",
        "audio_seconds": 15.0,
        "transcript": "the full transcribed text",
        "segments": ["text piece 1", "text piece 2", ...]
      }

CRITICAL DIFFERENCE FROM WHISPER: this server returns TEXT ONLY, with NO
word-level or segment-level timestamps. That is why this pipeline uses
silence-based cutting (VAD finds the cut points) instead of
timestamp-based cutting: we cut the audio into clips FIRST (at silence),
then send each already-short clip here to get its text. Each clip's whole
transcript is paired with that clip's whole audio.

This module is written defensively: network failures, timeouts, and bad
responses are caught and returned as an error result rather than crashing
the whole run (one bad clip must not kill processing of the rest).
"""
from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional

logger = logging.getLogger("tts_pipeline.custom_stt")

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


@dataclasses.dataclass
class SttResult:
    ok: bool
    transcript: str = ""
    segments: Optional[List[str]] = None
    detected_language: str = ""
    audio_seconds: float = 0.0
    error: str = ""


def transcribe_clip(
    wav_path: str,
    server_url: str,
    language: str,
    timeout_sec: float = 120.0,
) -> SttResult:
    """Send one audio clip to the custom STT server and return its text.

    `server_url` is the base, e.g. "http://0.0.0.0:8123". `language` is the
    server's language string, e.g. "hindi" (mapped from the filename prefix
    by languages.py). Never raises -- returns ok=False on any failure so the
    caller can quarantine/skip that clip and continue.
    """
    if requests is None:
        return SttResult(ok=False, error="the 'requests' library is not installed (pip install requests)")

    endpoint = server_url.rstrip("/") + "/transcribe"
    try:
        with open(wav_path, "rb") as f:
            files = {"file": (wav_path.split("/")[-1], f, "audio/wav")}
            params = {"language": language}
            resp = requests.post(endpoint, params=params, files=files, timeout=timeout_sec)
    except requests.exceptions.Timeout:
        return SttResult(ok=False, error=f"STT server timed out after {timeout_sec}s")
    except requests.exceptions.ConnectionError as e:
        return SttResult(ok=False, error=f"could not connect to STT server at {endpoint}: {e}")
    except Exception as e:  # noqa: BLE001
        return SttResult(ok=False, error=f"STT request failed: {type(e).__name__}: {e}")

    if resp.status_code != 200:
        return SttResult(ok=False, error=f"STT server returned HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return SttResult(ok=False, error=f"STT server returned non-JSON response: {resp.text[:200]} ({e})")

    # The confirmed schema has a "transcript" field. Be tolerant of a couple
    # of common alternative key names just in case, but prefer "transcript".
    transcript = None
    if isinstance(data, dict):
        transcript = data.get("transcript")
        if transcript is None:
            transcript = data.get("text")  # fallback
    elif isinstance(data, str):
        transcript = data  # server returned a bare string

    if transcript is None:
        return SttResult(ok=False, error=f"could not find transcript in STT response: {str(data)[:200]}")

    segments = data.get("segments") if isinstance(data, dict) else None
    detected = data.get("language", "") if isinstance(data, dict) else ""
    audio_sec = data.get("audio_seconds", 0.0) if isinstance(data, dict) else 0.0

    return SttResult(
        ok=True,
        transcript=transcript.strip(),
        segments=segments,
        detected_language=detected,
        audio_seconds=float(audio_sec) if audio_sec else 0.0,
    )


def check_server(server_url: str, timeout_sec: float = 10.0) -> tuple[bool, str]:
    """Quick reachability check before a run. Returns (ok, message)."""
    if requests is None:
        return False, "the 'requests' library is not installed (pip install requests)"
    try:
        resp = requests.get(server_url.rstrip("/") + "/", timeout=timeout_sec)
        return True, f"server reachable (HTTP {resp.status_code})"
    except Exception as e:  # noqa: BLE001
        return False, f"server not reachable at {server_url}: {type(e).__name__}: {e}"
