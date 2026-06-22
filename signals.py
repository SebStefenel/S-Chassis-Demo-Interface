"""
Eclipse S-CORE signal publisher.

Publishes VSS-path signals to the Kuksa Data Broker (gRPC, port 55555).
If kuksa-client is not installed or the broker is unreachable, all
signal calls are no-ops so the monitor still runs without S-CORE.

VSS paths used:
  Vehicle.Driver.IsAuthenticated     bool
  Vehicle.Driver.Identifier          string
  Vehicle.Driver.DrowsinessLevel     uint8  (0 = alert, 10 = very drowsy)
  Vehicle.Driver.AttentionLvl        string ("ALERT" | "ATTENTIVE")
  Vehicle.Cabin.Seat.Row1.DriverSide.Position  uint8
  Vehicle.Cabin.HVAC.AmbientAirTemperature     float
"""

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)

try:
    from kuksa_client.grpc import Datapoint, VSSClient
    _KUKSA_OK = True
except ImportError:
    _KUKSA_OK = False
    log.warning("kuksa-client not installed — S-CORE signals will be logged only")


class SCOREPublisher:
    """Thin adapter between the SDV monitor and the Kuksa Data Broker."""

    def __init__(self, host: str = "127.0.0.1", port: int = 55555) -> None:
        self._client = None
        if not _KUKSA_OK:
            return
        try:
            self._client = VSSClient(host, port)
            self._client.__enter__()
            log.info("Connected to S-CORE broker at %s:%d", host, port)
        except Exception as exc:
            log.warning("S-CORE broker not reachable (%s) — signals disabled", exc)

    # ── internal ──────────────────────────────────────────────────────────────

    def _pub(self, signals: Dict[str, Any]) -> None:
        log.info("SIGNAL %s", signals)
        if self._client is None:
            return
        try:
            self._client.set_current_values({k: Datapoint(v) for k, v in signals.items()})
        except Exception as exc:
            log.warning("Signal publish error: %s", exc)

    # ── public API ────────────────────────────────────────────────────────────

    def scanning(self) -> None:
        """Emitted while AUTH state is actively scanning for a face."""
        self._pub({
            "Vehicle.Driver.IsAuthenticated": False,
            "Vehicle.Driver.Identifier": "",
        })

    def authenticated(self, name: str, settings: Dict[str, Any]) -> None:
        """Emitted when a driver is matched; loads their vehicle preferences."""
        self._pub({
            "Vehicle.Driver.IsAuthenticated": True,
            "Vehicle.Driver.Identifier": name,
        })
        if "seat_position" in settings:
            self._pub({
                "Vehicle.Cabin.Seat.Row1.DriverSide.Position": int(settings["seat_position"])
            })
        if "climate_temp_c" in settings:
            self._pub({
                "Vehicle.Cabin.HVAC.AmbientAirTemperature": float(settings["climate_temp_c"])
            })

    def unrecognized(self) -> None:
        """Emitted on auth timeout when face was seen but not matched."""
        self._pub({
            "Vehicle.Driver.IsAuthenticated": False,
            "Vehicle.Driver.Identifier": "UNKNOWN",
        })

    def drowsiness(self, ear: float, is_drowsy: bool) -> None:
        """Emitted every monitor frame with current EAR and alert state."""
        level = max(0, min(10, int((0.35 - ear) / 0.035))) if ear < 0.35 else 0
        self._pub({
            "Vehicle.Driver.DrowsinessLevel": level,
            "Vehicle.Driver.AttentionLvl": "ALERT" if is_drowsy else "ATTENTIVE",
        })

    def close(self) -> None:
        if self._client is not None:
            self._client.__exit__(None, None, None)
