"""
alert.py — Logique de niveaux d'alerte basée sur la densité en zone.

Trois niveaux :
  🟢 VERT   : densité < seuil_orange
  🟠 ORANGE : seuil_orange <= densité < seuil_rouge
  🔴 ROUGE  : densité >= seuil_rouge
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    GREEN = "green"
    ORANGE = "orange"
    RED = "red"


@dataclass
class AlertConfig:
    threshold_orange: int = 8
    threshold_red: int = 15


# Correspondances BGR pour OpenCV
ALERT_BGR: dict[AlertLevel, tuple[int, int, int]] = {
    AlertLevel.GREEN: (0, 200, 0),
    AlertLevel.ORANGE: (0, 140, 255),
    AlertLevel.RED: (0, 0, 220),
}

# Correspondances emoji + libellé pour l'UI Streamlit
ALERT_DISPLAY: dict[AlertLevel, dict] = {
    AlertLevel.GREEN: {"emoji": "🟢", "label": "NORMAL", "color": "#00c800"},
    AlertLevel.ORANGE: {"emoji": "🟠", "label": "ÉLEVÉ", "color": "#ff8c00"},
    AlertLevel.RED: {"emoji": "🔴", "label": "CRITIQUE", "color": "#dc0000"},
}


class AlertEngine:
    """
    Calcule le niveau d'alerte à partir du comptage en zone.

    Paramètres
    ----------
    config : AlertConfig
        Seuils configurables (défauts : orange=8, rouge=15).
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig()

    def evaluate(self, zone_count: int) -> AlertLevel:
        """Retourne le niveau d'alerte pour un comptage donné."""
        if zone_count >= self.config.threshold_red:
            return AlertLevel.RED
        if zone_count >= self.config.threshold_orange:
            return AlertLevel.ORANGE
        return AlertLevel.GREEN

    def get_bgr(self, zone_count: int) -> tuple[int, int, int]:
        return ALERT_BGR[self.evaluate(zone_count)]

    def get_display(self, zone_count: int) -> dict:
        """Retourne le dict d'affichage (emoji, label, color hex) pour Streamlit."""
        return ALERT_DISPLAY[self.evaluate(zone_count)]
