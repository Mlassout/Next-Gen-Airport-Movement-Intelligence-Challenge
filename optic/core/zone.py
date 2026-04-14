"""
zone.py — Calcul de la densité de personnes dans une zone rectangulaire.

La zone est définie par deux coins (x1, y1) et (x2, y2) dans l'espace image.
La densité est le nombre de centroïdes présents dans ce rectangle à chaque frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

# Couleurs overlay (BGR)
ZONE_COLOR_LOW = (0, 200, 0)       # Vert
ZONE_COLOR_MED = (0, 140, 255)     # Orange
ZONE_COLOR_HIGH = (0, 0, 220)      # Rouge


class DensityZone:
    """
    Zone rectangulaire de comptage de densité.

    Paramètres
    ----------
    x1, y1 : int  Coin supérieur gauche.
    x2, y2 : int  Coin inférieur droit.
    """

    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 100, y2: int = 100) -> None:
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self._current_count: int = 0

    @property
    def count(self) -> int:
        return self._current_count

    def update(self, detections: list[Detection]) -> int:
        """
        Compte les personnes dont le centroïde est dans la zone.

        Retourne le nombre courant.
        """
        count = 0
        for det in detections:
            cx, cy = det.centroid
            if self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2:
                count += 1
        self._current_count = count
        return count

    def contains(self, cx: int, cy: int) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def draw(self, frame: np.ndarray, alert_color: tuple[int, int, int] | None = None) -> None:
        """
        Dessine le rectangle de zone avec son compteur (in-place).

        alert_color : couleur BGR à utiliser (ou None pour couleur par défaut).
        """
        color = alert_color if alert_color else ZONE_COLOR_LOW

        # Rectangle semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x1, self.y1), (self.x2, self.y2), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Contour
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, 2)

        # Étiquette
        label = f"Zone : {self._current_count}"
        cv2.putText(
            frame, label,
            (self.x1 + 6, self.y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
