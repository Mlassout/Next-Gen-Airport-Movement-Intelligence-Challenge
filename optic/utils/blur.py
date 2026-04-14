"""
blur.py — Floutage RGPD des zones visage/tête.

Stratégie : le haut du bounding box person (~20 % supérieur) est considéré
comme la zone tête. Un flou gaussien y est appliqué.
Aucune image brute n'est conservée.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.detector import Detection


def blur_heads(
    frame: np.ndarray,
    detections: list[Detection],
    ksize: int = 31,
) -> np.ndarray:
    """
    Applique un flou gaussien sur la zone tête de chaque détection.

    Paramètres
    ----------
    frame   : image BGR à modifier (copie interne — l'original n'est pas altéré).
    detections : liste des détections avec leur propriété head_box.
    ksize   : taille du kernel gaussien (impair, plus grand = flou plus fort).

    Retourne
    --------
    Frame avec les têtes floutées.
    """
    result = frame.copy()
    k = ksize if ksize % 2 == 1 else ksize + 1  # forcer impair

    for det in detections:
        x1, y1, x2, y2 = det.head_box

        # Clamp aux dimensions de la frame
        h, w = result.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = result[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        result[y1:y2, x1:x2] = blurred

    return result
