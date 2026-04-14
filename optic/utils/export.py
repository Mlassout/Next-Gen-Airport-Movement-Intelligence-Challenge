"""
export.py — Export de captures PNG et d'extraits vidéo MP4 annotés.

Deux modes :
  - capture_frame()  : sauvegarde la frame courante en PNG (sans stockage d'identité)
  - VideoBuffer      : bufférise des frames annotées pour export MP4 (max_seconds limité)
"""

from __future__ import annotations

import csv
import io
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def to_csv_bytes(rows: list[dict]) -> bytes:
    """
    Convertit une liste de dicts en bytes CSV UTF-8.

    Retourne b"" si la liste est vide.
    """
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def capture_frame(frame: np.ndarray) -> bytes:
    """
    Encode la frame annotée en PNG et retourne les bytes.

    Utilisation Streamlit :
        png_bytes = capture_frame(annotated_frame)
        st.download_button("Télécharger PNG", png_bytes, "capture.png", "image/png")
    """
    success, buffer = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("Échec de l'encodage PNG")
    return buffer.tobytes()


class VideoBuffer:
    """
    Buffer circulaire de frames annotées pour export MP4.

    Paramètres
    ----------
    fps : float
        FPS de la vidéo source (utilisé pour le timestamp de sortie).
    max_seconds : int
        Durée maximale de l'extrait bufférisé (défaut : 60 s).
    """

    def __init__(self, fps: float = 25.0, max_seconds: int = 60) -> None:
        self.fps = fps
        max_frames = int(fps * max_seconds)
        self._buffer: deque[np.ndarray] = deque(maxlen=max_frames)

    def push(self, frame: np.ndarray) -> None:
        """Ajoute une frame au buffer (copie pour éviter les mutations)."""
        self._buffer.append(frame.copy())

    def export_mp4(self) -> bytes:
        """
        Encode le contenu du buffer en MP4 (H.264 / fallback MJPEG)
        et retourne les bytes.

        Retourne b"" si le buffer est vide.
        """
        if not self._buffer:
            return b""

        h, w = self._buffer[0].shape[:2]
        buf = io.BytesIO()

        # Écriture dans un fichier temporaire en mémoire via un path /tmp
        tmp_path = f"/tmp/optic_export_{int(time.time())}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, self.fps, (w, h))

        for frame in self._buffer:
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()

        Path(tmp_path).unlink(missing_ok=True)
        return data

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    @property
    def duration_seconds(self) -> float:
        return len(self._buffer) / max(self.fps, 1.0)
