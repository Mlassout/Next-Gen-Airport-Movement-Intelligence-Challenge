"""
tracker.py — Gestion des états de tracking inter-frames.

Centralise l'historique des positions pour chaque ID de tracking :
  - dernière position connue
  - côté de la ligne virtuelle (pour anti-rebond)
  - frame de dernière mise à jour (pour nettoyer les IDs disparus)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackState:
    """État courant d'un objet tracké."""
    track_id: int
    cx: int
    cy: int
    last_side: Optional[int] = None   # -1 = au-dessus de la ligne, +1 = en-dessous
    last_frame: int = 0
    counted: bool = False             # a-t-il déjà été compté dans ce sens ?


class TrackManager:
    """
    Registre léger des états de tracking.

    Paramètres
    ----------
    max_missing_frames : int
        Nombre de frames consécutives sans détection avant de supprimer l'état.
        Permet d'éviter une fuite mémoire sur longue vidéo.
    """

    def __init__(self, max_missing_frames: int = 30) -> None:
        self._states: dict[int, TrackState] = {}
        self.max_missing_frames = max_missing_frames
        self._current_frame: int = 0

    def update(self, track_id: int, cx: int, cy: int) -> TrackState:
        """Met à jour ou crée l'état d'un objet tracké."""
        if track_id not in self._states:
            self._states[track_id] = TrackState(
                track_id=track_id,
                cx=cx,
                cy=cy,
                last_frame=self._current_frame,
            )
        else:
            state = self._states[track_id]
            state.cx = cx
            state.cy = cy
            state.last_frame = self._current_frame
        return self._states[track_id]

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    def set_side(self, track_id: int, side: int) -> None:
        if track_id in self._states:
            self._states[track_id].last_side = side

    def tick(self, frame_number: int) -> None:
        """Avance le compteur de frames et nettoie les IDs disparus."""
        self._current_frame = frame_number
        stale = [
            tid for tid, s in self._states.items()
            if frame_number - s.last_frame > self.max_missing_frames
        ]
        for tid in stale:
            del self._states[tid]

    def reset(self) -> None:
        """Remet à zéro tous les états (nouvelle vidéo / nouvel analysé)."""
        self._states.clear()
        self._current_frame = 0

    @property
    def active_ids(self) -> list[int]:
        return list(self._states.keys())
