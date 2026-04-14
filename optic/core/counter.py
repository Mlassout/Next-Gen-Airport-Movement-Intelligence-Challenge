"""
counter.py — Comptage par ligne virtuelle à orientation libre.

La ligne est définie par deux points A(x1,y1) et B(x2,y2).
Le côté d'un centroïde est déterminé par le signe du produit vectoriel
AB × AP (distance perpendiculaire signée).

  côté + (vert)  = à gauche quand on va de A vers B
  côté − (orange) = à droite quand on va de A vers B

Un franchissement est comptabilisé quand le signe change et que le point
est hors de la bande de tolérance (distance perpendiculaire > tolerance px).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .tracker import TrackManager

LINE_COLOR_DEFAULT = (0, 255, 200)
LINE_COLOR_CROSS   = (0, 100, 255)
COLOR_POS = (0, 220, 0)      # vert  → côté +
COLOR_NEG = (0, 120, 255)    # orange → côté −
COLOR_GRAY = (100, 100, 100)


class VirtualLineCounter:
    """
    Compteur de passages sur une ligne à orientation libre.

    Paramètres
    ----------
    x1, y1 : int  Point A (début de la ligne).
    x2, y2 : int  Point B (fin de la ligne).
    tolerance : int
        Demi-largeur de la bande neutre en pixels perpendiculaires.
    direction : int
        +1 → compte uniquement les passages vers le côté + (vert)
        -1 → uniquement vers le côté − (orange)
         0 → les deux
    """

    def __init__(
        self,
        x1: int = 0, y1: int = 240,
        x2: int = 640, y2: int = 240,
        tolerance: int = 10,
        direction: int = 0,
    ) -> None:
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.tolerance = tolerance
        self.direction = direction
        self._count_pos: int = 0   # franchissements vers le côté +
        self._count_neg: int = 0   # franchissements vers le côté −
        self._track_manager = TrackManager()

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._count_pos + self._count_neg

    @property
    def count_pos(self) -> int:
        return self._count_pos

    @property
    def count_neg(self) -> int:
        return self._count_neg

    # ------------------------------------------------------------------
    # Géométrie
    # ------------------------------------------------------------------

    def _signed_dist(self, cx: int, cy: int) -> float:
        """
        Distance perpendiculaire signée de (cx, cy) à la ligne AB.
        Positive = côté gauche quand on va de A vers B.
        """
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = math.hypot(dx, dy)
        if length < 1:
            return 0.0
        # produit vectoriel AB × AP / |AB|
        return (dx * (cy - self.y1) - dy * (cx - self.x1)) / length

    def _normal(self) -> tuple[float, float]:
        """Vecteur unitaire perpendiculaire pointant vers le côté + ."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = math.hypot(dx, dy)
        if length < 1:
            return 0.0, 1.0
        return -dy / length, dx / length   # rotation +90°

    # ------------------------------------------------------------------
    # Mise à jour
    # ------------------------------------------------------------------

    def update(self, track_id: int, cx: int, cy: int, frame_number: int) -> int:
        """
        Met à jour et détecte les franchissements.

        Retourne :
          0  — pas de franchissement
         +1  — franchissement vers le côté + (vert)
         -1  — franchissement vers le côté − (orange)
        """
        self._track_manager.tick(frame_number)
        state = self._track_manager.update(track_id, cx, cy)

        dist = self._signed_dist(cx, cy)
        if abs(dist) <= self.tolerance:
            return 0  # bande neutre

        new_side = 1 if dist > 0 else -1
        prev_side = state.last_side
        self._track_manager.set_side(track_id, new_side)

        if prev_side is None or new_side == prev_side:
            return 0

        # Franchissement confirmé : new_side est le côté d'arrivée
        if self.direction == 0 or self.direction == new_side:
            if new_side == 1:
                self._count_pos += 1
            else:
                self._count_neg += 1
            return new_side

        return 0

    def reset(self) -> None:
        self._count_pos = 0
        self._count_neg = 0
        self._track_manager.reset()

    # ------------------------------------------------------------------
    # Rendu overlay
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, crossings: dict[int, int] | None = None) -> None:
        """
        Dessine la ligne, la bande de tolérance et les indicateurs de sens.
        """
        has_crossing = bool(crossings)
        line_color = LINE_COLOR_CROSS if has_crossing else LINE_COLOR_DEFAULT

        nx, ny = self._normal()
        tol = self.tolerance
        lx1, ly1, lx2, ly2 = self.x1, self.y1, self.x2, self.y2

        # Bande de tolérance (parallélogramme semi-transparent)
        pts = np.array([
            [int(lx1 + nx * tol), int(ly1 + ny * tol)],
            [int(lx2 + nx * tol), int(ly2 + ny * tol)],
            [int(lx2 - nx * tol), int(ly2 - ny * tol)],
            [int(lx1 - nx * tol), int(ly1 - ny * tol)],
        ], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (200, 200, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Ligne principale + points A / B
        cv2.line(frame, (lx1, ly1), (lx2, ly2), line_color, 2)
        cv2.circle(frame, (lx1, ly1), 5, (0, 150, 255), -1)
        cv2.circle(frame, (lx2, ly2), 5, (255, 100, 0), -1)

        # Flèches perpendiculaires au milieu de la ligne
        mx, my = (lx1 + lx2) // 2, (ly1 + ly2) // 2
        arr = 22

        c_pos = COLOR_POS   if self.direction in (0,  1) else COLOR_GRAY
        c_neg = COLOR_NEG   if self.direction in (0, -1) else COLOR_GRAY
        thick_pos = 2 if self.direction in (0,  1) else 1
        thick_neg = 2 if self.direction in (0, -1) else 1

        ep = (int(mx + nx * arr), int(my + ny * arr))
        cv2.arrowedLine(frame, (mx, my), ep, c_pos, thick_pos, tipLength=0.4)
        cv2.putText(frame, "+", (ep[0] + 3, ep[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_pos, 2)

        em = (int(mx - nx * arr), int(my - ny * arr))
        cv2.arrowedLine(frame, (mx, my), em, c_neg, thick_neg, tipLength=0.4)
        cv2.putText(frame, "\u2212", (em[0] + 3, em[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_neg, 2)

        # Compteur textuel (ancré au-dessus du point A)
        lbl_x = lx1 + 5
        lbl_y = max(min(ly1, ly2) - 8, 18)
        if self.direction == 0:
            label = f"+ {self._count_pos}   \u2212 {self._count_neg}   T {self.count}"
        elif self.direction == 1:
            label = f"+ {self._count_pos} passages"
        else:
            label = f"\u2212 {self._count_neg} passages"

        cv2.putText(frame, label, (lbl_x, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2, cv2.LINE_AA)
