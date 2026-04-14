"""
detector.py — Pipeline YOLOv8 pour la détection et le tracking de personnes.

Encapsule le modèle Ultralytics YOLOv8n et expose une interface simple :
  - inférence sur une frame BGR (numpy array)
  - retour de bounding boxes + IDs de tracking + image annotée
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Classe COCO correspondant à "person"
PERSON_CLASS_ID = 0

# Couleur bbox par défaut (BGR)
BOX_COLOR = (0, 200, 255)
ID_COLOR = (255, 255, 255)


@dataclass
class Detection:
    """Résultat de détection pour une seule personne sur une frame."""
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.cx, self.cy)

    @property
    def head_box(self) -> tuple[int, int, int, int]:
        """Retourne la boîte englobante de la zone tête (~20% supérieur de la bbox)."""
        head_height = max(10, int((self.y2 - self.y1) * 0.20))
        return (self.x1, self.y1, self.x2, self.y1 + head_height)


class Detector:
    """
    Wrapper YOLOv8 avec mode tracking persist.

    Paramètres
    ----------
    model_path : str
        Chemin ou nom du modèle (ex: 'yolov8n.pt'). Téléchargement auto si absent.
    conf_threshold : float
        Seuil de confiance minimum pour conserver une détection.
    inference_size : int
        Dimension de redimensionnement avant inférence (pour CPU, 320 ou 416 conseillé).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        inference_size: int = 416,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.inference_size = inference_size
        self._frame_count = 0

    def detect_and_track(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> tuple[list[Detection], np.ndarray]:
        """
        Lance la détection + tracking sur une frame BGR.

        Retourne
        --------
        detections : list[Detection]
            Liste des personnes détectées avec leur ID de tracking.
        annotated : np.ndarray
            Frame BGR avec les bounding boxes et IDs dessinés.
        """
        self._frame_count += 1
        annotated = frame.copy()

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=self.conf_threshold,
                imgsz=self.inference_size,
                verbose=False,
                device="cpu",
            )
        except Exception:
            return [], annotated

        detections: list[Detection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                # ID de tracking (None si tracking perdu)
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())
                else:
                    track_id = -(i + 1)  # ID temporaire négatif

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].item())

                det = Detection(
                    track_id=track_id,
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    confidence=conf,
                )
                detections.append(det)

                if draw:
                    _draw_detection(annotated, det)

        return detections, annotated


def _draw_detection(frame: np.ndarray, det: Detection) -> None:
    """Dessine bbox + ID + centroïde sur la frame (in-place)."""
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), BOX_COLOR, 2)

    label = f"#{det.track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (det.x1, det.y1 - th - 6), (det.x1 + tw + 4, det.y1), BOX_COLOR, -1)
    cv2.putText(
        frame, label,
        (det.x1 + 2, det.y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ID_COLOR, 1, cv2.LINE_AA,
    )

    cv2.circle(frame, det.centroid, 4, (0, 255, 0), -1)
