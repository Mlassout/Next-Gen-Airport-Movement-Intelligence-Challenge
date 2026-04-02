"""
O.P.T.I.C — Optimisation Predictive des Trafics et Indicateurs de Congestion
EF1-EF9 | YOLOv8n + OpenCV + Streamlit
Architecture : thread background pour YOLO, fragment Streamlit pour l'affichage.
"""

import base64
import csv
import io
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ── CONSTANTES ────────────────────────────────────────────────────────────────
CONF      = 0.35   # seuil de confiance YOLO
COOL      = 2.0    # anti-rebond : délai min (secondes) entre deux comptages du même ID
SKIP      = 4      # YOLO tourne 1 frame sur SKIP ; les autres affichent les dernières boîtes

# ── MODELE ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="O.P.T.I.C", layout="wide")
st.title("O.P.T.I.C — Optimisation Prédictive des Trafics et Indicateurs de Congestion")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ── ÉTAT PARTAGÉ MODULE-LEVEL ─────────────────────────────────────────────────
# Persiste entre les reruns Streamlit (module importé une seule fois).
# Accessible depuis le thread YOLO ET depuis le fragment d'affichage.
_lock = threading.Lock()
_S: dict = {
    "running":   False,
    "frame_jpg": None,        # dernière frame encodée JPEG (bytes)
    "count_in":  0,
    "count_out": 0,
    "density":   0,
    "events":    [],
    "line": (0.0, 0.5, 1.0, 0.5),   # (lx1, ly1, lx2, ly2) en fractions [0..1]
    "zone": (0.2, 0.2, 0.8, 0.8),   # (zx1, zy1, zx2, zy2) en fractions [0..1]
}

if "started" not in st.session_state:
    st.session_state.started = False

# ── HELPERS GÉOMÉTRIQUES ──────────────────────────────────────────────────────
def line_side(px, py, x1, y1, x2, y2) -> int:
    """Signe du produit vectoriel : détermine de quel côté de la ligne se trouve le point."""
    v = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if v > 0 else (-1 if v < 0 else 0)

def on_segment(px, py, x1, y1, x2, y2) -> bool:
    """Vérifie que la projection du point tombe dans le segment (pas juste sur la droite)."""
    dx, dy = x2 - x1, y2 - y1
    sq = dx * dx + dy * dy
    if sq == 0:
        return True
    t = ((px - x1) * dx + (py - y1) * dy) / sq
    return 0.0 <= t <= 1.0

# ── DESSIN ────────────────────────────────────────────────────────────────────
def draw_overlay(frame, lx1, ly1, lx2, ly2, zx1, zy1, zx2, zy2, boxes):
    """Dessine ligne de comptage, zone de densité et bounding boxes sur la frame."""
    # Zone de densité
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 180, 255), 2)
    cv2.putText(frame, "ZONE", (zx1 + 4, zy1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    # Ligne de comptage
    cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 60, 255), 3)
    cv2.circle(frame, (lx1, ly1), 6, (0, 60, 255), -1)
    cv2.circle(frame, (lx2, ly2), 6, (0, 60, 255), -1)
    # Bounding boxes
    for bx1, by1, bx2, by2, tid, in_zone in boxes:
        c = (0, 180, 255) if in_zone else (50, 220, 50)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), c, 2)
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        cv2.circle(frame, (cx, cy), 4, c, -1)
        cv2.putText(frame, f"ID:{tid}", (bx1, max(by1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

def to_jpg(frame_bgr) -> bytes:
    """Convertit une frame BGR en JPEG bytes (pour affichage HTML base64)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()

def b64_img(jpg: bytes, width: str = "100%") -> str:
    return (f'<img src="data:image/jpeg;base64,{base64.b64encode(jpg).decode()}" '
            f'style="width:{width};border-radius:4px">')

# ── THREAD DE TRAITEMENT YOLO ─────────────────────────────────────────────────
def _worker(src):
    """
    Lit la vidéo en continu.
    - Lance YOLO toutes les SKIP frames (EF2 / EF3).
    - Entre deux YOLO, ré-affiche les dernières boîtes connues (fluidité).
    - Détecte les traversées de ligne (EF4) avec anti-rebond temporel (EF5).
    - Calcule la densité dans la zone (EF6).
    - Écrit dans _S sous _lock pour éviter les race conditions.
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        with _lock:
            _S["running"] = False
        return

    prev_side: dict[int, int]   = {}
    last_t:    dict[int, float] = {}
    boxes:     list             = []   # dernières boîtes YOLO connues
    fidx = 0

    while True:
        with _lock:
            if not _S["running"]:
                break
            lf = _S["line"]
            zf = _S["zone"]

        ret, frame = cap.read()
        if not ret:
            break

        t0   = time.time()
        h, w = frame.shape[:2]

        lx1 = int(lf[0] * w); ly1 = int(lf[1] * h)
        lx2 = int(lf[2] * w); ly2 = int(lf[3] * h)
        zx1 = int(zf[0] * w); zy1 = int(zf[1] * h)
        zx2 = int(zf[2] * w); zy2 = int(zf[3] * h)

        if fidx % SKIP == 0:
            try:
                res = model.track(frame, persist=True, classes=0,
                                  conf=CONF, imgsz=320, verbose=False)
            except Exception:
                fidx += 1
                continue

            new_events = []
            density    = 0
            boxes      = []

            if res[0].boxes.id is not None:
                for box, tid_f in zip(res[0].boxes.xyxy.cpu().numpy(),
                                      res[0].boxes.id.cpu().numpy()):
                    bx1, by1, bx2, by2 = map(int, box)
                    tid = int(tid_f)
                    cx  = (bx1 + bx2) // 2
                    cy  = (by1 + by2) // 2

                    in_zone = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                    if in_zone:
                        density += 1

                    # EF4 + EF5 : traversée de ligne avec anti-rebond
                    side = line_side(cx, cy, lx1, ly1, lx2, ly2)
                    prev = prev_side.get(tid)
                    if prev is not None:
                        lt = last_t.get(tid, 0.0)
                        if (on_segment(cx, cy, lx1, ly1, lx2, ly2)
                                and prev != 0 and side != 0 and prev != side
                                and t0 - lt > COOL):
                            d = "IN" if prev == -1 else "OUT"
                            last_t[tid] = t0
                            new_events.append((d, tid))
                    if side != 0:
                        prev_side[tid] = side

                    boxes.append((bx1, by1, bx2, by2, tid, in_zone))

            with _lock:
                _S["density"] = density
                for d, tid in new_events:
                    if d == "IN":
                        _S["count_in"]  += 1
                    else:
                        _S["count_out"] += 1
                    _S["events"].append({
                        "Horodatage": datetime.now().strftime("%H:%M:%S"),
                        "Direction":  d,
                        "ID":         tid,
                    })

        # Dessine les dernières boîtes connues sur toutes les frames
        draw_overlay(frame, lx1, ly1, lx2, ly2, zx1, zy1, zx2, zy2, boxes)
        jpg = to_jpg(frame)
        with _lock:
            _S["frame_jpg"] = jpg

        fidx += 1

    cap.release()
    with _lock:
        _S["running"] = False


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # EF1 : chargement vidéo
    src_type = st.radio("Source", ["Fichier", "Webcam"], horizontal=True, key="src_type")
    if src_type == "Fichier":
        st.text_input("Chemin vers la vidéo (.mp4)", "video.mp4", key="video_path")

    st.divider()
    st.subheader("Ligne de comptage")
    st.caption("Définit l'axe de passage — EF4")
    ca, cb = st.columns(2)
    with ca:
        lx1_v = st.slider("X1 %", 0, 100,   0, key="lx1")
        ly1_v = st.slider("Y1 %", 0, 100,  50, key="ly1")
    with cb:
        lx2_v = st.slider("X2 %", 0, 100, 100, key="lx2")
        ly2_v = st.slider("Y2 %", 0, 100,  50, key="ly2")

    st.divider()
    st.subheader("Zone de densité")
    st.caption("Comptage des personnes dans la zone — EF6")
    za, zb = st.columns(2)
    with za:
        zx1_v = st.slider("Gauche %", 0, 100, 20, key="zx1")
        zy1_v = st.slider("Haut %",   0, 100, 20, key="zy1")
    with zb:
        zx2_v = st.slider("Droite %", 0, 100, 80, key="zx2")
        zy2_v = st.slider("Bas %",    0, 100, 80, key="zy2")

    # Sync des params dans l'état partagé (le thread les lira à chaque frame)
    with _lock:
        _S["line"] = (lx1_v / 100, ly1_v / 100, lx2_v / 100, ly2_v / 100)
        _S["zone"] = (zx1_v / 100, zy1_v / 100, zx2_v / 100, zy2_v / 100)

    st.divider()
    st.subheader("Seuils d'alerte")
    st.caption("EF7")
    thr_o = st.slider("Orange (pers.)", 1, 30,  5, key="thr_o")
    thr_r = st.slider("Rouge (pers.)",  1, 50, 15, key="thr_r")

    st.divider()
    b1, b2, b3 = st.columns(3)
    start_btn = b1.button("Start", use_container_width=True)
    stop_btn  = b2.button("Stop",  use_container_width=True)
    reset_btn = b3.button("Reset", use_container_width=True)


# ── ACTIONS BOUTONS ───────────────────────────────────────────────────────────
if start_btn:
    with _lock:
        already = _S["running"]
    if not already:
        src = (0 if st.session_state.src_type == "Webcam"
               else st.session_state.get("video_path", "video.mp4"))
        with _lock:
            _S["running"] = True
        threading.Thread(target=_worker, args=(src,), daemon=True).start()
        st.session_state.started = True

if stop_btn:
    with _lock:
        _S["running"] = False
    st.session_state.started = False

if reset_btn:
    with _lock:
        _S.update({
            "running": False, "frame_jpg": None,
            "count_in": 0, "count_out": 0,
            "density": 0, "events": [],
        })
    st.session_state.started = False


# ── FRAGMENT D'AFFICHAGE — EF8 ────────────────────────────────────────────────
# run_every=0.1 → le fragment se rafraîchit 10x/s.
# Il ne fait AUCUN traitement lourd : lit juste _S et affiche.
@st.fragment(run_every=0.1)
def live():
    with _lock:
        jpg     = _S["frame_jpg"]
        cin     = _S["count_in"]
        cout    = _S["count_out"]
        density = _S["density"]
        running = _S["running"]
        lf      = _S["line"]
        zf      = _S["zone"]

    thr_o = st.session_state.get("thr_o", 5)
    thr_r = st.session_state.get("thr_r", 15)

    vid_col, stat_col = st.columns([3, 1])

    # ── indicateurs (EF8 / EF7) ──
    with stat_col:
        st.subheader("Indicateurs")
        st.metric("Entrées",  cin)
        st.metric("Sorties",  cout)
        st.metric("Flux net", cin - cout)
        st.metric("En zone",  density)

        if density >= thr_r:
            cbg, lvl = "#b71c1c", "ROUGE"
        elif density >= thr_o:
            cbg, lvl = "#bf360c", "ORANGE"
        else:
            cbg, lvl = "#1b5e20", "VERT"
        st.markdown(
            f'<div style="background:{cbg};border-radius:6px;padding:14px;'
            f'text-align:center;color:white;font-weight:bold;font-size:1.1em;margin-top:8px">'
            f'{lvl} — {density} pers.</div>',
            unsafe_allow_html=True)

    # ── flux vidéo annoté (EF8) ──
    with vid_col:
        if jpg is not None:
            st.markdown(b64_img(jpg), unsafe_allow_html=True)
        else:
            # Apercu statique : montre la position de la ligne et de la zone
            bg   = np.zeros((360, 640, 3), dtype=np.uint8)
            bh, bw = bg.shape[:2]
            draw_overlay(
                bg,
                int(lf[0]*bw), int(lf[1]*bh), int(lf[2]*bw), int(lf[3]*bh),
                int(zf[0]*bw), int(zf[1]*bh), int(zf[2]*bw), int(zf[3]*bh),
                [])
            preview_jpg = to_jpg(bg)
            lbl = "Démarrage en cours..." if running else "Aperçu — cliquez sur Start"
            st.markdown(
                b64_img(preview_jpg) +
                f'<p style="color:#aaa;font-size:0.8em;text-align:center;margin:4px 0">{lbl}</p>',
                unsafe_allow_html=True)

        if not running and st.session_state.started:
            st.success("Analyse terminée.")


live()


# ── EXPORT — EF9 ─────────────────────────────────────────────────────────────
with _lock:
    events_snap = list(_S["events"])
    last_jpg    = _S["frame_jpg"]

if events_snap or last_jpg:
    st.divider()
    st.subheader("Export")
    col_a, col_b = st.columns(2)

    with col_a:
        if events_snap:
            st.dataframe(pd.DataFrame(events_snap), use_container_width=True)
            buf = io.StringIO()
            wr  = csv.DictWriter(buf, fieldnames=["Horodatage", "Direction", "ID"])
            wr.writeheader()
            wr.writerows(events_snap)
            st.download_button(
                "Télécharger CSV",
                buf.getvalue(),
                f"passages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True)

    with col_b:
        if last_jpg:
            st.markdown(b64_img(last_jpg, width="100%"), unsafe_allow_html=True)
            st.download_button(
                "Télécharger capture JPG",
                last_jpg,
                f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                "image/jpeg",
                use_container_width=True)
