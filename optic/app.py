"""
app.py — O.P.T.I.C. : Optimized Passenger Traffic Intelligence & Control
Point d'entrée Streamlit.

Lancement : streamlit run app.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from core.detector import Detector
from core.counter import VirtualLineCounter
from core.zone import DensityZone
from core.alert import AlertEngine, AlertConfig
from utils.blur import blur_heads
from utils.export import capture_frame, VideoBuffer, to_csv_bytes

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="O.P.T.I.C.",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.alert-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 8px;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    text-align: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults = {
        "running": False,
        "frame_count": 0,
        "count_pos": 0,
        "count_neg": 0,
        "zone_count": 0,
        "fps": 0.0,
        "capture_bytes": None,
        "export_bytes": None,
        "csv_frames_bytes": None,
        "csv_events_bytes": None,
        "last_error": "",
        "preview_frame": None,
        "preview_w": 640,
        "preview_h": 480,
        "_vid_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ---------------------------------------------------------------------------
# Preview frame loaders
# ---------------------------------------------------------------------------
def _load_preview(uploaded) -> None:
    cache_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state._vid_key == cache_key:
        return
    tmp = Path("/tmp/optic_preview.video")
    tmp.write_bytes(uploaded.getvalue())
    cap = cv2.VideoCapture(str(tmp))
    ret, frame = cap.read()
    cap.release()
    tmp.unlink(missing_ok=True)
    if ret:
        h, w = frame.shape[:2]
        st.session_state.preview_frame = frame
        st.session_state.preview_w = w
        st.session_state.preview_h = h
        st.session_state._vid_key = cache_key


def _capture_webcam_preview(cam_index: int) -> bool:
    """Lit une frame depuis la webcam et la stocke comme preview. Retourne True si OK."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return False
    # Laisser le temps à la caméra de s'initialiser (exposition auto)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret:
        h, w = frame.shape[:2]
        st.session_state.preview_frame = frame
        st.session_state.preview_w = w
        st.session_state.preview_h = h
        st.session_state._vid_key = f"webcam_{cam_index}"
    return ret

# ---------------------------------------------------------------------------
# Overlay helper (utilisé pour la preview et pour le dessin sur la boucle)
# ---------------------------------------------------------------------------
def _draw_config_overlay(
    frame: np.ndarray,
    lx1: int, ly1: int, lx2: int, ly2: int, tol: int, direction: int,
    zx1: int, zy1: int, zx2: int, zy2: int,
) -> np.ndarray:
    """Dessine ligne (avec côtés +/−) et zone sur une frame. Retourne une copie."""
    out = frame.copy()

    # --- Zone ---
    if zx2 > zx1 and zy2 > zy1:
        zov = out.copy()
        cv2.rectangle(zov, (zx1, zy1), (zx2, zy2), (0, 200, 0), -1)
        cv2.addWeighted(zov, 0.12, out, 0.88, 0, out)
        cv2.rectangle(out, (zx1, zy1), (zx2, zy2), (0, 200, 0), 2)
        cv2.putText(out, f"Zone  {zx2-zx1}\u00d7{zy2-zy1} px",
                    (zx1 + 5, zy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

    # --- Ligne ---
    dx, dy = lx2 - lx1, ly2 - ly1
    length = math.hypot(dx, dy)
    if length < 1:
        return out

    # Vecteur normal vers le côté +
    nx, ny = -dy / length, dx / length

    # Bande de tolérance
    pts = np.array([
        [int(lx1 + nx * tol), int(ly1 + ny * tol)],
        [int(lx2 + nx * tol), int(ly2 + ny * tol)],
        [int(lx2 - nx * tol), int(ly2 - ny * tol)],
        [int(lx1 - nx * tol), int(ly1 - ny * tol)],
    ], dtype=np.int32)
    tov = out.copy()
    cv2.fillPoly(tov, [pts], (200, 200, 0))
    cv2.addWeighted(tov, 0.18, out, 0.82, 0, out)

    # Ligne
    cv2.line(out, (lx1, ly1), (lx2, ly2), (0, 255, 200), 2)

    # Points A et B
    cv2.circle(out, (lx1, ly1), 6, (0, 150, 255), -1)
    cv2.putText(out, "A", (lx1 + 8, ly1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
    cv2.circle(out, (lx2, ly2), 6, (255, 100, 0), -1)
    cv2.putText(out, "B", (lx2 + 8, ly2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    # Flèches perpendiculaires (côtés + et −)
    mx, my = (lx1 + lx2) // 2, (ly1 + ly2) // 2
    arr = 28
    GRAY = (100, 100, 100)

    # Côté + (vert)
    ep = (int(mx + nx * arr), int(my + ny * arr))
    c, th = ((0, 220, 0), 2) if direction in (0, 1) else (GRAY, 1)
    cv2.arrowedLine(out, (mx, my), ep, c, th, tipLength=0.35)
    cv2.putText(out, "+", (ep[0] + 4, ep[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    # Côté − (orange)
    em = (int(mx - nx * arr), int(my - ny * arr))
    c, th = ((0, 120, 255), 2) if direction in (0, -1) else (GRAY, 1)
    cv2.arrowedLine(out, (mx, my), em, c, th, tipLength=0.35)
    cv2.putText(out, "\u2212", (em[0] + 4, em[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    return out

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ O.P.T.I.C.")
    st.caption("Analyse vidéo · Flux passagers")
    st.divider()

    # --- Source ---
    st.subheader("Source")
    source_type = st.radio(
        "Source vidéo",
        options=["📁 Fichier vidéo", "📷 Webcam"],
        horizontal=True,
        key="source_type",
        label_visibility="collapsed",
    )
    is_webcam = source_type == "📷 Webcam"

    uploaded   = None
    cam_index  = 0

    if not is_webcam:
        uploaded = st.file_uploader(
            "Charger .mp4 / .avi",
            type=["mp4", "avi", "mov"],
            help="Traité localement — aucune donnée envoyée.",
        )
        if uploaded:
            _load_preview(uploaded)
    else:
        cam_index = st.selectbox(
            "Caméra",
            options=[0, 1, 2, 3],
            format_func=lambda i: f"Caméra {i}",
            help="0 = caméra par défaut du système.",
        )
        if st.button("📷 Aperçu webcam", use_container_width=True):
            with st.spinner("Ouverture de la caméra…"):
                ok = _capture_webcam_preview(cam_index)
            if not ok:
                st.error(f"Impossible d'ouvrir la caméra {cam_index}.")

    vid_w = st.session_state.preview_w
    vid_h = st.session_state.preview_h
    if st.session_state._vid_key:
        st.caption(f"Dimensions : **{vid_w} × {vid_h} px**")

    # --- Détection ---
    st.subheader("Détection")
    conf_threshold = st.slider("Confiance min.", 0.2, 0.8, 0.35, 0.05)
    inference_size = st.select_slider(
        "Taille inférence (px)",
        options=[320, 416, 480, 640],
        value=416,
        help="Réduire améliore le FPS sur CPU.",
    )
    skip_frames = st.slider("Skip frames (0 = aucun)", 0, 5, 1)

    # --- Ligne de comptage ---
    st.subheader("Ligne de comptage")
    st.caption("Déplace A et B — la preview se met à jour en direct.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("🔵 **Point A**")
        line_x1 = st.slider("X₁", 0, vid_w, 0, key="lx1")
        line_y1 = st.slider("Y₁", 0, vid_h, vid_h // 2, key="ly1")
    with col_b:
        st.markdown("🔴 **Point B**")
        line_x2 = st.slider("X₂", 0, vid_w, vid_w, key="lx2")
        line_y2 = st.slider("Y₂", 0, vid_h, vid_h // 2, key="ly2")

    line_tolerance = st.slider("Tolérance anti-rebond (px)", 2, 50, 10, key="line_tol")

    line_direction = st.radio(
        "Sens comptabilisé",
        options=["Les deux", "Vers côté + 🟢", "Vers côté − 🟠"],
        horizontal=True,
        key="line_dir",
    )
    direction_map = {"Les deux": 0, "Vers côté + 🟢": 1, "Vers côté − 🟠": -1}

    # --- Zone de densité ---
    st.subheader("Zone de densité")
    zone_x1 = st.slider("X gauche",  0, vid_w, vid_w // 4, key="zx1")
    zone_x2 = st.slider("X droite",  0, vid_w, 3 * vid_w // 4, key="zx2")
    zone_y1 = st.slider("Y haut",    0, vid_h, vid_h // 4, key="zy1")
    zone_y2 = st.slider("Y bas",     0, vid_h, 3 * vid_h // 4, key="zy2")

    # --- Alertes ---
    st.subheader("Seuils d'alerte")
    threshold_orange = st.slider("🟠 Seuil orange", 1, 30, 8)
    threshold_red    = st.slider("🔴 Seuil rouge",  2, 50, 15)

    # --- Options ---
    st.subheader("Options")
    enable_blur   = st.toggle("Floutage visages (RGPD)", value=False)
    enable_buffer = st.toggle("Buffériser pour export MP4", value=False)

    st.divider()
    col_s, col_x = st.columns(2)
    with col_s:
        btn_start = st.button("▶ Démarrer", use_container_width=True,
                               type="primary",
                               disabled=not (uploaded or is_webcam))
    with col_x:
        btn_stop = st.button("⏹ Arrêter", use_container_width=True)

# ---------------------------------------------------------------------------
# Zone principale
# ---------------------------------------------------------------------------
st.title("✈️ O.P.T.I.C. — Analyse de flux passagers")

if not uploaded and not is_webcam:
    st.info("Chargez une vidéo ou sélectionnez **Webcam** dans la barre latérale.", icon="📂")
    st.stop()

col_vid, col_dash = st.columns([3, 1])

with col_vid:
    frame_placeholder = st.empty()

with col_dash:
    st.subheader("Métriques")
    ph_pos   = st.empty()
    ph_neg   = st.empty()
    ph_total = st.empty()
    ph_zone  = st.empty()
    ph_alert = st.empty()
    ph_fps   = st.empty()
    st.divider()
    btn_capture    = st.button("📸 Capturer frame",  use_container_width=True)
    btn_export_mp4 = st.button("🎬 Exporter MP4",    use_container_width=True)

dl_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
if btn_start:
    st.session_state.running   = True
    st.session_state.count_pos = 0
    st.session_state.count_neg = 0
    st.session_state.frame_count = 0
    st.session_state.zone_count  = 0
    st.session_state.capture_bytes     = None
    st.session_state.export_bytes      = None
    st.session_state.csv_frames_bytes  = None
    st.session_state.csv_events_bytes  = None
    st.session_state.last_error        = ""

if btn_stop:
    st.session_state.running = False

with dl_placeholder.container():
    if st.session_state.capture_bytes:
        st.download_button("⬇️ Télécharger capture PNG",
                           data=st.session_state.capture_bytes,
                           file_name="optic_capture.png", mime="image/png",
                           use_container_width=True)
    if st.session_state.export_bytes:
        st.download_button("⬇️ Télécharger extrait MP4",
                           data=st.session_state.export_bytes,
                           file_name="optic_extrait.mp4", mime="video/mp4",
                           use_container_width=True)
    if st.session_state.csv_frames_bytes:
        st.download_button("⬇️ CSV — stats par frame",
                           data=st.session_state.csv_frames_bytes,
                           file_name="optic_frames.csv", mime="text/csv",
                           use_container_width=True)
    if st.session_state.csv_events_bytes:
        st.download_button("⬇️ CSV — événements franchissement",
                           data=st.session_state.csv_events_bytes,
                           file_name="optic_events.csv", mime="text/csv",
                           use_container_width=True)

# ---------------------------------------------------------------------------
# Mode preview (pas en cours d'analyse)
# ---------------------------------------------------------------------------
if not st.session_state.running:
    if st.session_state.last_error:
        st.error(f"Erreur lors de l'analyse : {st.session_state.last_error}")
    elif st.session_state.frame_count > 0:
        st.success(
            f"Analyse terminée — {st.session_state.frame_count} frames · "
            f"→+ {st.session_state.count_pos} · →− {st.session_state.count_neg}"
        )

    ph_pos.metric("→+ (côté vert)",    st.session_state.count_pos)
    ph_neg.metric("→− (côté orange)",  st.session_state.count_neg)
    ph_total.metric("Total passages",  st.session_state.count_pos + st.session_state.count_neg)
    ph_zone.metric("Personnes en zone", st.session_state.zone_count)
    ph_fps.metric("FPS",               f"{st.session_state.fps:.1f}")

    if st.session_state.preview_frame is not None:
        caption = (
            "Preview webcam — ajuste les points A/B et la zone, puis clique sur ▶ Démarrer"
            if is_webcam else
            "Preview — ajuste les points A/B et la zone, puis clique sur ▶ Démarrer"
        )
        preview = _draw_config_overlay(
            st.session_state.preview_frame,
            int(line_x1), int(line_y1), int(line_x2), int(line_y2),
            int(line_tolerance), direction_map[line_direction],
            int(zone_x1), int(zone_y1), int(zone_x2), int(zone_y2),
        )
        frame_placeholder.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
            caption=caption,
        )
    elif is_webcam:
        frame_placeholder.info(
            "Clique sur **📷 Aperçu webcam** dans la sidebar pour vérifier le cadrage "
            "avant de démarrer.",
            icon="📷",
        )
    st.stop()

# ---------------------------------------------------------------------------
# Boucle d'analyse
# ---------------------------------------------------------------------------
detector = Detector(conf_threshold=conf_threshold, inference_size=inference_size)
counter  = VirtualLineCounter(
    x1=int(line_x1), y1=int(line_y1),
    x2=int(line_x2), y2=int(line_y2),
    tolerance=int(line_tolerance),
    direction=direction_map[line_direction],
)
zone = DensityZone(
    x1=int(zone_x1), y1=int(zone_y1),
    x2=int(zone_x2), y2=int(zone_y2),
)
alert_engine = AlertEngine(AlertConfig(
    threshold_orange=threshold_orange,
    threshold_red=threshold_red,
))
video_buffer: VideoBuffer | None = None

# --- Ouverture de la source ---
tmp_video: Path | None = None
if is_webcam:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        st.error(f"Impossible d'ouvrir la caméra {cam_index}. "
                 "Vérifiez qu'elle n'est pas utilisée par une autre application.")
        st.session_state.running = False
        st.stop()
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
else:
    tmp_video = Path(f"/tmp/optic_input_{int(time.time())}.video")
    tmp_video.write_bytes(uploaded.getvalue())
    cap = cv2.VideoCapture(str(tmp_video))
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la vidéo. Vérifiez le format du fichier.")
        st.session_state.running = False
        tmp_video.unlink(missing_ok=True)
        st.stop()
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
if enable_buffer:
    video_buffer = VideoBuffer(fps=source_fps / max(1, skip_frames + 1))

frame_idx   = 0
t_fps_win   = time.perf_counter()
fps_counter = 0
current_fps = 0.0
frame_stats:    list[dict] = []
crossing_events: list[dict] = []

# True quand la boucle se termine normalement (fin vidéo ou erreur gérée).
# False quand Streamlit injecte StopException (clic "Arrêter") — dans ce cas
# on ne déclenche PAS st.rerun() : Streamlit va lui-même relancer le script
# avec btn_stop=True, et les CSV seront déjà dans session_state.
_loop_ended_naturally = False

try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                st.session_state.last_error = "Flux webcam interrompu."
            st.session_state.running = False
            break

        frame_idx += 1
        if skip_frames > 0 and (frame_idx % (skip_frames + 1)) != 0:
            continue

        st.session_state.frame_count = frame_idx

        detections, annotated = detector.detect_and_track(frame, draw=True)

        if enable_blur and detections:
            annotated = blur_heads(annotated, detections)

        # Comptage
        crossings: dict[int, int] = {}
        for det in detections:
            r = counter.update(det.track_id, det.cx, det.cy, frame_idx)
            if r != 0:
                crossings[det.track_id] = r
        st.session_state.count_pos = counter.count_pos
        st.session_state.count_neg = counter.count_neg
        counter.draw(annotated, crossings)

        # Zone + alerte
        zone_count = zone.update(detections)
        st.session_state.zone_count = zone_count
        alert_color = alert_engine.get_bgr(zone_count)
        zone.draw(annotated, alert_color)
        alert_info = alert_engine.get_display(zone_count)

        if video_buffer is not None:
            video_buffer.push(annotated)

        # --- Collecte données CSV ---
        time_s = round(frame_idx / source_fps, 3)
        frame_stats.append({
            "frame":           frame_idx,
            "time_s":          time_s,
            "persons":         len(detections),
            "zone_count":      zone_count,
            "alert":           alert_info["label"],
            "count_pos_cumul": counter.count_pos,
            "count_neg_cumul": counter.count_neg,
            "total_cumul":     counter.count,
        })
        for tid, direction in crossings.items():
            crossing_events.append({
                "frame":           frame_idx,
                "time_s":          time_s,
                "track_id":        tid,
                "direction":       "+" if direction == 1 else "-",
                "count_pos_cumul": counter.count_pos,
                "count_neg_cumul": counter.count_neg,
                "total_cumul":     counter.count,
            })

        # FPS
        fps_counter += 1
        now = time.perf_counter()
        if now - t_fps_win >= 1.0:
            current_fps = fps_counter / (now - t_fps_win)
            fps_counter  = 0
            t_fps_win    = now
            st.session_state.fps = current_fps

        # Rendu
        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True,
        )

        # Métriques
        ph_pos.metric("→+ (côté vert)",   counter.count_pos)
        ph_neg.metric("→− (côté orange)",  counter.count_neg)
        ph_total.metric("Total passages",  counter.count)
        ph_zone.metric("Personnes en zone", zone_count)
        ph_fps.metric("FPS", f"{current_fps:.1f}")
        ph_alert.markdown(
            f'<div class="alert-badge" style="background:{alert_info["color"]}">'
            f'{alert_info["emoji"]} {alert_info["label"]}</div>',
            unsafe_allow_html=True,
        )

        if btn_capture:
            st.session_state.capture_bytes = capture_frame(annotated)
            btn_capture = False

        if btn_export_mp4 and video_buffer is not None:
            st.session_state.export_bytes = video_buffer.export_mp4()
            btn_export_mp4 = False

        time.sleep(0.001)

    # Boucle terminée normalement (fin de vidéo ou running=False interne)
    _loop_ended_naturally = True

except Exception as exc:
    st.session_state.last_error = str(exc)
    st.session_state.running = False
    _loop_ended_naturally = True  # relancer pour afficher l'erreur

finally:
    cap.release()
    if tmp_video is not None:
        tmp_video.unlink(missing_ok=True)
    st.session_state.csv_frames_bytes = to_csv_bytes(frame_stats)
    st.session_state.csv_events_bytes = to_csv_bytes(crossing_events)
    st.session_state.running = False
    # On appelle st.rerun() seulement pour les fins naturelles (fin vidéo,
    # erreur). Pour un arrêt manuel (StopException de Streamlit), on laisse
    # l'exception se propager : Streamlit va démarrer un nouveau run avec
    # btn_stop=True, les CSV étant déjà en session_state.
    if _loop_ended_naturally:
        st.rerun()
