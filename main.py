import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time
import csv
import io
import pandas as pd
from datetime import datetime

CONF_THRESHOLD = 0.35
COOLDOWN_SEC   = 2.0

st.set_page_config(page_title="O.P.T.I.C", layout="wide")
st.title("O.P.T.I.C — Optimisation Predictive des Trafics et Indicateurs de Congestion")

# ─── MODELE ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ─── SESSION STATE ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "running":            False,
    "cap":                None,
    "count_in":           0,
    "count_out":          0,
    "prev_side":          {},
    "last_counted_time":  {},
    "events":             [],
    "time_series":        [],
    "frame_num":          0,
    "last_density":       0,
    "last_frame":         None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_all():
    st.session_state.running            = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.count_in          = 0
    st.session_state.count_out         = 0
    st.session_state.prev_side         = {}
    st.session_state.last_counted_time = {}
    st.session_state.events            = []
    st.session_state.time_series       = []
    st.session_state.frame_num         = 0
    st.session_state.last_density      = 0
    st.session_state.last_frame        = None


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def line_side(px, py, x1, y1, x2, y2):
    val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if val > 0 else (-1 if val < 0 else 0)


def projects_onto_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq == 0:
        return True
    t = ((px - x1) * dx + (py - y1) * dy) / lsq
    return 0.0 <= t <= 1.0


def draw_overlay(frame, lx1, ly1, lx2, ly2, zx1, zy1, zx2, zy2):
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 180, 255), -1)
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, dst=out)
    cv2.rectangle(out, (zx1, zy1), (zx2, zy2), (0, 180, 255), 2)
    cv2.putText(out, "ZONE", (zx1 + 4, zy1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    cv2.line(out, (lx1, ly1), (lx2, ly2), (0, 60, 255), 3)
    cv2.circle(out, (lx1, ly1), 7, (0, 60, 255), -1)
    cv2.circle(out, (lx2, ly2), 7, (0, 60, 255), -1)
    return out


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    source_type = st.radio("Source video", ["Fichier", "Webcam"], horizontal=True, key="source_type")
    if source_type == "Fichier":
        st.text_input("Chemin de la video", "samples_data_vtest.avi", key="video_path")

    st.checkbox("Floutage RGPD", value=True, key="enable_blur")

    st.divider()
    st.subheader("Ligne de comptage")
    st.caption("0% = gauche/haut  —  100% = droite/bas")
    st.markdown("**Point de depart**")
    ca, cb = st.columns(2)
    with ca:
        st.slider("Gauche →", 0, 100,  0,  key="lx1")
        st.slider("Haut ↓",   0, 100, 50,  key="ly1")
    with cb:
        st.slider("Gauche →", 0, 100, 100, key="lx2")
        st.slider("Haut ↓",   0, 100,  50, key="ly2")

    st.divider()
    st.subheader("Zone de densite")
    za, zb = st.columns(2)
    with za:
        st.slider("Bord gauche", 0, 100, 20, key="zx1")
        st.slider("Bord haut",   0, 100, 20, key="zy1")
    with zb:
        st.slider("Bord droit",  0, 100, 80, key="zx2")
        st.slider("Bord bas",    0, 100, 80, key="zy2")

    st.divider()
    st.subheader("Seuils d'alerte")
    st.slider("Orange (pers.)", 1, 30,  5, key="thr_orange")
    st.slider("Rouge  (pers.)", 1, 50, 15, key="thr_red")

    st.divider()
    b1, b2, b3 = st.columns(3)
    start_btn = b1.button("Start", use_container_width=True)
    stop_btn  = b2.button("Stop",  use_container_width=True)
    reset_btn = b3.button("Reset", use_container_width=True)

# ─── CONTROLE VIDEO (pleine page — sidebar) ───────────────────────────────────
if start_btn and not st.session_state.running:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    src = (0 if st.session_state.source_type == "Webcam"
           else st.session_state.get("video_path", "samples_data_vtest.avi"))
    _cap = cv2.VideoCapture(src)
    if not _cap.isOpened():
        st.sidebar.error("Impossible d'ouvrir la source video.")
    else:
        st.session_state.cap     = _cap
        st.session_state.running = True

if stop_btn:
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

if reset_btn:
    reset_all()


# ─── FRAGMENT VIDEO (re-render partiel, sans recharger la page) ───────────────
@st.fragment(run_every=0.04)   # ~25 fps max ; limité en pratique par YOLO
def video_loop():
    # Lire les valeurs des sliders depuis session_state (mis a jour par la sidebar)
    lx1 = st.session_state.get("lx1", 0)
    ly1 = st.session_state.get("ly1", 50)
    lx2 = st.session_state.get("lx2", 100)
    ly2 = st.session_state.get("ly2", 50)
    zx1 = st.session_state.get("zx1", 20)
    zy1 = st.session_state.get("zy1", 20)
    zx2 = st.session_state.get("zx2", 80)
    zy2 = st.session_state.get("zy2", 80)
    thr_o       = st.session_state.get("thr_orange", 5)
    thr_r       = st.session_state.get("thr_red", 15)
    enable_blur = st.session_state.get("enable_blur", True)

    vid_col, stat_col = st.columns([3, 1])

    # ── MODE CONFIGURATION ──
    if not st.session_state.running or st.session_state.cap is None:
        with vid_col:
            if st.session_state.last_frame is not None:
                bg = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_RGB2BGR)
            else:
                bg = np.full((360, 640, 3), 30, dtype=np.uint8)
            h_bg, w_bg = bg.shape[:2]
            preview = draw_overlay(
                bg,
                int(lx1/100*w_bg), int(ly1/100*h_bg),
                int(lx2/100*w_bg), int(ly2/100*h_bg),
                int(zx1/100*w_bg), int(zy1/100*h_bg),
                int(zx2/100*w_bg), int(zy2/100*h_bg),
            )
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                     caption="Apercu — modifiez les sliders pour repositionner",
                     use_container_width=True)

        with stat_col:
            _show_stats(thr_o, thr_r, st.session_state.last_density)
        return

    # ── MODE ANALYSE ──
    cap = st.session_state.cap
    t0  = time.time()
    ret, frame = cap.read()

    if not ret:
        st.session_state.running = False
        cap.release()
        st.session_state.cap = None
        with vid_col:
            st.success("Fin de la video — analyse terminee.")
        with stat_col:
            _show_stats(thr_o, thr_r, st.session_state.last_density)
        return

    h, w = frame.shape[:2]
    lx1_px = int(lx1/100*w);  ly1_px = int(ly1/100*h)
    lx2_px = int(lx2/100*w);  ly2_px = int(ly2/100*h)
    zx1_px = int(zx1/100*w);  zy1_px = int(zy1/100*h)
    zx2_px = int(zx2/100*w);  zy2_px = int(zy2/100*h)

    frame = draw_overlay(frame, lx1_px, ly1_px, lx2_px, ly2_px,
                         zx1_px, zy1_px, zx2_px, zy2_px)

    try:
        results = model.track(frame, persist=True, classes=0,
                              conf=CONF_THRESHOLD, verbose=False)
    except AttributeError as e:
        if "'Conv' object has no attribute 'bn'" in str(e):
            load_model.clear()
            results = model.track(frame, persist=True, classes=0,
                                  conf=CONF_THRESHOLD, verbose=False)
        else:
            raise

    density = 0

    if results[0].boxes.id is not None:
        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, tid_f in zip(boxes, track_ids):
            bx1, by1, bx2, by2 = box
            cx  = int((bx1 + bx2) / 2)
            cy  = int((by1 + by2) / 2)
            tid = int(tid_f)

            if enable_blur:
                try:
                    roi = frame[int(by1):int(by2), int(bx1):int(bx2)]
                    if roi.size > 0:
                        frame[int(by1):int(by2), int(bx1):int(bx2)] = (
                            cv2.GaussianBlur(roi, (51, 51), 0)
                        )
                except Exception:
                    pass

            in_zone = zx1_px <= cx <= zx2_px and zy1_px <= cy <= zy2_px
            if in_zone:
                density += 1

            current_side = line_side(cx, cy, lx1_px, ly1_px, lx2_px, ly2_px)
            prev = st.session_state.prev_side.get(tid)
            if prev is not None:
                last_t = st.session_state.last_counted_time.get(tid, 0.0)
                on_seg = projects_onto_segment(cx, cy, lx1_px, ly1_px, lx2_px, ly2_px)
                if (on_seg and prev != 0 and current_side != 0
                        and prev != current_side
                        and (t0 - last_t) > COOLDOWN_SEC):
                    direction = "IN" if prev == -1 else "OUT"
                    if direction == "IN":
                        st.session_state.count_in  += 1
                    else:
                        st.session_state.count_out += 1
                    st.session_state.last_counted_time[tid] = t0
                    st.session_state.events.append({
                        "Horodatage": datetime.now().strftime("%H:%M:%S"),
                        "Direction":  direction,
                        "ID":         tid,
                    })
            if current_side != 0:
                st.session_state.prev_side[tid] = current_side

            color = (0, 180, 255) if in_zone else (50, 220, 50)
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"ID:{tid}",
                        (int(bx1), max(int(by1) - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    st.session_state.time_series.append({
        "Heure":   datetime.now().strftime("%H:%M:%S"),
        "Entrees": st.session_state.count_in,
        "Sorties": st.session_state.count_out,
        "Densite": density,
    })
    if len(st.session_state.time_series) > 500:
        st.session_state.time_series = st.session_state.time_series[-500:]

    st.session_state.last_density = density
    st.session_state.frame_num   += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame = frame_rgb

    with vid_col:
        st.image(frame_rgb, use_container_width=True)
    with stat_col:
        _show_stats(thr_o, thr_r, density)


def _show_stats(thr_o, thr_r, density):
    st.subheader("Indicateurs")
    st.metric("Entrees",      st.session_state.count_in)
    st.metric("Sorties",      st.session_state.count_out)
    st.metric("Flux net",     st.session_state.count_in - st.session_state.count_out)
    st.metric("Densite zone", density)
    if density >= thr_r:
        lvl, color = "ROUGE",  "#d50000"
    elif density >= thr_o:
        lvl, color = "ORANGE", "#e65100"
    else:
        lvl, color = "VERT",   "#1b5e20"
    st.markdown(
        f"""<div style="background:{color};border-radius:8px;padding:14px;
            text-align:center;margin-top:6px">
            <b style="color:white;font-size:1.1em">ALERTE {lvl}</b><br>
            <span style="color:#eee">{density} pers. en zone</span>
            </div>""",
        unsafe_allow_html=True,
    )


video_loop()

# ─── GRAPHIQUE + EXPORT (hors fragment, affiches apres arret) ─────────────────
if not st.session_state.running and st.session_state.time_series:
    st.divider()
    st.subheader("Flux temporel")
    df_ts = pd.DataFrame(st.session_state.time_series).set_index("Heure")
    st.line_chart(df_ts[["Entrees", "Sorties", "Densite"]])

if st.session_state.events:
    st.divider()
    st.subheader("Historique des passages")
    st.dataframe(pd.DataFrame(st.session_state.events), use_container_width=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["Horodatage", "Direction", "ID"])
    writer.writeheader()
    writer.writerows(st.session_state.events)
    st.download_button(
        "Exporter CSV",
        buf.getvalue(),
        file_name=f"passages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
