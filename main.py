import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="AeroVision AI", layout="wide")
st.title("✈️ AeroVision : Dashboard de Flux Aéroportuaire")
st.sidebar.header("Configuration")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') # Modèle pré-entraîné sur dataset COCO

model = load_model()

# --- PARAMÈTRES DANS LA BARRE LATÉRALE ---
video_source = st.sidebar.text_input("Chemin de la vidéo", "video_test.mp4")
line_y_pos = st.sidebar.slider("Position de la ligne de comptage", 0, 1000, 400)
conf_threshold = st.sidebar.slider("Seuil de confiance IA", 0.0, 1.0, 0.4)
enable_blur = st.sidebar.checkbox("Anonymisation RGPD (Floutage)", value=True)

# --- INITIALISATION DES VARIABLES ---
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.already_counted = set()

# --- ZONE D'AFFICHAGE ---
col1, col2 = st.columns([3, 1])
FRAME_WINDOW = col1.image([]) # Fenêtre pour la vidéo
metric_placeholder = col2.empty() # Place pour le compteur

# --- BOUCLE DE TRAITEMENT ---
cap = cv2.VideoCapture(video_source)

if st.sidebar.button("Démarrer l'Analyse"):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Fin de la vidéo ou fichier introuvable.")
            break

        # 1. IA : Détection et Tracking (Suivi multi-objets)
        results = model.track(frame, persist=True, classes=0, conf=conf_threshold, verbose=False)

        # Dessiner la ligne de comptage
        cv2.line(frame, (0, line_y_pos), (frame.shape[1], line_y_pos), (255, 0, 0), 3)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # --- OPTION FLOUTAGE RGPD ---
                if enable_blur:
                    try:
                        roi = frame[int(y1):int(y2), int(x1):int(x2)]
                        # Appliquer un flou gaussien pour protéger l'identité
                        roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        frame[int(y1):int(y2), int(x1):int(x2)] = roi
                    except:
                        pass # Sécurité si la boîte sort de l'image

                # --- LOGIQUE DE COMPTAGE ---
                # Si le centre franchit la ligne (marge de 15px)
                if (line_y_pos - 15) < cy < (line_y_pos + 15):
                    if track_id not in st.session_state.already_counted:
                        st.session_state.already_counted.add(track_id)
                        st.session_state.count += 1

                # Dessiner un indicateur sur la personne
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{int(track_id)}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- MISE À JOUR INTERFACE ---
        # Convertir BGR (OpenCV) en RGB (Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        
        # Mise à jour du compteur en temps réel
        metric_placeholder.metric("Passagers Détectés", st.session_state.count)

    cap.release()