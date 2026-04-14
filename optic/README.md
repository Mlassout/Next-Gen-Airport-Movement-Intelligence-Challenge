# O.P.T.I.C. — Optimized Passenger Traffic Intelligence & Control

O.P.T.I.C. est un POC d'analyse vidéo de flux passagers pour environnements aéroportuaires. Il détecte et suit les personnes en temps réel via YOLOv8, comptabilise les passages à travers une ligne virtuelle configurable, mesure la densité en zone, et déclenche des alertes visuelles lorsque les seuils d'occupation sont dépassés. L'ensemble tourne en local sur CPU, sans aucun service externe, dans le respect des principes RGPD (pas de stockage d'identité, floutage des visages optionnel).

## Prérequis

- Python 3.9+
- Fichier vidéo `.mp4` ou `.avi` local

## Installation

```bash
cd optic
pip install -r requirements.txt
```

Le modèle `yolov8n.pt` est téléchargé automatiquement par Ultralytics lors du premier lancement.

## Lancement

```bash
streamlit run app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501).

## Utilisation (3 actions)

1. **Charger** une vidéo via le sélecteur de fichier dans la barre latérale
2. **Configurer** la ligne virtuelle, la zone et les seuils d'alerte (valeurs par défaut opérationnelles)
3. **Lancer** l'analyse avec le bouton ▶ Démarrer — arrêt avec ⏹ Arrêter

## Fonctionnalités

| Fonction | Description |
|---|---|
| Détection | YOLOv8n, classe `person`, bounding boxes annotées |
| Suivi | IDs persistants inter-frames (`model.track(persist=True)`) |
| Comptage | Ligne horizontale configurable, sens de passage, anti-rebond |
| Densité | Rectangle de zone, comptage de centroïdes en temps réel |
| Alertes | 3 niveaux 🟢🟠🔴 avec seuils configurables |
| Dashboard | Flux annoté, compteur, densité, alerte, FPS |
| Export | Capture frame PNG, extrait vidéo MP4 annoté |
| Floutage | Option RGPD — flou gaussien sur la zone tête (haut des bboxes) |

## Architecture

```
optic/
├── app.py              # Interface Streamlit, boucle vidéo principale
├── core/
│   ├── detector.py     # Wrapper YOLOv8, inférence + tracking
│   ├── tracker.py      # Gestion des états de tracking inter-frames
│   ├── counter.py      # Comptage par ligne virtuelle + anti-rebond
│   ├── zone.py         # Densité en zone rectangle
│   └── alert.py        # Logique niveaux d'alerte
├── utils/
│   ├── blur.py         # Floutage des visages (RGPD)
│   └── export.py       # Export PNG / MP4 annoté
└── requirements.txt
```

## Limites connues du POC

- Performances limitées à ~10-15 FPS sur CPU (yolov8n) — dégradation possible sur vidéos haute résolution
- Le tracker YOLOv8 peut perdre des IDs lors d'occultations prolongées (foule dense)
- La ligne virtuelle est uniquement horizontale (pas d'orientation libre)
- L'export MP4 annoté bufférise en mémoire — limité à ~60 secondes sur machines avec peu de RAM
- Pas de persistence entre sessions (redémarrage = remise à zéro des compteurs)

## Captures d'écran

*[placeholder — à ajouter après première exécution]*
