#!/bin/bash
# Script di avvio per l'applicazione Filtri Webcam in Real Time

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creazione ambiente virtuale Python..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Errore: impossibile creare l'ambiente virtuale. Assicurati che python3-venv sia installato."
        exit 1
    fi
fi

echo "Attivazione ambiente virtuale..."
source "$VENV_DIR/bin/activate"

echo "Installazione dipendenze..."
pip install -r requirements.txt

# Scarica modello MediaPipe se mancante
MODEL_PATH="assets/face_landmarker.task"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Download modello face landmarker..."
    mkdir -p assets
    curl -L -o "$MODEL_PATH" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
fi

echo "Avvio applicazione..."
python3 main.py

deactivate
