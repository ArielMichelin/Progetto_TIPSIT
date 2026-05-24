REQUISITI
- Python 3.9+
- Webcam (USB o integrata)
- Windows, macOS, Linux, o Raspberry Pi

Avvio:
WINDOWS:
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  python main.py

LINUX/MAC/RASPBERRY PI:
  chmod +x run.sh
  ./run.sh


COMANDI TASTIERA

Filtri:
  0 = Normale 
  1 = Scala di grigi
  2 = Negativo 
  3 = Sepia 
  4 = Sfondo blur 
  5 = Cartoon 
  6 = Termico 
  7 = Pixelate 
  8 = Vignetta 

Overlay:
  H = Cappello 
  G = Occhiali da sole 
  M = Baffi 
  L = Etichetta, con più persone si adatta automaticamente

Effetti movimento:
  K = Rilevamento movimento

Controlli generali:
  F = Attiva/disattiva flip fotocamera
  S = Scatta
  R = Registrazione video on/off (salva in assets/recordings)
  Q = Esci

COME FUNZIONANO I FILTRI

SCALA DI GRIGI:
  Converte da colore a grigio. Semplicemente rimuove il colore, mantiene solo brillantezza.
  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

NEGATIVO:
  Inverte ogni pixel: nuovo = 255 - originale
  Per ogni canale BGR.

SEPIA:
  Prima converte a grigio, poi applica tinta giallo-marrone.
  Effetto fotografico antico.

SFONDO BLUR:
  Il filtro che mi ha dato più problemi. Funziona così:
  1. MediaPipe rileva 478 punti sulla faccia
  2. Prendo i 35 punti che formano il contorno del volto 
  3. Creo una maschera binaria: bianco dove c'è il volto, nero altrove
  4. Faccio blur gaussiano di tutta l'immagine
  5. Combino: volto originale + sfondo blurrato usando la maschera
 
CARTOON:
  Bilateral filter (sfuma colori ma mantiene bordi) + edge detection.

TERMICO (HEATMAP):
  Applica colormap JET del OpenCV. Pixel luminosi = rossi, scuri = blu.
  
PIXELATE:
  Resize down a piccolo (es. 32x32), poi resize up a originale con interpolazione nearest.
  

VIGNETTA:
  Crea una maschera gaussiana dal centro verso bordi (luminosa al centro, scura ai bordi).
  

OVERLAY SULLA FACCIA

CAPPELLO:
  PNG con canale alpha che sovrappongo sopra la testa.
  Uso landmark 10 (fronte) per posizione Y.
  Uso landmark 234 e 454 (lati della faccia) per capire la larghezza del volto.
  Scaling: cappello = 1.5x larghezza volto.
  Si aggiorna ogni frame, quindi segue il movimento della testa.

OCCHIALI:
  PNG posizionato tra landmark 33 e 263 (angoli occhi).
  Larghezza = 1.6x distanza tra gli occhi.
  Stessa tecnica alpha blending del cappello.

BAFFI:
  Landmark 1 (punta naso), landmark 61 e 291 (angoli bocca).
  Larghezza = 1.8x larghezza bocca.

ETICHETTA:
  Scrivo testo sopra la fronte di ogni volto rilevato.
  Se ci sono più volti, numerati: "Player 1", "Player 2", etc.

RILEVAMENTO MOVIMENTO:
  Calcolo differenza tra frame attuale e frame precedente.
  Applico threshold (30) per ignorare rumore.
  Overlay verde sulle zone che si muovono.
  È tipo un heatmap del movimento.

COME FUNZIONA MEDIAIPE

Uso MediaPipe Face Landmarker (tasks API, non il vecchio solutions).
Rileva 478 punti per ogni volto (landmark facciali).
punti usati:
  10   = Fronte
  152  = Mento
  234  = Occhio sinistro (angolo esterno)
  263  = Occhio destro (angolo esterno)
  1    = Punta naso
  61   = Angolo sinistro bocca
  291  = Angolo destro bocca
Fino a 5 volti.

RASPBERRY PI


Se usi Raspberry Pi installa le dipendenze di sistema:

  sudo apt-get update
  sudo apt-get install libgl1-mesa-glx libhdf5-dev libhdf5-serial-dev \
    libcblas-dev libatlas-base-dev libjasper-dev
