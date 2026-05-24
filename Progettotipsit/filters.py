import cv2
import numpy as np

def apply_grayscale(frame):
    """
    Converte il frame a colori in scala di grigi e poi lo riconverte
    in BGR per mantenere i 3 canali, utile per la compatibilità in output.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_negative(frame):
    """
    Inverte i colori del frame sottraendo ogni pixel da 255.
    """
    return cv2.bitwise_not(frame)

def apply_sepia(frame):
    """
    Applica un effetto sepia trasformando i colori con una matrice predefinita.
    """
    # Matrice standard per l'effetto sepia
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    # Applica la trasformazione e assicura che i valori non superino 255
    sepia = cv2.transform(frame, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_cartoon(frame):
    """
    Crea un effetto fumetto appiattendo i colori con un bilateral filter
    e sovrapponendo i bordi rilevati con l'algoritmo Canny o adaptiveThreshold.
    """
    # Converti in scala di grigi per la rilevazione dei bordi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Usa median blur per ridurre il rumore
    gray = cv2.medianBlur(gray, 5)
    
    # Rileva i bordi usando un adaptive threshold
    edges = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # Applica un bilateral filter all'immagine originale per dare l'effetto "pittura/appiattito"
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    
    # Combina i bordi neri con l'immagine a colori
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_heatmap(frame):
    """
    Effetto termico (Heatmap): converte in scala di grigi e applica una mappa di colori.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return heatmap

def apply_pixelate(frame, blocks=20):
    """
    Effetto Pixelate: rimpicciolisce pesantemente il frame e lo riingrandisce
    utilizzando l'interpolazione nearest-neighbor, creando l'effetto "pixel art".
    """
    h, w = frame.shape[:2]
    # Ridimensiona verso il basso
    small = cv2.resize(frame, (blocks, int(blocks * (h / w))), interpolation=cv2.INTER_LINEAR)
    # Ridimensiona verso l'alto
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def apply_vignette(frame):
    """
    Effetto Vignettatura: scurisce progressivamente i bordi dell'immagine.
    """
    h, w = frame.shape[:2]
    
    # Genera maschere 1D per le righe (X) e per le colonne (Y)
    # np.getGaussianKernel restituisce un array 2D di forma (ksize, 1)
    kernel_x = cv2.getGaussianKernel(w, w / 2)
    kernel_y = cv2.getGaussianKernel(h, h / 2)
    
    # Moltiplica i due kernel per ottenere una matrice 2D gaussiana
    kernel = kernel_y * kernel_x.T
    
    # Normalizza il kernel in modo che il centro sia 1.0
    mask = kernel / kernel.max()
    
    # Crea una maschera 3D e fonde l'immagine con essa
    mask_3d = np.dstack([mask] * 3)
    
    # Moltiplica l'immagine originale per la maschera
    vignette = frame * mask_3d
    return np.clip(vignette, 0, 255).astype(np.uint8)
