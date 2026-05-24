import cv2
import numpy as np
import mediapipe as mp
import os

# Directory degli asset (relativa al file corrente)
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

# Mediapipe Face Landmarker (nuova API tasks)
_face_landmarker = None

def _get_landmarker():
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    model_path = os.path.join(ASSETS_DIR, 'face_landmarker.task')
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


# Cache per immagini overlay
_overlay_cache = {}

def _load_overlay(filename):
    # Carica PNG con canale alpha, usa cache
    if filename in _overlay_cache:
        return _overlay_cache[filename]
    path = os.path.join(ASSETS_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Attenzione: impossibile caricare {path}")
        return None
    _overlay_cache[filename] = img
    return img


def _overlay_png(frame, overlay_img, x, y, w, h):
    # Sovrappone immagine PNG con alpha sul frame
    if overlay_img is None or w <= 0 or h <= 0:
        return frame

    resized = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)

    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        return frame

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if resized.shape[2] == 4:
        alpha = resized[oy1:oy2, ox1:ox2, 3] / 255.0
        alpha_3d = np.dstack([alpha] * 3)
        bg = frame[y1:y2, x1:x2].astype(float)
        fg = resized[oy1:oy2, ox1:ox2, :3].astype(float)
        frame[y1:y2, x1:x2] = (fg * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = resized[oy1:oy2, ox1:ox2]

    return frame


def _detect_landmarks(frame):
    # Rileva landmark facciali con mediapipe
    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    return result.face_landmarks


def detect_faces_count(frame):
    # Conta facce rilevate
    return len(_detect_landmarks(frame))


def detect_faces_only(frame):
    # Compatibilità col vecchio codice
    return detect_faces_count(frame)


def apply_background_blur(frame):
    # Sfoca sfondo mantenendo facce nitide
    h, w = frame.shape[:2]
    faces = _detect_landmarks(frame)
    num_faces = len(faces)

    blurred = cv2.GaussianBlur(frame, (51, 51), 0)

    if num_faces == 0:
        return blurred, 0

    mask = np.zeros((h, w), dtype=np.uint8)

    # Indici contorno viso nel face mesh mediapipe
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    for face_lm in faces:
        pts = []
        for idx in FACE_OVAL:
            lm = face_lm[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])
        pts = np.array(pts, dtype=np.int32)

        center = pts.mean(axis=0).astype(int)
        expanded = ((pts - center) * 1.4 + center).astype(np.int32)
        cv2.fillConvexPoly(mask, expanded, 255)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    result = frame * mask_3d + blurred * (1 - mask_3d)
    return result.astype(np.uint8), num_faces


def apply_hat(frame):
    # Sovrappone cappello sopra ogni faccia
    h, w = frame.shape[:2]
    hat_img = _load_overlay('cappello.png')
    faces = _detect_landmarks(frame)

    result = frame.copy()
    for face_lm in faces:
        top = face_lm[10]      # fronte
        chin = face_lm[152]    # mento
        left = face_lm[234]    # lato sinistro
        right = face_lm[454]   # lato destro

        face_w = int(abs(right.x - left.x) * w * 1.5)
        face_h = int(abs(chin.y - top.y) * h)

        hat_w = face_w
        hat_h = int(hat_w * 0.75)

        cx = int((left.x + right.x) / 2 * w)
        hat_x = cx - hat_w // 2
        hat_y = int(top.y * h) - hat_h + int(face_h * 0.15)

        result = _overlay_png(result, hat_img, hat_x, hat_y, hat_w, hat_h)

    return result, len(faces)


def apply_sunglasses(frame):
    # Sovrappone occhiali da sole sugli occhi
    h, w = frame.shape[:2]
    glasses_img = _load_overlay('occhiali.png')
    faces = _detect_landmarks(frame)

    result = frame.copy()
    for face_lm in faces:
        left_eye_outer = face_lm[33]
        right_eye_outer = face_lm[263]

        eye_cx = int((left_eye_outer.x + right_eye_outer.x) / 2 * w)
        eye_cy = int((left_eye_outer.y + right_eye_outer.y) / 2 * h)

        eye_dist = int(abs(right_eye_outer.x - left_eye_outer.x) * w)
        glasses_w = int(eye_dist * 1.6)
        glasses_h = int(glasses_w * 0.45)

        gx = eye_cx - glasses_w // 2
        gy = eye_cy - glasses_h // 2

        result = _overlay_png(result, glasses_img, gx, gy, glasses_w, glasses_h)

    return result, len(faces)


def apply_mustache(frame):
    # Sovrappone baffi sotto il naso
    h, w = frame.shape[:2]
    mustache_img = _load_overlay('baffi.png')
    faces = _detect_landmarks(frame)

    result = frame.copy()
    for face_lm in faces:
        nose_tip = face_lm[1]
        upper_lip = face_lm[0]
        left_mouth = face_lm[61]
        right_mouth = face_lm[291]

        mouth_w = int(abs(right_mouth.x - left_mouth.x) * w)
        m_w = int(mouth_w * 1.8)
        m_h = int(m_w * 0.4)

        cx = int((left_mouth.x + right_mouth.x) / 2 * w)
        cy = int((nose_tip.y + upper_lip.y) / 2 * h)

        mx = cx - m_w // 2
        my = cy - m_h // 2

        result = _overlay_png(result, mustache_img, mx, my, m_w, m_h)

    return result, len(faces)


def apply_face_label(frame, label="Player"):
    # Scrive etichetta sopra ogni faccia
    h, w = frame.shape[:2]
    faces = _detect_landmarks(frame)

    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, face_lm in enumerate(faces):
        top = face_lm[10]
        left = face_lm[234]
        right = face_lm[454]

        cx = int((left.x + right.x) / 2 * w)
        ty = int(top.y * h) - 20

        text = f"{label} {i+1}" if len(faces) > 1 else label
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]

        tx = cx - text_size[0] // 2
        cv2.rectangle(result, (tx - 5, ty - text_size[1] - 5),
                      (tx + text_size[0] + 5, ty + 5), (0, 0, 0), -1)
        cv2.putText(result, text, (tx, ty), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return result, len(faces)


def apply_motion_detection(frame, prev_frame):
    # Evidenzia zone con movimento
    if prev_frame is None:
        return frame

    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    motion_mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255

    result = frame.copy()
    motion_area = motion_mask > 0
    result[motion_area] = cv2.addWeighted(
        frame, 0.5, green_overlay, 0.5, 0
    )[motion_area]

    return result
