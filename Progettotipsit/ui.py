import cv2
import numpy as np

def draw_hud(frame, fps, current_filter_name, num_faces):
    # HUD sovrimpresso: FPS, filtro attivo, conteggio facce
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # FPS in alto a destra
    fps_text = f"FPS: {fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, font, 0.7, 2)[0]
    cv2.putText(frame, fps_text, (w - fps_size[0] - 10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Filtro in alto a sinistra
    filter_text = f"Filtro: {current_filter_name}"
    cv2.putText(frame, filter_text, (10, 30), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Facce
    face_text = f"Facce: {num_faces}"
    color_faces = (0, 255, 0) if num_faces > 0 else (0, 0, 255)
    cv2.putText(frame, face_text, (10, 60), font, 0.7, color_faces, 2, cv2.LINE_AA)

    return frame


def draw_filter_bar(frame, available_filters, current_filter_index):
    # Barra in basso con filtri disponibili, evidenzia attivo
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    bar_height = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    num_filters = len(available_filters)
    if num_filters == 0:
        return frame

    # Due righe se troppi filtri
    per_row = (num_filters + 1) // 2
    row_height = bar_height // 2

    for i, filt in enumerate(available_filters):
        row = i // per_row
        col = i % per_row

        text = f"{filt['key'].upper()}:{filt['name']}"
        is_active = (i == current_filter_index)
        color = (0, 255, 255) if is_active else (180, 180, 180)
        thickness = 2 if is_active else 1
        scale = 0.4

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        seg_w = w // per_row
        x_pos = col * seg_w + (seg_w - text_size[0]) // 2
        y_pos = h - bar_height + row * row_height + row_height - 5

        if is_active:
            cv2.rectangle(frame, (x_pos - 3, y_pos - text_size[1] - 3),
                          (x_pos + text_size[0] + 3, y_pos + 3), (0, 255, 255), 1)

        cv2.putText(frame, text, (x_pos, y_pos), font, scale, color, thickness, cv2.LINE_AA)

    return frame
