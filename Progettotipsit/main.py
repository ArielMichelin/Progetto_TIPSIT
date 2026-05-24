import cv2
import time
import os
from datetime import datetime

import filters
import effects
import ui

def main():
    available_filters = [
        {'key': '0', 'name': 'Normale', 'func': None, 'type': 'normal'},
        {'key': '1', 'name': 'Grigio', 'func': filters.apply_grayscale, 'type': 'color'},
        {'key': '2', 'name': 'Negativo', 'func': filters.apply_negative, 'type': 'color'},
        {'key': '3', 'name': 'Sepia', 'func': filters.apply_sepia, 'type': 'color'},
        {'key': '4', 'name': 'Sfondo Blur', 'func': effects.apply_background_blur, 'type': 'face_effect'},
        {'key': '5', 'name': 'Cartoon', 'func': filters.apply_cartoon, 'type': 'color'},
        {'key': '6', 'name': 'Termico', 'func': filters.apply_heatmap, 'type': 'color'},
        {'key': '7', 'name': 'Pixelate', 'func': filters.apply_pixelate, 'type': 'color'},
        {'key': '8', 'name': 'Vignetta', 'func': filters.apply_vignette, 'type': 'color'},
        {'key': 'h', 'name': 'Cappello', 'func': effects.apply_hat, 'type': 'face_effect'},
        {'key': 'g', 'name': 'Occhiali', 'func': effects.apply_sunglasses, 'type': 'face_effect'},
        {'key': 'm', 'name': 'Baffi', 'func': effects.apply_mustache, 'type': 'face_effect'},
        {'key': 'l', 'name': 'Etichetta', 'func': effects.apply_face_label, 'type': 'face_effect'},
        {'key': 'k', 'name': 'Movimento', 'func': None, 'type': 'motion'},
    ]

    current_filter_index = 0
    flip_mode = False
    prev_frame = None
    recording = False
    video_writer = None

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: Impossibile accedere alla webcam.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Webcam avviata.")
    print("Tasti: 0-8 filtri colore | H cappello | G occhiali | M baffi | L etichetta")
    print("       J ghost | K movimento | F flip | S screenshot | R registra | Q esci")

    prev_time = time.time()

    os.makedirs('assets/screenshots', exist_ok=True)
    os.makedirs('assets/recordings', exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore: Impossibile leggere il frame dalla webcam.")
            break

        if flip_mode:
            frame = cv2.flip(frame, 1)

        processed_frame = frame.copy()

        num_faces = 0
        current_filter = available_filters[current_filter_index]

        if current_filter['type'] == 'face_effect':
            processed_frame, num_faces = current_filter['func'](processed_frame)
        elif current_filter['type'] == 'motion':
            processed_frame = effects.apply_motion_detection(processed_frame, prev_frame)
            num_faces = effects.detect_faces_count(frame)
        elif current_filter['func'] is not None:
            processed_frame = current_filter['func'](processed_frame)
            num_faces = effects.detect_faces_count(frame)
        else:
            num_faces = effects.detect_faces_count(frame)

        prev_frame = frame.copy()

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # HUD
        filter_display = current_filter['name']
        if flip_mode:
            filter_display += " (Flip)"
        if recording:
            filter_display += " [REC]"

        processed_frame = ui.draw_hud(processed_frame, fps, filter_display, num_faces)
        processed_frame = ui.draw_filter_bar(processed_frame, available_filters, current_filter_index)

        # Indicatore registrazione
        if recording:
            cv2.circle(processed_frame, (frame_w - 30, 70), 10, (0, 0, 255), -1)

        # Scrivi frame se registrazione attiva
        if recording and video_writer is not None:
            video_writer.write(processed_frame)

        cv2.imshow('Real-Time Webcam Filters', processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assets/screenshots/screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Screenshot salvato: {filename}")
        elif key == ord('f'):
            flip_mode = not flip_mode
        elif key == ord('r'):
            if not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vid_path = f"assets/recordings/recording_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(vid_path, fourcc, 20.0, (frame_w, frame_h))
                recording = True
                print(f"Registrazione avviata: {vid_path}")
            else:
                recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Registrazione fermata.")
        else:
            for i, filt in enumerate(available_filters):
                if key == ord(filt['key']):
                    current_filter_index = i
                    break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
