import cv2
import numpy as np
import time
from kivy.clock import Clock
from kivy.graphics.texture import Texture


def start_mediapipe_camera(screen, src=0, interval=1.0 / 30.0, max_num_hands=2, model_complexity=0):
    """Start a camera capture and Mediapipe Hands detector attached to `screen`.

    Uses lazy import of `mediapipe` so the app won't fail if it's not installed until used.
    Creates: `screen._mp_capture`, `screen._mp_hands`, `screen._mp_drawing`, and schedules
    `mediapipe_camera_update` on the Kivy Clock.
    Returns True on success, False if mediapipe is not available or camera failed to open.
    """
    try:
        import mediapipe as mp
    except Exception:
        try:
            if hasattr(screen, 'ids') and hasattr(screen.ids, 'text_output'):
                screen.ids.text_output.text = 'mediapipe not installed'
        except Exception:
            pass
        return False

    cap = cv2.VideoCapture(src)
    if not cap or not cap.isOpened():
        try:
            if hasattr(screen, 'ids') and hasattr(screen.ids, 'text_output'):
                screen.ids.text_output.text = 'camera open failed'
        except Exception:
            pass
        return False

    # optional: set reasonable resolution
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass

    screen._mp_capture = cap
    screen._mp_module = mp
    screen._mp_hands = mp.solutions.hands.Hands(static_image_mode=False,
                                                max_num_hands=max_num_hands,
                                                model_complexity=model_complexity,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)
    screen._mp_drawing = mp.solutions.drawing_utils
    screen._mp_event = Clock.schedule_interval(lambda dt: mediapipe_camera_update(screen, dt), interval)
    screen._mp_last_label = 'None'
    return True


def stop_mediapipe_camera(screen):
    """Stop the Mediapipe camera and release resources attached to `screen`."""
    try:
        if hasattr(screen, '_mp_event') and screen._mp_event:
            screen._mp_event.cancel()
    except Exception:
        pass
    try:
        if hasattr(screen, '_mp_capture') and screen._mp_capture is not None:
            screen._mp_capture.release()
            screen._mp_capture = None
    except Exception:
        pass
    try:
        if hasattr(screen, '_mp_hands') and screen._mp_hands is not None:
            screen._mp_hands.close()
            screen._mp_hands = None
    except Exception:
        pass


def mediapipe_camera_update(screen, dt):
    """Read frame, run Mediapipe Hands, and draw a palm bounding box onto `ids.cam_view` texture."""
    if not hasattr(screen, '_mp_capture') or screen._mp_capture is None:
        return
    cap = screen._mp_capture
    ret, frame = cap.read()
    if not ret or frame is None:
        return

    # convert to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    label = 'None'

    try:
        hands = getattr(screen, '_mp_hands', None)
        mp = getattr(screen, '_mp_module', None)
        if hands is not None:
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                for hand_landmarks in results.multi_hand_landmarks:
                    # choose palm-related landmark indices to form a tight palm box
                    palm_idxs = [0, 1, 2, 5, 9, 13, 17]
                    pts = []
                    for idx in palm_idxs:
                        lm = hand_landmarks.landmark[idx]
                        x = int(max(0, min(1, lm.x)) * w)
                        y = int(max(0, min(1, lm.y)) * h)
                        pts.append((x, y))
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1 = max(min(xs) - 12, 0)
                    y1 = max(min(ys) - 12, 0)
                    x2 = min(max(xs) + 12, w - 1)
                    y2 = min(max(ys) + 12, h - 1)

                    # draw palm box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = 'Hand'
                    # draw landmarks for feedback (optional)
                    try:
                        if mp is not None and hasattr(mp, 'solutions'):
                            screen._mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    except Exception:
                        pass
                    # we only draw the first detected hand's palm box
                    break
    except Exception:
        pass

    # blit to Kivy texture
    try:
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        try:
            screen.ids.cam_view.texture = texture
        except Exception:
            pass
    except Exception:
        pass

    # optional status label
    try:
        if hasattr(screen.ids, 'text_output'):
            screen.ids.text_output.text = f"Detected: {label}"
    except Exception:
        pass
