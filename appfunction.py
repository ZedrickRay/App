import cv2
import numpy as np
import time
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from PIL import Image as PILImage

# Camera orientation handling:
# - 'normal' : show frame as-captured
# - 'flip_texture' : flip the Kivy texture vertically before blit (fixes OpenGL Y-inversion)
# - 'rotate' : rotate the frame 180 degrees in OpenCV before creating texture
CAMERA_ORIENTATION = 'rotate'

def start_camera(screen, src=0, interval=1.0/30.0, detection=True, detection_interval=3):
    """Attach a cv2.VideoCapture to the provided Kivy Screen and start the update loop.
    The screen must have an `ids.cam_view` Image in its kv definition.

    Parameters:
    - screen: Kivy Screen instance with `ids.cam_view` and optional `ids.text_output`.
    - src: camera device index (default 0)
    - interval: Clock interval for updates
    - detection: enable lightweight hand detection (skin-contour) and draw box
    - detection_interval: run detection every N frames to reduce CPU
    """
    screen.capture = cv2.VideoCapture(src)
    # detection controls
    screen._detection_enabled = bool(detection)
    screen._detection_interval = max(1, int(detection_interval))
    screen._detection_counter = 0
    # smoothing state for detection bounding box
    screen._last_box = None
    screen._box_alpha = 0.6  # smoothing factor between 0..1 (higher = less lag)
    # persistence: keep drawing last box for a short time when detection briefly fails
    screen._last_detect_time = 0.0
    screen._detection_persistence = 0.6  # seconds to keep last box when detection is lost
    # schedule periodic update; capture update function is camera_update
    screen._camera_event = Clock.schedule_interval(lambda dt: camera_update(screen, dt), interval)


def stop_camera(screen):
    """Stop camera update and release capture if present."""
    try:
        if hasattr(screen, '_camera_event') and screen._camera_event:
            screen._camera_event.cancel()
    except Exception:
        pass
    try:
        if hasattr(screen, 'capture') and screen.capture is not None:
            screen.capture.release()
            screen.capture = None
    except Exception:
        pass

def camera_update(screen, dt):
    """Read a frame from `screen.capture`, handle orientation, and blit to `ids.cam_view.texture`."""
    if not hasattr(screen, 'capture') or screen.capture is None:
        return
    ret, frame = screen.capture.read()
    if not ret or frame is None:
        return
    # handle orientation
    if CAMERA_ORIENTATION == 'rotate':
        try:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        except Exception:
            pass

    # optional detection: run every N frames
    if getattr(screen, '_detection_enabled', False):
        screen._detection_counter += 1
        if screen._detection_counter >= getattr(screen, '_detection_interval', 3):
            screen._detection_counter = 0
            try:
                frame, label = detect_hand_and_draw(frame, screen)
                # update a small status label if the kv provides it
                try:
                    if hasattr(screen.ids, 'text_output'):
                        screen.ids.text_output.text = f"Detected: {label}"
                except Exception:
                    pass
            except Exception:
                pass

    # convert and blit
    try:
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        if CAMERA_ORIENTATION == 'flip_texture':
            try:
                texture.flip_vertical()
            except Exception:
                pass
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        screen.ids.cam_view.texture = texture
    except Exception:
        pass


def detect_hand_and_draw(frame, screen=None):
    """Lightweight hand detector using HSV skin-color contour.
    Returns (annotated_frame, label) where label is 'Hand' or 'None'.
    If `screen` is provided, the function will smooth bounding boxes across frames
    using `screen._last_box` and `screen._box_alpha`.
    """
    label = 'None'
    if frame is None:
        return frame, label

    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin color range; may need tuning per camera/lighting
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # morphological ops to fill gaps and remove noise, then dilate to connect parts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.medianBlur(mask, 5)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel2, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Adaptive area thresholds based on frame size
        fh, fw = frame.shape[0], frame.shape[1]
        frame_area = float(fh * fw)
        min_total_area = max(500, int(frame_area * 0.001))

        if contours:
            # consider multiple contours (sometimes hand segments are split)
            small_thresh = 300
            large_contours = [c for c in contours if cv2.contourArea(c) > small_thresh]
            total_area = sum(int(cv2.contourArea(c)) for c in large_contours)

            if total_area >= min_total_area and large_contours:
                # combine points from all large contours and compute convex hull
                pts = np.vstack([c.reshape(-1, 2) for c in large_contours])
                hull = cv2.convexHull(pts)
                x, y, w_box, h_box = cv2.boundingRect(hull)

                # expand box slightly to ensure whole hand is included
                pad = int(max(w_box, h_box) * 0.18) + 8
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + w_box + pad, fw - 1)
                y2 = min(y + h_box + pad, fh - 1)

                # temporal smoothing with exponential moving average
                if screen is not None:
                    last = getattr(screen, '_last_box', None)
                    alpha = getattr(screen, '_box_alpha', 0.6)
                    if last is not None:
                        lx1, ly1, lx2, ly2 = last
                        x1 = int(lx1 * alpha + x1 * (1 - alpha))
                        y1 = int(ly1 * alpha + y1 * (1 - alpha))
                        x2 = int(lx2 * alpha + x2 * (1 - alpha))
                        y2 = int(ly2 * alpha + y2 * (1 - alpha))
                    screen._last_box = (x1, y1, x2, y2)
                    screen._last_detect_time = time.time()

                # draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = 'Hand'
                # draw label background
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                txt_x1 = x1
                txt_y1 = max(y1 - txt_h - 8, 0)
                txt_x2 = x1 + txt_w + 8
                txt_y2 = y1
                cv2.rectangle(frame, (txt_x1, txt_y1), (txt_x2, txt_y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # if detection briefly fails, keep last box for short persistence
                if screen is not None and getattr(screen, '_last_box', None) is not None:
                    last_time = getattr(screen, '_last_detect_time', 0.0)
                    persist = getattr(screen, '_detection_persistence', 0.6)
                    if time.time() - last_time <= persist:
                        x1, y1, x2, y2 = screen._last_box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = 'Hand'
    except Exception:
        pass

    return frame, label

def save_uploaded_image(path):
    """Save a selected image path to a local file (keeps simple logic used by UI)."""
    try:
        img = PILImage.open(path)
        img.save("uploaded_image.jpg")
        return True
    except Exception:
        return False
