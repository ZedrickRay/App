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


# --- TFLite lazy loader/cache ---
_GLOBAL_TFLITE = None
_GLOBAL_TFLITE_PATH = None

def _load_tflite_interpreter(model_path):
    """Try to load a TFLite Interpreter from `tflite_runtime` or `tensorflow`.
    Returns an interpreter or None on failure.
    """
    Interpreter = None
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except Exception:
            try:
                from tensorflow.lite import Interpreter
            except Exception:
                Interpreter = None
            else:
                Interpreter = Interpreter
        else:
            Interpreter = Interpreter
    else:
        Interpreter = Interpreter

    if Interpreter is None:
        return None
    try:
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except Exception:
        return None


def _get_default_interpreter(paths=("Assets/best_float32.tflite", "Assets/hand.tflite")):
    global _GLOBAL_TFLITE, _GLOBAL_TFLITE_PATH
    if _GLOBAL_TFLITE is not None:
        return _GLOBAL_TFLITE
    for p in paths:
        try:
            interp = _load_tflite_interpreter(p)
            if interp is not None:
                _GLOBAL_TFLITE = interp
                _GLOBAL_TFLITE_PATH = p
                return interp
        except Exception:
            continue
    return None


def start_tflite(screen, model_path="Assets/best_float32.tflite"):
    """Load the specified TFLite model and attach the interpreter to the screen as
    `screen._tflite_interp`. Returns True if loaded, False otherwise.
    """
    try:
        interp = _load_tflite_interpreter(model_path)
        if interp is None:
            try:
                if hasattr(screen, 'ids') and hasattr(screen.ids, 'text_output'):
                    screen.ids.text_output.text = 'tflite load failed'
            except Exception:
                pass
            return False
        screen._tflite_interp = interp
        return True
    except Exception:
        return False


def stop_tflite(screen):
    """Remove a previously attached TFLite interpreter from the screen."""
    try:
        if hasattr(screen, '_tflite_interp'):
            screen._tflite_interp = None
    except Exception:
        pass
    return True

def start_camera(screen, src=0, interval=1.0/30.0, detection=True, detection_interval=3, proc_width=320, proc_height=240):
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
    # try to set a stable capture resolution (may be ignored by some backends)
    try:
        screen.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(proc_width))
        screen.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(proc_height))
    except Exception:
        pass
    # store processing size for detection scaling
    screen._proc_size = (int(proc_width), int(proc_height))
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
    # store previous small grayscale frame for motion checks (used to reject static lights)
    screen._prev_small = None
    # motion configuration: fraction of pixels in bbox that must change to accept as moving
    screen._motion_threshold = 0.006  # ~0.6% of bbox pixels
    # time window (seconds) to allow static hand without requiring motion after a recent detection
    screen._motion_grace_period = 1.0
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
                frame, label = detect_hand_and_draw(frame, screen, proc_size=getattr(screen, '_proc_size', (320, 240)))
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


def detect_hand_and_draw(frame, screen=None, proc_size=None):
    """Lightweight hand detector using HSV skin-color contour.
    Returns (annotated_frame, label) where label is 'Hand' or 'None'.
    If `screen` is provided, the function will smooth bounding boxes across frames
    using `screen._last_box` and `screen._box_alpha`.
    """
    label = 'None'
    if frame is None:
        return frame, label

    # Try TFLite keypoint model first (if available). If it returns keypoints,
    # build a bounding box from those points and return immediately.
    try:
        interp = None
        if screen is not None and hasattr(screen, '_tflite_interp'):
            interp = getattr(screen, '_tflite_interp')
        if interp is None:
            interp = _get_default_interpreter()
        if interp is not None:
            try:
                orig_h, orig_w = frame.shape[:2]
                input_details = interp.get_input_details()
                output_details = interp.get_output_details()
                inp = input_details[0]
                # input shape may be (1,H,W,C)
                h_in = int(inp['shape'][1])
                w_in = int(inp['shape'][2])
                dtype = inp['dtype']

                small = cv2.resize(frame, (w_in, h_in))
                if dtype == np.uint8:
                    in_data = small.astype(np.uint8)
                else:
                    in_data = small.astype(np.float32) / 255.0
                in_data = np.expand_dims(in_data, axis=0)
                interp.set_tensor(input_details[0]['index'], in_data)
                interp.invoke()

                kp = None
                for out in output_details:
                    try:
                        o = interp.get_tensor(out['index'])
                    except Exception:
                        o = None
                    if o is None:
                        continue
                    # prefer shapes like (1, N, 2) or (N,2)
                    if o.ndim == 3 and o.shape[0] == 1 and (o.shape[2] == 2 or o.shape[2] == 3):
                        kp = o[0]
                        break
                    if o.ndim == 2 and (o.shape[1] == 2 or o.shape[1] == 3):
                        kp = o
                        break
                    # flattened single-row outputs that include keypoints
                    if o.ndim == 2 and o.shape[0] == 1:
                        flat = o.flatten()
                        if flat.size >= 42:
                            try:
                                kp = flat[:42].reshape(21, 2)
                                break
                            except Exception:
                                pass

                if kp is not None:
                    flat = kp.reshape(-1, kp.shape[-1])
                    xs = flat[:, 0].astype(float)
                    ys = flat[:, 1].astype(float)
                    # normalized coordinates (<=1.5) or absolute pixel coords
                    if xs.max() <= 1.5 and ys.max() <= 1.5:
                        xs = (xs * orig_w).astype(int)
                        ys = (ys * orig_h).astype(int)
                    else:
                        xs = xs.astype(int)
                        ys = ys.astype(int)

                    xmin = int(max(xs.min() - 10, 0))
                    ymin = int(max(ys.min() - 10, 0))
                    xmax = int(min(xs.max() + 10, orig_w - 1))
                    ymax = int(min(ys.max() + 10, orig_h - 1))

                    # shrink lower part to focus on palm (avoid forearm)
                    height = ymax - ymin
                    if height > 12:
                        ymax = max(ymin + int(height * 0.75), ymin + 12)

                    # draw keypoints and bbox
                    for (X, Y) in zip(xs, ys):
                        try:
                            cv2.circle(frame, (int(X), int(Y)), 3, (0, 0, 255), -1)
                        except Exception:
                            pass
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    label = 'Hand'
                    return frame, label
            except Exception:
                pass
    except Exception:
        pass
    # allow caller to provide a processing size (width, height)
    if proc_size is None and screen is not None:
        proc_size = getattr(screen, '_proc_size', (320, 240))
    if proc_size is None:
        proc_size = (320, 240)

    try:
        orig_h, orig_w = frame.shape[:2]
        proc_w, proc_h = int(proc_size[0]), int(proc_size[1])

        # resize to processing size for stable detection
        small = cv2.resize(frame, (proc_w, proc_h))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # compute small grayscale for motion checks
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # tighter skin color range in HSV to reduce bright non-skin detections
        lower_hsv = np.array([0, 40, 50], dtype=np.uint8)
        upper_hsv = np.array([22, 255, 235], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # additional skin detection in YCrCb space reduces false positives from bright lights
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        # narrower YCrCb Cr/ Cb selection to avoid bright lights
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 170, 125], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # combine masks to be more selective
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # morphological ops to remove noise and connect hand regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel2, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        proc_area = float(proc_w * proc_h)
        min_area = max(200, int(proc_area * 0.002))

        chosen_box = None

        if contours:
            # prefer the largest contours, ignore tiny noise
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            # sort by area descending
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            chosen = None
            # compute frame center for preference
            center_x = proc_w / 2.0
            center_y = proc_h / 2.0
            max_center_dist = max(proc_w, proc_h) * 0.6

            for c in contours:
                area = cv2.contourArea(c)
                # require reasonable area fraction
                if area < min_area:
                    continue

                # hull / solidity checks
                hull_pts = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull_pts) if hull_pts is not None else 0
                solidity = float(area) / hull_area if hull_area > 0 else 0

                # bounding box and ratios
                x, y, w_box, h_box = cv2.boundingRect(hull_pts)
                if w_box == 0 or h_box == 0:
                    continue
                aspect = float(w_box) / float(h_box)
                bbox_area = float(w_box * h_box)
                bbox_fill = float(area) / bbox_area if bbox_area > 0 else 0

                # ignore contours touching edges (likely background/arm) or extreme aspect ratios
                if x <= 3 or y <= 3 or x + w_box >= proc_w - 3 or y + h_box >= proc_h - 3:
                    continue
                if aspect < 0.35 or aspect > 2.2:
                    continue
                if solidity < 0.45:
                    continue
                if bbox_fill < 0.30:
                    continue

                # compute centroid distance to center; prefer centered contours
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + w_box // 2, y + h_box // 2
                center_dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
                if center_dist > max_center_dist:
                    # too far from center, skip
                    continue

                # reject bounding boxes that are extremely bright (likely lights/reflections)
                try:
                    v_channel = hsv[y:y + h_box, x:x + w_box, 2]
                    mean_v = float(np.mean(v_channel)) if v_channel.size > 0 else 255.0
                    if mean_v > 240 or mean_v < 30:
                        continue
                except Exception:
                    pass

                # motion check: prefer regions that have changed recently to avoid static lights
                try:
                    prev_small = getattr(screen, '_prev_small', None) if screen is not None else None
                    last_time = getattr(screen, '_last_detect_time', 0.0) if screen is not None else 0.0
                    motion_needed = True
                    if screen is not None:
                        # if we recently detected a hand, allow it to remain still for a grace period
                        if time.time() - last_time <= getattr(screen, '_motion_grace_period', 1.0):
                            motion_needed = False
                    if prev_small is not None and motion_needed:
                        # crop prev and current grayscale regions
                        try:
                            prev_crop = prev_small[y:y + h_box, x:x + w_box]
                            cur_crop = small_gray[y:y + h_box, x:x + w_box]
                            if prev_crop.size == cur_crop.size and prev_crop.size > 0:
                                diff = cv2.absdiff(cur_crop, prev_crop)
                                _, diff_t = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                                motion_frac = float(np.count_nonzero(diff_t)) / float(diff_t.size)
                                if motion_frac < getattr(screen, '_motion_threshold', 0.006):
                                    # not enough motion, likely a static light or reflection
                                    continue
                        except Exception:
                            pass
                except Exception:
                    pass

                # convexity defects: count finger-like gaps
                hull_idx = cv2.convexHull(c, returnPoints=False)
                defects = None
                defect_count = 0
                try:
                    if hull_idx is not None and len(hull_idx) > 3:
                        defects = cv2.convexityDefects(c, hull_idx)
                        if defects is not None:
                            perim = cv2.arcLength(c, True)
                            for i in range(defects.shape[0]):
                                s, e, f, depth = defects[i, 0]
                                depth_val = depth / 256.0
                                if depth_val > max(10.0, 0.02 * perim):
                                    defect_count += 1
                except Exception:
                    defect_count = 0

                # require at least 2-3 defects (fingers) for a confident hand, or accept if very solid and central
                if defect_count >= 2 or (solidity > 0.6 and bbox_fill > 0.45):
                    chosen = (hull_pts, x, y, w_box, h_box)
                    break

            if chosen is not None:
                hull_pts, x, y, w_box, h_box = chosen
                # expand slightly
                pad = int(max(w_box, h_box) * 0.18) + 4
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + w_box + pad, proc_w - 1)
                y2 = min(y + h_box + pad, proc_h - 1)

                # reject boxes that cover too much of processing frame (likely background/false)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area <= proc_area * 0.65:
                    chosen_box = (x1, y1, x2, y2)

        # if we have a chosen box in proc coords, scale to original frame coords
        if chosen_box is not None:
            x1, y1, x2, y2 = chosen_box
            sx = float(orig_w) / float(proc_w)
            sy = float(orig_h) / float(proc_h)
            ox1 = int(x1 * sx)
            oy1 = int(y1 * sy)
            ox2 = int(x2 * sx)
            oy2 = int(y2 * sy)

            # smoothing on original coords
            if screen is not None:
                last = getattr(screen, '_last_box', None)
                alpha = getattr(screen, '_box_alpha', 0.6)
                if last is not None:
                    lx1, ly1, lx2, ly2 = last
                    ox1 = int(lx1 * alpha + ox1 * (1 - alpha))
                    oy1 = int(ly1 * alpha + oy1 * (1 - alpha))
                    ox2 = int(lx2 * alpha + ox2 * (1 - alpha))
                    oy2 = int(ly2 * alpha + oy2 * (1 - alpha))
                screen._last_box = (ox1, oy1, ox2, oy2)
                screen._last_detect_time = time.time()

            # draw rectangle on original frame
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
            label = 'Hand'
            (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            txt_x1 = ox1
            txt_y1 = max(oy1 - txt_h - 8, 0)
            txt_x2 = ox1 + txt_w + 8
            txt_y2 = oy1
            cv2.rectangle(frame, (txt_x1, txt_y1), (txt_x2, txt_y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (ox1 + 4, oy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # no detection: draw last valid box briefly if present
            if screen is not None and getattr(screen, '_last_box', None) is not None:
                last_time = getattr(screen, '_last_detect_time', 0.0)
                persist = getattr(screen, '_detection_persistence', 0.6)
                if time.time() - last_time <= persist:
                    ox1, oy1, ox2, oy2 = screen._last_box
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                    label = 'Hand'
        # update previous small grayscale for next frame motion checks
        try:
            if screen is not None:
                screen._prev_small = small_gray.copy()
        except Exception:
            pass
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
#checkpoint
