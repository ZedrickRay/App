# Repo-specific Copilot instructions

This project contains two Kivy-based Python apps in the workspace root:
- `app.py` — Desktop app (main sign-language app). Primary screens and camera logic live here and are wired to `my.kv`.
- `appfront.py` — Mobile-styled UI prototype. Layout lives in `appfront.kv`.

Goal for AI coding agents
- Make small, safe, cross-cutting edits that keep Kivy/KV and Python code in sync.
- Prioritize non-invasive fixes (configuration, lifecycle, UI wiring) before large refactors.

Big-picture architecture & hotspots
- UI & navigation: Kivy `.kv` files define screens and IDs. Key KV files:
  - `my.kv` (desktop screens: `MainWindow`, `FilipinoWindow`, `EnglishWindow`, `HandSigns`, `HandSign`, `InstructionsWindow`, `InstructionsWindow1`).
  - `appfront.kv` (mobile prototype screens: `MainScreen`, `CameraScreen`, `AboutScreen`).
- App entry points:
  - `app.py` loads `my.kv` via `Builder.load_file("my.kv")` and runs `MyApp()`.
  - `appfront.py` loads `appfront.kv` and runs `MyMobileApp()` for the prototype.
- Camera & imaging flow (edit here): `HandSigns` and `HandSign` screens in `app.py` open the camera in `on_enter`, read frames in a scheduled `update`, convert to a Kivy `Texture`, and assign to the `Image` widget with id `cam_view`.
  - Typical code locations to change: `HandSigns.on_enter`, `HandSigns.update`, `HandSigns.on_leave` (same for `HandSign`).
  - Texture updates use `Texture.create(...).blit_buffer(...)`. Orientation problems are commonly fixed by toggling `cv2.rotate(...)` or `texture.flip_vertical()` before blit.

Important patterns & conventions
- Kivy lifecycle: open resources (camera, threads) in `on_enter`, release/cancel in `on_leave`.
- Keep KV IDs stable: code expects `ids.cam_view`, `ids.text_output`, `ids.sentence_output`, `ids.editable_text` in `my.kv`.
- Minimal threading: current repo historically experimented with background grabbers/detectors; present code tries to keep camera capture simple and often runs `cv2.VideoCapture` in `on_enter`.
- Detection code (if present): earlier versions used a `detect_hand_and_draw` helper (MediaPipe or HSV fallback). If making changes that re-introduce detection, keep it optional behind a toggle (e.g., `DETECTION_ENABLED`) and run heavy work off the UI thread or at a low rate.

Dependencies & development commands
- The app requires (at minimum): `kivy`, `opencv-python`, and `Pillow`.
- Optional: `mediapipe` (heavy, native) — import lazily if added back.
- Typical commands to run locally (Windows PowerShell):

```powershell
# Run desktop app
python -u "c:\Users\user\Desktop\App\app.py"
# Run mobile prototype
python -u "c:\Users\user\Desktop\App\appfront.py"
```

- Recommended pip install (adjust versions for compatibility):

```powershell
pip install kivy opencv-python pillow
# optionally
pip install mediapipe
```

Debugging camera issues (project-specific)
- If the camera feed is upside-down or mirrored:
  - Check for `cv2.rotate(frame, cv2.ROTATE_180)` in `app.py` and either remove it or add it depending on the device.
  - Check for `texture.flip_vertical()` before `blit_buffer`; flipping the texture sometimes fixes OpenGL/Y-axis inversion.
- If OpenCV reports MSMF warnings like `CvCapture_MSMF::grabFrame ... can't grab frame` on Windows:
  - Try forcing DirectShow: `cv2.VideoCapture(0, cv2.CAP_DSHOW)` in the `on_enter` camera open site.
  - Lower requested FPS and resolution: set properties with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)` and `cap.set(cv2.CAP_PROP_FPS, 30)`.

Code-change guidelines for AI agents
- Small, reversible edits only: add feature flags, keep defaults unchanged unless asked.
- When modifying UI layout, update both `.kv` and Python usage of `ids` together.
- Avoid importing heavy native modules at module-import time. If adding MediaPipe, import lazily inside the worker thread and gate with a subprocess import-test if concerned about crashes.
- Tests: there are no automated tests in the repo — use manual run commands listed above to validate changes.

Examples (how to fix orientation)
- To rotate frames before blitting (in `HandSigns.update`):
```py
# before converting
frame = cv2.rotate(frame, cv2.ROTATE_180)
# then create texture and blit
```
- To flip the Kivy texture instead (alternative):
```py
tex = Texture.create(size=(w,h), colorfmt='bgr')
tex.flip_vertical()
tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
```

Where to look for common edits
- `app.py`: primary app flow, screens, camera code, and KV wiring.
- `my.kv`: desktop UI layout and IDs — keep these in sync with `app.py`.
- `appfront.py` / `appfront.kv`: lightweight mobile UI prototype.
- `Assets/` for images referenced by KV (`Image.source`) — update paths carefully on Windows.

When in doubt, ask the user
- Confirm whether to prioritize real-time (lower latency) vs. robust detection (offload detection to a worker and accept lower UI FPS).
- Confirm target platform (desktop Windows vs. Android/Kivy mobile) because OpenCV and camera backends differ.

If you want, I will commit this file as `.github/copilot-instructions.md` and iterate on any missing details you want included.
