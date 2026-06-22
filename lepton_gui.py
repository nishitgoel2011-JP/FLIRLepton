"""
FLIR Lepton IR Camera GUI
Requires: opencv-python, Pillow, numpy, ultralytics
Hardware: PureThermal USB board (UVC device)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# Available colormaps: (display name, cv2 colormap constant or None for grayscale)
COLORMAPS = [
    ("Grayscale", None),
    ("Ironbow", cv2.COLORMAP_INFERNO),
    ("Rainbow", cv2.COLORMAP_RAINBOW),
    ("Hot", cv2.COLORMAP_HOT),
    ("Jet", cv2.COLORMAP_JET),
    ("Cool", cv2.COLORMAP_COOL),
    ("Plasma", cv2.COLORMAP_PLASMA),
]

VIDEO_FORMATS = [
    ("MP4 (H.264)", ".mp4", cv2.VideoWriter_fourcc(*"mp4v")),
    ("AVI (MJPEG)", ".avi", cv2.VideoWriter_fourcc(*"MJPG")),
    ("AVI (uncompressed)", ".avi", cv2.VideoWriter_fourcc(*"XVID")),
]

DISPLAY_SCALE = 4   # Upscale factor for the small Lepton sensor
TARGET_FPS = 9.0    # Lepton native frame rate (~8.7 fps)

# HOG detection: upscale gray frame to this width before running detector
DETECT_WIDTH = 320


class LeptonCamera:
    """Wraps OpenCV capture for the PureThermal UVC device."""

    def __init__(self, device_index: int = 0):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {device_index}. "
                "Check that the PureThermal board is connected."
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (w, h)

    def read_frame(self):
        """Return a normalised uint8 frame (H x W) or None on failure."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        if frame.dtype == np.uint16:
            mn, mx = frame.min(), frame.max()
            if mx > mn:
                frame8 = ((frame - mn) * 255.0 / (mx - mn)).astype(np.uint8)
            else:
                frame8 = np.zeros_like(frame, dtype=np.uint8)
            return frame8
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.astype(np.uint8)

    def release(self):
        self.cap.release()


class PersonDetector:
    """
    YOLOv8 person detector via the ultralytics package.

    On first use the model weights (yolov8n.pt, ~6 MB) are downloaded
    automatically by ultralytics and cached in ~/.config/Ultralytics/.

    Detection is run on a 3-channel upscaled copy of the sensor frame so
    YOLO has enough resolution to work with the tiny Lepton sensor.
    Bounding boxes are returned in display coordinates.
    """

    # YOLO COCO class index for "person"
    _PERSON_CLASS = 0

    def __init__(self):
        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed.\n"
                "Run:  pip install ultralytics"
            )
        # yolov8n = nano variant — fastest, smallest; swap to yolov8s/m for
        # better accuracy at the cost of inference time
        self._model = _YOLO("yolov8n.pt")
        self._lock = threading.Lock()
        self._boxes: list[tuple[int, int, int, int]] = []  # (x,y,w,h) display coords
        self._count: int = 0

    def detect(self, gray_frame: np.ndarray, display_w: int, display_h: int,
               sensitivity: float = 0.5) -> tuple[list, int]:
        """
        Run YOLOv8 on gray_frame; return (boxes_in_display_coords, count).

        sensitivity: 0.0–1.0 slider maps to confidence threshold
                     0.0 → conf=0.10 (high recall), 1.0 → conf=0.90 (high precision)
        """
        src_h, src_w = gray_frame.shape

        # Upscale and convert to 3-channel BGR for YOLO
        scale = DETECT_WIDTH / src_w
        det_h = max(int(src_h * scale), 1)
        det_frame = cv2.resize(gray_frame, (DETECT_WIDTH, det_h),
                               interpolation=cv2.INTER_LINEAR)

        # CLAHE contrast enhancement helps on flat thermal frames
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        det_frame = clahe.apply(det_frame)
        det_bgr = cv2.cvtColor(det_frame, cv2.COLOR_GRAY2BGR)

        # Map sensitivity slider → confidence threshold (inverted)
        conf_thresh = 0.90 - sensitivity * 0.80   # 0→0.90, 1→0.10

        results = self._model.predict(
            det_bgr,
            classes=[self._PERSON_CLASS],
            conf=conf_thresh,
            verbose=False,
        )

        boxes = []
        if results and len(results[0].boxes):
            sx = display_w / DETECT_WIDTH
            sy = display_h / det_h
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((
                    int(x1 * sx), int(y1 * sy),
                    int((x2 - x1) * sx), int((y2 - y1) * sy),
                ))

        with self._lock:
            self._boxes = boxes
            self._count = len(boxes)

        return boxes, len(boxes)

    @property
    def last_count(self) -> int:
        with self._lock:
            return self._count

    @property
    def last_boxes(self) -> list:
        with self._lock:
            return list(self._boxes)


def apply_colormap(gray_frame: np.ndarray, cmap_code) -> Image.Image:
    if cmap_code is None:
        rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    else:
        colored = cv2.applyColorMap(gray_frame, cmap_code)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_colormap_bgr(gray_frame: np.ndarray, cmap_code) -> np.ndarray:
    if cmap_code is None:
        return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray_frame, cmap_code)


def draw_detections(pil_img: Image.Image, boxes: list, count: int) -> Image.Image:
    """Draw bounding boxes and person count badge onto a PIL RGB image."""
    if not boxes and count == 0:
        return pil_img

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    box_color = (0, 255, 80)       # bright green
    label_bg = (0, 180, 60)

    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=box_color, width=2)
        label = "Person"
        tw, th = 50, 14
        draw.rectangle([x, y - th - 2, x + tw, y], fill=label_bg)
        draw.text((x + 2, y - th - 1), label, fill=(255, 255, 255))

    # Count badge — top-right corner
    badge = f"Persons: {count}"
    bw, bh = 110, 22
    iw, ih = img.size
    draw.rectangle([iw - bw - 4, 4, iw - 4, 4 + bh], fill=(20, 20, 20))
    draw.text((iw - bw, 6), badge, fill=(0, 255, 80))

    return img


class LeptonGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FLIR Lepton IR Camera")
        self.root.resizable(False, False)

        self.camera: LeptonCamera | None = None
        self.running = False
        self._lock = threading.Lock()
        self._last_gray: np.ndarray | None = None
        self._capture_requested = False

        # Video recording state
        self._video_writer: cv2.VideoWriter | None = None
        self._recording = False
        self._video_frame_count = 0
        self._video_path = ""

        # Person detection
        self._detector = PersonDetector()
        self._detect_enabled = False
        self._detect_frame_skip = 0   # counter for throttling
        self._last_boxes: list = []
        self._last_count: int = 0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # ---- camera canvas ----
        self.canvas = tk.Canvas(self.root, bg="black",
                                width=80 * DISPLAY_SCALE,
                                height=60 * DISPLAY_SCALE)
        self.canvas.grid(row=0, column=0, columnspan=2, **pad)
        self._img_ref = None

        # ---- status bar ----
        self.status_var = tk.StringVar(value="Camera not started.")
        tk.Label(self.root, textvariable=self.status_var,
                 anchor="w", relief="sunken").grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=6)

        # ---- camera controls ----
        cam_frame = ttk.LabelFrame(self.root, text="Camera")
        cam_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(cam_frame, text="Device index:").grid(row=0, column=0, sticky="w", **pad)
        self.device_var = tk.IntVar(value=0)
        ttk.Spinbox(cam_frame, from_=0, to=10, textvariable=self.device_var,
                    width=5).grid(row=0, column=1, sticky="w", **pad)

        self.start_btn = ttk.Button(cam_frame, text="Start Camera",
                                    command=self._toggle_camera)
        self.start_btn.grid(row=0, column=2, **pad)

        ttk.Label(cam_frame, text="Colormap:").grid(row=1, column=0, sticky="w", **pad)
        self.cmap_var = tk.StringVar(value=COLORMAPS[1][0])
        ttk.Combobox(cam_frame, textvariable=self.cmap_var,
                     values=[c[0] for c in COLORMAPS],
                     state="readonly", width=12).grid(row=1, column=1, sticky="w", **pad)

        # ---- save settings ----
        save_frame = ttk.LabelFrame(self.root, text="Save Settings")
        save_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(save_frame, text="Filename:").grid(row=0, column=0, sticky="w", **pad)
        self.filename_var = tk.StringVar(value="capture")
        ttk.Entry(save_frame, textvariable=self.filename_var,
                  width=22).grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(save_frame, text="Save to:").grid(row=1, column=0, sticky="w", **pad)
        self.savedir_var = tk.StringVar(value=os.path.join(os.getcwd(), 'Output'))
        ttk.Entry(save_frame, textvariable=self.savedir_var,
                  width=30).grid(row=1, column=1, columnspan=2, sticky="ew", **pad)

        # ---- image capture ----
        img_frame = ttk.LabelFrame(self.root, text="Image Capture")
        img_frame.grid(row=4, column=0, sticky="ew", **pad)

        self.capture_btn = ttk.Button(img_frame, text="Capture Image",
                                      command=self._request_capture,
                                      state="disabled")
        self.capture_btn.grid(row=0, column=0, padx=8, pady=6)
        ttk.Label(img_frame, text="→ saves as <filename>.png").grid(
            row=0, column=1, sticky="w", padx=4)

        # ---- video recording ----
        vid_frame = ttk.LabelFrame(self.root, text="Video Recording")
        vid_frame.grid(row=4, column=1, sticky="ew", **pad)

        ttk.Label(vid_frame, text="Format:").grid(row=0, column=0, sticky="w", **pad)
        self.vfmt_var = tk.StringVar(value=VIDEO_FORMATS[0][0])
        ttk.Combobox(vid_frame, textvariable=self.vfmt_var,
                     values=[f[0] for f in VIDEO_FORMATS],
                     state="readonly", width=18).grid(row=0, column=1, **pad)

        ttk.Label(vid_frame, text="FPS:").grid(row=1, column=0, sticky="w", **pad)
        self.fps_var = tk.DoubleVar(value=TARGET_FPS)
        ttk.Spinbox(vid_frame, from_=1.0, to=30.0, increment=0.5,
                    textvariable=self.fps_var, width=6,
                    format="%.1f").grid(row=1, column=1, sticky="w", **pad)

        self.record_btn = ttk.Button(vid_frame, text="Start Recording",
                                     command=self._toggle_recording,
                                     state="disabled")
        self.record_btn.grid(row=2, column=0, columnspan=2, pady=6)

        self.rec_indicator = tk.Label(vid_frame, text="", fg="red",
                                      font=("TkDefaultFont", 10, "bold"))
        self.rec_indicator.grid(row=3, column=0, columnspan=2)

        # ---- person detection ----
        det_frame = ttk.LabelFrame(self.root, text="Person Detection")
        det_frame.grid(row=5, column=0, columnspan=2, sticky="ew", **pad)

        # Enable toggle
        self.detect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_frame, text="Enable live person detection",
                        variable=self.detect_var,
                        command=self._on_detect_toggle).grid(
            row=0, column=0, columnspan=3, sticky="w", **pad)

        # Sensitivity slider
        ttk.Label(det_frame, text="Sensitivity:").grid(row=1, column=0, sticky="w", **pad)
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(det_frame, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.sensitivity_var,
                  length=140).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Label(det_frame, text="Low → High").grid(row=1, column=2, sticky="w")

        # Overlay on saved files checkbox
        self.overlay_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(det_frame,
                        text="Include detection overlay in saved images/video",
                        variable=self.overlay_save_var).grid(
            row=2, column=0, columnspan=3, sticky="w", **pad)

        # Large person count display
        count_panel = tk.Frame(det_frame, bg="#1a1a2e", bd=2, relief="sunken")
        count_panel.grid(row=3, column=0, columnspan=3, sticky="ew",
                         padx=6, pady=(2, 6))

        tk.Label(count_panel, text="Persons detected:",
                 bg="#1a1a2e", fg="#aaaacc",
                 font=("TkDefaultFont", 10)).pack(side="left", padx=8)

        self.count_var = tk.StringVar(value="—")
        tk.Label(count_panel, textvariable=self.count_var,
                 bg="#1a1a2e", fg="#00ff50",
                 font=("TkDefaultFont", 28, "bold"),
                 width=3, anchor="e").pack(side="left", padx=4)

        self.det_status_var = tk.StringVar(value="Detection off")
        tk.Label(count_panel, textvariable=self.det_status_var,
                 bg="#1a1a2e", fg="#888888",
                 font=("TkDefaultFont", 9)).pack(side="left", padx=12)

    # ------------------------------------------------------------------ camera control

    def _toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        idx = self.device_var.get()
        try:
            cam = LeptonCamera(idx)
        except RuntimeError as e:
            messagebox.showerror("Camera Error", str(e))
            return
        self.camera = cam
        self.running = True
        self.start_btn.config(text="Stop Camera")
        self.capture_btn.config(state="normal")
        self.record_btn.config(state="normal")
        w, h = cam.resolution
        self.canvas.config(width=max(w, 80) * DISPLAY_SCALE,
                           height=max(h, 60) * DISPLAY_SCALE)
        self.status_var.set(
            f"Running — {w}x{h} "
            f"({'Lepton 3.x' if h >= 120 else 'Lepton 2.x'})"
        )
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _stop_camera(self):
        if self._recording:
            self._stop_recording()
        self.running = False
        time.sleep(0.1)
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_btn.config(text="Start Camera")
        self.capture_btn.config(state="disabled")
        self.record_btn.config(state="disabled")
        self.status_var.set("Camera stopped.")

    # ------------------------------------------------------------------ capture loop

    def _capture_loop(self):
        while self.running:
            t0 = time.monotonic()

            frame = self.camera.read_frame() if self.camera else None
            if frame is None:
                time.sleep(0.05)
                continue

            with self._lock:
                self._last_gray = frame.copy()

            cmap_code = self._selected_cmap()
            pil_img = apply_colormap(frame, cmap_code)

            # Scale up for display
            dw = self.canvas.winfo_width() or pil_img.width * DISPLAY_SCALE
            dh = self.canvas.winfo_height() or pil_img.height * DISPLAY_SCALE
            display_img = pil_img.resize((dw, dh), Image.NEAREST)

            # --- Person detection (runs every 3rd frame to stay smooth) ---
            boxes, count = [], 0
            if self._detect_enabled:
                self._detect_frame_skip += 1
                if self._detect_frame_skip >= 3:
                    self._detect_frame_skip = 0
                    sensitivity = self.sensitivity_var.get()  # 0.0–1.0
                    boxes, count = self._detector.detect(
                        frame, dw, dh, sensitivity=sensitivity
                    )
                    self._last_boxes = boxes
                    self._last_count = count
                    self.root.after(0, self._update_count_display, count)
                else:
                    boxes = self._last_boxes
                    count = self._last_count

                display_img = draw_detections(display_img, boxes, count)

            tk_img = ImageTk.PhotoImage(display_img)
            self.root.after(0, self._update_canvas, tk_img)

            # --- Video recording ---
            if self._recording and self._video_writer is not None:
                if self.overlay_save_var.get() and self._detect_enabled:
                    bgr = cv2.cvtColor(np.array(
                        draw_detections(pil_img.resize((dw, dh), Image.NEAREST),
                                        self._last_boxes, self._last_count)
                    ), cv2.COLOR_RGB2BGR)
                    # Resize back to native sensor resolution for the file
                    wr_w, wr_h = self.camera.resolution
                    bgr = cv2.resize(bgr, (wr_w, wr_h))
                else:
                    bgr = apply_colormap_bgr(frame, cmap_code)
                self._video_writer.write(bgr)
                self._video_frame_count += 1
                elapsed = self._video_frame_count / self.fps_var.get()
                self.root.after(0, self._update_rec_indicator, elapsed)

            # --- Image capture ---
            if self._capture_requested:
                self._capture_requested = False
                overlay = self._detect_enabled and self.overlay_save_var.get()
                self.root.after(0, self._save_image, frame.copy(),
                                list(self._last_boxes), self._last_count, overlay)

            # Pace to target FPS
            elapsed = time.monotonic() - t0
            sleep = max(0.0, (1.0 / max(self.fps_var.get(), 1)) - elapsed)
            time.sleep(sleep)

    def _update_canvas(self, tk_img):
        self._img_ref = tk_img
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)

    def _update_count_display(self, count: int):
        self.count_var.set(str(count))
        self.det_status_var.set(
            "No persons" if count == 0 else
            f"{'Person' if count == 1 else 'Persons'} in frame"
        )

    def _update_rec_indicator(self, elapsed_seconds: float):
        m, s = divmod(int(elapsed_seconds), 60)
        self.rec_indicator.config(
            text=f"● REC  {m:02d}:{s:02d}  ({self._video_frame_count} frames)"
        )

    # ------------------------------------------------------------------ image save

    def _request_capture(self):
        if not self.running:
            return
        self._capture_requested = True

    def _save_image(self, gray_frame: np.ndarray, boxes: list,
                    count: int, with_overlay: bool):
        cmap_code = self._selected_cmap()
        pil_img = apply_colormap(gray_frame, cmap_code)
        if with_overlay:
            dw = self.canvas.winfo_width() or pil_img.width * DISPLAY_SCALE
            dh = self.canvas.winfo_height() or pil_img.height * DISPLAY_SCALE
            pil_img = draw_detections(
                pil_img.resize((dw, dh), Image.NEAREST), boxes, count
            )

        name = self._sanitise(self.filename_var.get() or "capture")
        save_dir = self.savedir_var.get().strip() or os.path.expanduser("~")
        os.makedirs(save_dir, exist_ok=True)
        path = self._unique_path(save_dir, name, ".png")
        pil_img.save(path)
        self.status_var.set(f"Image saved: {path}")

    # ------------------------------------------------------------------ video recording

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.running or self.camera is None:
            return
        name = self._sanitise(self.filename_var.get() or "capture")
        save_dir = self.savedir_var.get().strip() or os.path.expanduser("~")
        os.makedirs(save_dir, exist_ok=True)
        fmt_name = self.vfmt_var.get()
        _, ext, fourcc = next(
            (f for f in VIDEO_FORMATS if f[0] == fmt_name), VIDEO_FORMATS[0]
        )
        path = self._unique_path(save_dir, name, ext)
        w, h = self.camera.resolution
        writer = cv2.VideoWriter(path, fourcc, self.fps_var.get(), (w, h))
        if not writer.isOpened():
            messagebox.showerror(
                "Video Error",
                f"Could not open VideoWriter for {path}.\n"
                "Try a different format or check codec availability."
            )
            return
        self._video_writer = writer
        self._video_path = path
        self._video_frame_count = 0
        self._recording = True
        self.record_btn.config(text="Stop Recording")
        self.rec_indicator.config(text="● REC  00:00  (0 frames)")
        self.status_var.set(f"Recording → {path}")

    def _stop_recording(self):
        self._recording = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self.record_btn.config(text="Start Recording")
        self.rec_indicator.config(text="")
        self.status_var.set(
            f"Video saved: {self._video_path}  ({self._video_frame_count} frames)"
        )

    # ------------------------------------------------------------------ detection toggle

    def _on_detect_toggle(self):
        if self.detect_var.get() and not _YOLO_AVAILABLE:
            messagebox.showerror(
                "Missing dependency",
                "ultralytics is not installed.\n\nRun:\n  pip install ultralytics\n\n"
                "YOLOv8n weights (~6 MB) will be downloaded automatically on first use."
            )
            self.detect_var.set(False)
            return

        self._detect_enabled = self.detect_var.get()
        if self._detect_enabled:
            self.count_var.set("0")
            self.det_status_var.set("Scanning…")
        else:
            self.count_var.set("—")
            self.det_status_var.set("Detection off")
            self._last_boxes = []
            self._last_count = 0

    # ------------------------------------------------------------------ helpers

    def _selected_cmap(self):
        name = self.cmap_var.get()
        for cmap_name, code in COLORMAPS:
            if cmap_name == name:
                return code
        return None

    @staticmethod
    def _sanitise(name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in "-_.") or "capture"

    @staticmethod
    def _unique_path(directory: str, base: str, ext: str) -> str:
        path = os.path.join(directory, base + ext)
        counter = 1
        while os.path.exists(path):
            path = os.path.join(directory, f"{base}_{counter}{ext}")
            counter += 1
        return path

    def _on_close(self):
        if self._recording:
            self._stop_recording()
        self.running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    LeptonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
