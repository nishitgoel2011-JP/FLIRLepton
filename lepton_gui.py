"""
FLIR Lepton IR Camera GUI
Requires: opencv-python, Pillow, numpy
Hardware: PureThermal USB board (UVC device)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

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
TARGET_FPS = 9.0    # Lepton native frame rate (8.7 fps); used for VideoWriter


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


def apply_colormap(gray_frame: np.ndarray, cmap_code) -> Image.Image:
    """Apply an OpenCV colormap (or None for grayscale) and return PIL Image."""
    if cmap_code is None:
        rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    else:
        colored = cv2.applyColorMap(gray_frame, cmap_code)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_colormap_bgr(gray_frame: np.ndarray, cmap_code) -> np.ndarray:
    """Return a BGR uint8 array suitable for cv2.VideoWriter."""
    if cmap_code is None:
        return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray_frame, cmap_code)


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

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # ---- top: camera canvas ----
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
        cmap_names = [c[0] for c in COLORMAPS]
        ttk.Combobox(cam_frame, textvariable=self.cmap_var,
                     values=cmap_names, state="readonly",
                     width=12).grid(row=1, column=1, sticky="w", **pad)

        # ---- save controls (shared filename / directory) ----
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

        # recording indicator label
        self.rec_indicator = tk.Label(vid_frame, text="", fg="red",
                                      font=("TkDefaultFont", 10, "bold"))
        self.rec_indicator.grid(row=3, column=0, columnspan=2)

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
            pil_img_display = pil_img.resize((dw, dh), Image.NEAREST)
            tk_img = ImageTk.PhotoImage(pil_img_display)
            self.root.after(0, self._update_canvas, tk_img)

            # Write frame to video if recording
            if self._recording and self._video_writer is not None:
                bgr = apply_colormap_bgr(frame, cmap_code)
                self._video_writer.write(bgr)
                self._video_frame_count += 1
                elapsed = self._video_frame_count / self.fps_var.get()
                self.root.after(0, self._update_rec_indicator, elapsed)

            if self._capture_requested:
                self._capture_requested = False
                self.root.after(0, self._save_image, frame.copy())

            time.sleep(1.0 / max(self.fps_var.get(), 1))

    def _update_canvas(self, tk_img):
        self._img_ref = tk_img
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)

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

    def _save_image(self, gray_frame: np.ndarray):
        cmap_code = self._selected_cmap()
        pil_img = apply_colormap(gray_frame, cmap_code)

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
            (f for f in VIDEO_FORMATS if f[0] == fmt_name),
            VIDEO_FORMATS[0],
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
            f"Video saved: {self._video_path}  "
            f"({self._video_frame_count} frames)"
        )

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
