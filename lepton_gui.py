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

DISPLAY_SCALE = 4   # Upscale factor for the small Lepton sensor


class LeptonCamera:
    """Wraps OpenCV capture for the PureThermal UVC device."""

    def __init__(self, device_index: int = 0):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Fall back to default backend
            self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {device_index}. "
                "Check that the PureThermal board is connected."
            )
        # Try to request Y16 for raw 16-bit data; falls back to YUYV on failure
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (w, h)

    def read_frame(self):
        """Return a normalised uint8 frame (H x W) or None on failure."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        # PureThermal Y16: frame is 16-bit single-channel
        if frame.dtype == np.uint16:
            # Normalise to 8-bit using the frame's own min/max for best contrast
            mn, mx = frame.min(), frame.max()
            if mx > mn:
                frame8 = ((frame - mn) * 255.0 / (mx - mn)).astype(np.uint8)
            else:
                frame8 = np.zeros_like(frame, dtype=np.uint8)
            return frame8
        # YUYV / BGR fallback: convert to grayscale
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
        self._img_ref = None   # keep reference to avoid GC

        # ---- status bar ----
        self.status_var = tk.StringVar(value="Camera not started.")
        tk.Label(self.root, textvariable=self.status_var,
                 anchor="w", relief="sunken").grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=6)

        # ---- controls frame ----
        ctrl = ttk.LabelFrame(self.root, text="Controls")
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        # Device index
        ttk.Label(ctrl, text="Device index:").grid(row=0, column=0, sticky="w", **pad)
        self.device_var = tk.IntVar(value=0)
        ttk.Spinbox(ctrl, from_=0, to=10, textvariable=self.device_var,
                    width=5).grid(row=0, column=1, sticky="w", **pad)

        # Start / Stop
        self.start_btn = ttk.Button(ctrl, text="Start Camera",
                                    command=self._toggle_camera)
        self.start_btn.grid(row=0, column=2, **pad)

        # Colormap
        ttk.Label(ctrl, text="Colormap:").grid(row=1, column=0, sticky="w", **pad)
        self.cmap_var = tk.StringVar(value=COLORMAPS[1][0])  # default: Ironbow
        cmap_names = [c[0] for c in COLORMAPS]
        ttk.Combobox(ctrl, textvariable=self.cmap_var,
                     values=cmap_names, state="readonly",
                     width=12).grid(row=1, column=1, sticky="w", **pad)

        # Filename
        ttk.Label(ctrl, text="Filename:").grid(row=2, column=0, sticky="w", **pad)
        self.filename_var = tk.StringVar(value="capture")
        ttk.Entry(ctrl, textvariable=self.filename_var,
                  width=20).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Label(ctrl, text=".png").grid(row=2, column=2, sticky="w")

        # Save directory
        ttk.Label(ctrl, text="Save to:").grid(row=3, column=0, sticky="w", **pad)
        self.savedir_var = tk.StringVar(value=os.path.expanduser("~"))
        ttk.Entry(ctrl, textvariable=self.savedir_var,
                  width=28).grid(row=3, column=1, columnspan=2,
                                 sticky="ew", **pad)

        # Capture button
        self.capture_btn = ttk.Button(ctrl, text="Capture Image",
                                      command=self._request_capture,
                                      state="disabled")
        self.capture_btn.grid(row=4, column=0, columnspan=3, pady=6)

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
        w, h = cam.resolution
        self.canvas.config(width=max(w, 80) * DISPLAY_SCALE,
                           height=max(h, 60) * DISPLAY_SCALE)
        self.status_var.set(
            f"Running — {w}x{h} "
            f"({'Lepton 3.x' if h >= 120 else 'Lepton 2.x'})"
        )
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _stop_camera(self):
        self.running = False
        time.sleep(0.1)
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_btn.config(text="Start Camera")
        self.capture_btn.config(state="disabled")
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
            pil_img = pil_img.resize((dw, dh), Image.NEAREST)

            tk_img = ImageTk.PhotoImage(pil_img)
            # Update canvas from main thread to be safe
            self.root.after(0, self._update_canvas, tk_img)

            if self._capture_requested:
                self._capture_requested = False
                self.root.after(0, self._save_image, frame.copy())

            time.sleep(0.033)   # ~30 fps cap

    def _update_canvas(self, tk_img):
        self._img_ref = tk_img
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)

    # ------------------------------------------------------------------ capture / save

    def _request_capture(self):
        if not self.running:
            return
        self._capture_requested = True

    def _save_image(self, gray_frame: np.ndarray):
        cmap_code = self._selected_cmap()
        pil_img = apply_colormap(gray_frame, cmap_code)

        name = self.filename_var.get().strip() or "capture"
        # Sanitise filename
        name = "".join(c for c in name if c.isalnum() or c in "-_.")
        save_dir = self.savedir_var.get().strip() or os.path.expanduser("~")
        os.makedirs(save_dir, exist_ok=True)

        # Auto-increment if file exists
        base_path = os.path.join(save_dir, name)
        path = f"{base_path}.png"
        counter = 1
        while os.path.exists(path):
            path = f"{base_path}_{counter}.png"
            counter += 1

        pil_img.save(path)
        self.status_var.set(f"Saved: {path}")

    # ------------------------------------------------------------------ helpers

    def _selected_cmap(self):
        name = self.cmap_var.get()
        for cmap_name, code in COLORMAPS:
            if cmap_name == name:
                return code
        return None

    def _on_close(self):
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
