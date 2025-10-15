#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# latx.py — Color primario por rostro (pantalla sólida) + DMX 22–56

import os, time, argparse, threading
from array import array
import cv2
import numpy as np
from ola.ClientWrapper import ClientWrapper

UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56
DMX_FPS = 44

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ---------- DMX ----------
class DMXOut:
    def __init__(self, universe=UNIVERSE, fps=DMX_FPS):
        self.universe = universe
        self.period = 1.0 / fps
        self.wrapper = ClientWrapper()
        self.client = self.wrapper.Client()
        self.dmx = array('B', [0]*512)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def _send_now(self):
        with self._lock:
            payload = self.dmx
        try:
            self.client.SendDmx(self.universe, payload, lambda s: None)  # array('B') OK
        except Exception:
            pass

    def set_rgb_color_over_range(self, lo, hi, rgb):
        r,g,b = [clamp(int(c), 0, 255) for c in rgb]
        span = hi - lo + 1
        with self._lock:
            for i in range(span):
                ch = lo - 1 + i
                m = i % 3
                self.dmx[ch] = (r,g,b)[m]
        self._send_now()

    def blackout(self):
        with self._lock:
            self.dmx[:] = array('B', [0]*512)
        self._send_now()

    def _loop(self):
        next_t = time.time()
        while not self._stop.is_set():
            self._send_now()
            next_t += self.period
            time.sleep(max(0, next_t - time.time()))

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.blackout()

# ---------- Color por rostro ----------
class FacePrimaryColor:
    """
    Calcula color promedio BGR del rostro más grande,
    lo suaviza (EMA) y lo reduce a un primario (R, G o B).
    Sin rostro: decae a negro.
    """
    def __init__(self, models_dir, ema_alpha=0.25, decay_per_sec=0.25, min_level=32):
        haar = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        self.det = cv2.CascadeClassifier(haar)
        if self.det.empty():
            raise RuntimeError('No se pudo cargar haarcascade_frontalface_default.xml')
        self.color_ema = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # BGR
        self.ema_alpha   = float(ema_alpha)
        self.decay_per_s = float(decay_per_sec)
        self.min_level   = int(min_level)
        self.last_ts = time.time()

    def update_from_frame(self, frame_bgr):
        now = time.time()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(gray, 1.05, 3, minSize=(60,60))

        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
            roi = frame_bgr[y:y+h, x:x+w]
            mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) > 15
            if mask.any():
                mean_bgr = roi[mask].reshape(-1,3).mean(axis=0)
            else:
                mean_bgr = roi.reshape(-1,3).mean(axis=0)

            # EMA
            self.color_ema = (1.0 - self.ema_alpha) * self.color_ema + self.ema_alpha * mean_bgr
        else:
            # decaimiento a negro
            dt = now - self.last_ts
            k = clamp(self.decay_per_s * dt, 0.0, 1.0)
            self.color_ema = (1.0 - k) * self.color_ema

        self.last_ts = now

        # Convertimos a RGB y reducimos a un primario (dominante)
        b, g, r = self.color_ema
        rgb = np.array([r, g, b])
        if rgb.max() < self.min_level:
            return (0, 0, 0)
        dominant = int(np.argmax(rgb))
        out = [0, 0, 0]
        out[dominant] = int(clamp(rgb[dominant], 0, 255))
        return tuple(out)  # (R,G,B)

# ---------- util ----------
def open_capture(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir /dev/video{index}")
    return cap

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--models", type=str, required=True)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    cap = open_capture(args.camera, args.width, args.height, args.fps)
    colorizer = FacePrimaryColor(args.models, ema_alpha=0.22, decay_per_sec=0.28, min_level=28)

    dmx = DMXOut(universe=UNIVERSE, fps=DMX_FPS)
    dmx.start()

    cv2.namedWindow("Latencia Afectiva", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Latencia Afectiva", args.width, args.height)
    if args.fullscreen:
        cv2.setWindowProperty("Latencia Afectiva", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    interval = 1.0 / max(0.1, args.fps)
    next_shot = time.time()

    try:
        while True:
            now = time.time()
            if now >= next_shot:
                next_shot += interval
                ok, frame = cap.read()
                if ok:
                    rgb = colorizer.update_from_frame(frame)  # (R,G,B)

                    # DMX
                    dmx.set_rgb_color_over_range(DMX_LOW, DMX_HIGH, rgb)

                    # Pantalla: color sólido (sin mezclar con cámara)
                    bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                    disp = np.full((args.height, args.width, 3), bgr, dtype=np.uint8)
                    cv2.imshow("Latencia Afectiva", disp)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        dmx.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
