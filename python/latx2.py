#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latx.py — Cámara a fullscreen + color DMX basado en el rostro (sin emociones).
- Toma el ROI del rostro más grande y calcula el color promedio (BGR).
- Suaviza el color (EMA) para dar sensación de latencia.
- Rellena el área del rostro en pantalla con ese color.
- Envía RGB continuo a Universe 0, canales 22–56 (repetición RGB).
Requisitos:
  - OpenCV (python3-opencv o wheel 4.10.x)
  - OLA (ola, ola-python) y ENTTEC parcheado a Universe 0
Lanza con:
  QT_QPA_PLATFORM=xcb python3 latx.py --fps 1 --camera 0 --width 800 --height 480 --fullscreen --models ~/.../models
"""

import os, time, argparse, threading
from array import array
import cv2
import numpy as np
from ola.ClientWrapper import ClientWrapper

# --- Config DMX ---
UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56
DMX_FPS = 44

def clamp(val, lo, hi): return max(lo, min(hi, val))

# ---------------- DMX ----------------

class DMXOut:
    """Envío continuo a 44 Hz + push inmediato en cada actualización."""
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
            # Enviar array('B') directamente evita el .tobytes del cliente OLA antiguo
            self.client.SendDmx(self.universe, payload, lambda s: None)
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
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(min(dt, self.period))
            else:
                next_t = time.time()

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

# ------------- Detección rostro + color -------------

class FaceColor:
    """Detecta rostro, calcula color promedio, suaviza (EMA) y mantiene último valor."""
    def __init__(self, models_dir, ema_alpha=0.25, decay_per_sec=0.15):
        haar = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        self.det = cv2.CascadeClassifier(haar)
        if self.det.empty():
            raise RuntimeError('No se pudo cargar haarcascade_frontalface_default.xml')
        self.color_ema = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # BGR
        self.ema_alpha = float(ema_alpha)
        self.decay_per_sec = float(decay_per_sec)
        self.last_ts = time.time()

    def update_from_frame(self, frame_bgr):
        now = time.time()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60,60))

        if len(faces) > 0:
            # rostro más grande
            x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
            roi = frame_bgr[y:y+h, x:x+w]
            # promedio BGR, ignorando extremos muy oscuros (umbral ligero)
            mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) > 15
            if mask.any():
                mean_bgr = roi[mask].reshape(-1,3).mean(axis=0)
            else:
                mean_bgr = roi.reshape(-1,3).mean(axis=0)

            # suavizado exponencial
            self.color_ema = (1.0 - self.ema_alpha) * self.color_ema + self.ema_alpha * mean_bgr

            # dibuja el rectángulo de color sobre la cara (overlay)
            overlay = frame_bgr.copy()
            b,g,r = [int(clamp(c, 0, 255)) for c in self.color_ema]
            cv2.rectangle(overlay, (x,y), (x+w, y+h), (b,g,r), thickness=-1)
            # mezcla (50% del color sobre rostro)
            cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

        else:
            # sin rostros: decaimos suave a negro (sensación de latencia)
            dt = now - self.last_ts
            k = clamp(self.decay_per_sec * dt, 0.0, 1.0)
            self.color_ema = (1.0 - k) * self.color_ema  # tiende a 0

        self.last_ts = now
        # devuelve color actual (convertido a RGB para DMX if needed)
        b,g,r = self.color_ema
        return (int(r), int(g), int(b))

# ------------- Main -------------------

def open_capture(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir /dev/video{index}")
    return cap

def put_label(img, text, x, y):
    cv2.putText(img, text, (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

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

    face_col = FaceColor(args.models, ema_alpha=0.25, decay_per_sec=0.20)

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
                    # procesar color (también pinta overlay del color en la cara)
                    rgb = face_col.update_from_frame(frame)
                    # enviar a DMX
                    dmx.set_rgb_color_over_range(DMX_LOW, DMX_HIGH, rgb)

                    # mostrar
                    disp = cv2.resize(frame, (args.width, args.height))
                    put_label(disp, f"RGB DMX: {rgb}", 12, 32)
                    put_label(disp, f"Cam FPS: {args.fps:.2f}", 12, args.height-18)
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
