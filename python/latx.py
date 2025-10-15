#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latx.py — Latencia Afectiva (cam + emoción + DMX)
- Captura webcam (C922) y muestra a pantalla completa (800x480).
- Detecta rostro(s) con Haarcascade y estima emociones con FER+ (ONNX).
- Muestra top-3 emociones y emoción estable por histéresis.
- Controla DMX por OLA (Universe 0) en rango 22–56 con patrones pausados.

Ejecutar (X11 sobre Wayland):
QT_QPA_PLATFORM=xcb \
python3 latx.py \
  --fps 1 --camera 0 \
  --width 800 --height 480 --fullscreen \
  --models ~/Documents/GitHub/latencia-afectiva-instalacion/models
"""

import os
import sys
import time
import math
import argparse
from collections import deque
from array import array

import cv2
import numpy as np

from ola.ClientWrapper import ClientWrapper  # requiere ola-python del sistema

# =========================
# ======= CONFIG ==========
# =========================

UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56          # tu rango activo
DMX_FPS = 44                         # tick DMX (~44 Hz mantiene drivers felices)
SCREEN_W, SCREEN_H = 800, 480

FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# Colores base por emoción (R,G,B) — se aplican sobre el rango 22–56
EMO_COLOR = {
    "happiness": (255, 180, 20),   # cálido
    "sadness"  : (40, 90, 255),    # azul
    "anger"    : (255, 40, 10),    # rojo
    "fear"     : (150, 0, 200),    # púrpura
    "surprise" : (255, 255, 255),  # blanco
    "neutral"  : (120, 120, 120),  # gris
    "disgust"  : (20, 200, 60),    # verde
    "contempt" : (255, 120, 60),   # naranja tenue
}

# Velocidades “pausadas” (sensación de latencia)
EMO_SPEED = {
    "happiness": 2.0,
    "sadness"  : 4.0,
    "anger"    : 0.7,
    "fear"     : 1.5,
    "surprise" : 1.0,
    "neutral"  : 5.0,
    "disgust"  : 2.0,
    "contempt" : 3.0,
}

# =========================
# ======= Utils ===========
# =========================

def put_label(img, text, x, y):
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def clamp01(x): return max(0.0, min(1.0, x))

# =========================
# ==== DMX Controller =====
# =========================

class DMXOut:
    """
    Salida DMX por OLA, envía frames a ~44 Hz pase lo que pase.
    Usa array('B') ya que OLA espera un objeto con .tobytes().
    """
    def __init__(self, universe=UNIVERSE, fps=DMX_FPS):
        self.universe = universe
        self.period_ms = int(1000 / fps)
        self.wrapper = ClientWrapper()
        self.client = self.wrapper.Client()
        self.dmx = array('B', [0] * 512)
        self._running = False

    def set_range_constant(self, lo, hi, value):
        v = 0 if value < 0 else 255 if value > 255 else int(value)
        for ch in range(lo-1, hi):  # índices 0-based
            self.dmx[ch] = v

    def set_rgb_color_over_range(self, lo, hi, rgb):
        """Si tu driver interpreta 22..56 como múltiples canales lineales (no por tiras),
           simplemente aplicamos un patrón por bloques RGB “difuso” a todo el rango."""
        r, g, b = [max(0, min(255, int(c))) for c in rgb]
        # Distribuye R/G/B repetidamente en el rango (suave, no exacto por fixture)
        span = hi - lo + 1
        for i in range(span):
            ch = lo - 1 + i
            m = i % 3
            if m == 0:
                self.dmx[ch] = r
            elif m == 1:
                self.dmx[ch] = g
            else:
                self.dmx[ch] = b

    def blackout(self):
        for i in range(512):
            self.dmx[i] = 0
        # envía uno de cortesía (no bloqueamos aquí; el loop lo reenviará)

    def _tick(self):
        # Enviar frame DMX
        try:
            self.client.SendDmx(self.universe, self.dmx, lambda s: None)
        finally:
            # reprograma
            self.wrapper.AddEvent(self.period_ms, self._tick)

    def run_forever(self):
        self._running = True
        self.wrapper.AddEvent(self.period_ms, self._tick)
        try:
            self.wrapper.Run()
        except KeyboardInterrupt:
            pass
        finally:
            # apaga y manda un frame final
            self.blackout()
            try:
                self.client.SendDmx(self.universe, self.dmx, lambda s: None)
            except Exception:
                pass

    def stop(self):
        self._running = False
        self.blackout()
        try:
            self.client.SendDmx(self.universe, self.dmx, lambda s: None)
        except Exception:
            pass

# =========================
# === Emotion Estimator ===
# =========================

class EmotionEstimator:
    """
    Detección con Haar + FER+ ONNX.
    No introduce 'neutral' cuando no hay cara: mantiene la última emoción estable.
    Muestra top-3 y número de caras.
    """
    def __init__(self, models_dir):
        haar = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        onnx = os.path.join(models_dir, 'emotion-ferplus-8.onnx')

        self.det = cv2.CascadeClassifier(haar)
        if self.det.empty():
            raise RuntimeError('No se pudo cargar Haarcascade.')

        self.net = cv2.dnn.readNetFromONNX(onnx)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.hist = deque(maxlen=6)
        self.last_faces = []
        self.last_emo = 'neutral'
        self.last_top3 = [('neutral', 1.0), ('happiness', 0.0), ('sadness', 0.0)]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _face_probs(self, gray_roi):
        eq = self.clahe.apply(gray_roi)
        blob = cv2.dnn.blobFromImage(eq, scalefactor=1/255.0, size=(64, 64),
                                     mean=(0,), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        vec = out.reshape(-1)
        vec = np.exp(vec - vec.max())
        probs = vec / np.sum(vec)
        return probs

    def analyze(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
        )
        self.last_faces = faces

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            roi = gray[y:y+h, x:x+w]
            probs = self._face_probs(roi)
            idxs = np.argsort(-probs)
            top3 = [(FERPLUS_LABELS[i], float(probs[i])) for i in idxs[:3]]
            self.last_top3 = top3

            dom = top3[0][0]
            self.hist.append(dom)
            counts = {e: self.hist.count(e) for e in set(self.hist)}
            stable = max(counts.items(), key=lambda kv: kv[1])[0]
            self.last_emo = stable

        return self.last_emo

    def draw(self, frame):
        for (x, y, w, h) in self.last_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"caras: {len(self.last_faces)}", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
        y0 = 84
        for emo, p in self.last_top3:
            cv2.putText(frame, f"{emo}: {int(p*100)}%", (12, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 70), 2, cv2.LINE_AA)
            y0 += 24
        return frame

# =========================
# ====== Patterns =========
# =========================

class LatencyPatterns:
    """Genera un color (R,G,B) pausado por emoción y tiempo."""
    def __init__(self):
        self.current = 'neutral'
        self.last_switch = time.time()

    def set_emotion(self, emo):
        if emo != self.current:
            self.current = emo
            self.last_switch = time.time()

    def color_for_now(self, now_s):
        emo = self.current if self.current in EMO_COLOR else 'neutral'
        base = EMO_COLOR.get(emo, EMO_COLOR['neutral'])
        speed = EMO_SPEED.get(emo, 2.0)

        t = (now_s % speed) / max(1e-6, speed)

        # Patrones pausados:
        if emo == 'happiness':
            # pulso suave hacia el ámbar
            amp = 0.3 + 0.7 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
        elif emo == 'sadness':
            amp = 0.2 + 0.5 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
        elif emo == 'anger':
            # pequeños flashes
            amp = 1.0 if t < 0.15 else 0.15
        elif emo == 'fear':
            amp = 0.25 + 0.6 * (0.5 + 0.5 * math.sin(2 * math.pi * t))
        elif emo == 'surprise':
            amp = 1.0 if t < 0.2 else 0.4
        elif emo == 'disgust':
            amp = 0.35 + 0.5 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
        elif emo == 'contempt':
            amp = 0.3 + 0.4 * (0.5 + 0.5 * math.sin(2 * math.pi * t))
        else:  # neutral
            amp = 0.25 + 0.35 * (0.5 - 0.5 * math.cos(2 * math.pi * t))

        r = int(base[0] * clamp01(amp))
        g = int(base[1] * clamp01(amp))
        b = int(base[2] * clamp01(amp))
        return (r, g, b)

# =========================
# ========= MAIN ==========
# =========================

def open_capture(index, width, height, fps):
    # Forzamos V4L2 (más estable en tu Pi)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara /dev/video{}.".format(index))
    return cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=SCREEN_W)
    ap.add_argument("--height", type=int, default=SCREEN_H)
    ap.add_argument("--models", type=str, required=True, help="Carpeta con los modelos ONNX/XML")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    # --- Cámara
    cap = open_capture(args.camera, args.width, args.height, args.fps)

    # --- Emoción
    est = EmotionEstimator(args.models)

    # --- DMX
    dmx = DMXOut(universe=UNIVERSE, fps=DMX_FPS)
    patterns = LatencyPatterns()

    # --- Ventana
    cv2.namedWindow("Latencia Afectiva", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Latencia Afectiva", args.width, args.height)
    if args.fullscreen:
        cv2.setWindowProperty("Latencia Afectiva", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Loop principal (cámara a 0.5–1 fps, DMX en hilo interno a 44 Hz)
    next_shot = time.time()
    interval = 1.0 / max(0.1, args.fps)

    try:
        while True:
            # Espera “latente” para sostener el FPS bajo
            now = time.time()
            if now < next_shot:
                # Mientras esperamos, actualiza color DMX segun emoción actual
                rgb = patterns.color_for_now(now)
                dmx.set_rgb_color_over_range(DMX_LOW, DMX_HIGH, rgb)
                # pequeña siesta
                time.sleep(0.01)
                continue
            next_shot += interval

            ok, frame = cap.read()
            if not ok:
                continue

            # Analiza emoción (1 fps aprox.)
            emo = est.analyze(frame)
            patterns.set_emotion(emo)

            # DMX color actual (el hilo DMX ya está enviando frames)
            rgb = patterns.color_for_now(time.time())
            dmx.set_rgb_color_over_range(DMX_LOW, DMX_HIGH, rgb)

            # Mostrar
            disp = cv2.resize(frame, (args.width, args.height))
            est.draw(disp)
            put_label(disp, f'Emoción: {emo}', 12, 32)
            put_label(disp, f'Cam FPS: {args.fps:.2f}', 12, args.height - 18)
            cv2.imshow("Latencia Afectiva", disp)

            # Tecla ESC para salir
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        dmx.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

