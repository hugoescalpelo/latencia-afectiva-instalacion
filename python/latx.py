#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latencia Afectiva — Emociones → DMX (Raspberry Pi 4 + C922 + OLA/ENTTEC Open DMX)

Qué hace:
  • Abre la cámara Logitech C922 y muestra video a pantalla 800×480 (opcional fullscreen).
  • Detecta rostros (Haar) a ~1 FPS y estima emociones con FER+ (ONNX) vía OpenCV DNN.
  • Según la emoción dominante, genera secuencias de luz LENTAS (sensación de latencia)
    y las envía continuamente por OLA al Universe 0, rango de canales 22–56 (RGB repetido).

Requisitos (en un *solo* venv con paquetes del sistema para OLA):
  python3 -m venv --system-site-packages ~/.venvs/latencia
  source ~/.venvs/latencia/bin/activate
  pip install --upgrade pip wheel setuptools
  sudo apt install -y ola-python  # paquete del sistema, visible por el venv
  pip install opencv-contrib-python==4.9.0.80 numpy pillow

Modelos (colócalos en ~/Documents/GitHub/latencia-afectiva-instalacion/models):
  - haarcascade_frontalface_default.xml
  - emotion-ferplus-8.onnx  (descargado con huggingface-cli)

Ejecución (recomendado en Wayland):
  QT_QPA_PLATFORM=xcb \
  python3 latencia_emociones_dmx.py \
    --fps 1 --camera 0 --width 800 --height 480 --fullscreen \
    --models ~/Documents/GitHub/latencia-afectiva-instalacion/models

Salir: ESC o Ctrl+C
"""

import argparse
import os
import time
import math
from collections import deque

import cv2
import numpy as np

from ola.ClientWrapper import ClientWrapper  # del sistema (ola-python)

# =========================
# ======= CONFIG ==========
# =========================
DISPLAY_W, DISPLAY_H = 800, 480
UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56     # rango activo
DMX_FPS = 25                    # frecuencia de envío DMX (mantener vivo el frame)
CAM_FPS_DEFAULT = 1             # latencia buscada

# FER+ clases (orden estándar del modelo)
FERPLUS_LABELS = [
    'neutral','happiness','surprise','sadness','anger','disgust','fear','contempt'
]

# Color base por emoción (R,G,B)
EMO_COLOR = {
    'happiness': (255,180,30),
    'sadness':   (50,90,255),
    'anger':     (255,40,10),
    'fear':      (150,0,200),
    'surprise':  (255,255,255),
    'neutral':   (120,120,120),
    'disgust':   (20,200,60),
    'contempt':  (200,150,30),
}

# Velocidad (periodo seg) del patrón por emoción — LENTO para sensación de latencia
EMO_PERIOD = {
    'happiness': 2.0,
    'sadness':   4.0,
    'anger':     0.8,
    'fear':      1.6,
    'surprise':  1.0,
    'neutral':   6.0,
    'disgust':   2.5,
    'contempt':  3.0,
}

# =========================
# ====== DMX ENGINE =======
# =========================
class DMXEngine:
    """Pequeño motor DMX con OLA que transmite continuamente el buffer actual."""
    def __init__(self, universe=UNIVERSE, tx_hz=DMX_FPS):
        self.universe = universe
        self.wrapper = ClientWrapper()
        self.client = self.wrapper.Client()
        self.dmx = bytearray(512)
        self.tx_ms = max(5, int(1000/tx_hz))
        self._running = False

    def set_range_rgb_repeat(self, low, high, rgb):
        """Rellena canales [low..high] con RGB repetido en grupos de 3 (R,G,B)."""
        r,g,b = [max(0, min(255, int(v))) for v in rgb]
        low_i = max(1, low) - 1
        high_i = min(512, high) - 1
        for ch in range(low_i, high_i+1, 3):
            if ch <= high_i:   self.dmx[ch] = r
            if ch+1 <= high_i: self.dmx[ch+1] = g
            if ch+2 <= high_i: self.dmx[ch+2] = b

    def blackout(self):
        for i in range(512):
            self.dmx[i] = 0

    def _tick(self):
        # Enviar y reprogramar
        self.client.SendDmx(self.universe, bytes(self.dmx), lambda s: None)
        if self._running:
            self.wrapper.AddEvent(self.tx_ms, self._tick)

    def start(self):
        self._running = True
        self.wrapper.AddEvent(self.tx_ms, self._tick)

    def run_forever(self):
        try:
            self.wrapper.Run()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.blackout()
            self.client.SendDmx(self.universe, bytes(self.dmx), lambda s: None)

# =========================
# ====== PATTERNER ========
# =========================
class Pattern:
    """Genera colores lentos por emoción (breathing / fades) para toda la tira."""
    def __init__(self):
        self.emotion = 'neutral'
        self.last_change = time.time()
        self.brightness = 200  # techo general

    def set_emotion(self, emo: str):
        if emo in EMO_COLOR and emo != self.emotion:
            self.emotion = emo
            self.last_change = time.time()

    def color_now(self, t=None):
        if t is None:
            t = time.time()
        base = EMO_COLOR.get(self.emotion, (120,120,120))
        period = EMO_PERIOD.get(self.emotion, 3.0)
        ph = (t - self.last_change) % period / max(1e-6, period)

        # breathing lento con leve retardo (sensación de latencia)
        amp = 0.40 + 0.60 * (0.5 - 0.5*math.cos(2*math.pi*ph))
        r = int(base[0] * amp)
        g = int(base[1] * amp)
        b = int(base[2] * amp)

        # tope de brillo
        scale = self.brightness/255.0
        return (int(r*scale), int(g*scale), int(b*scale))

# =========================
# ====== VISION (FER+) ====
# =========================
class EmotionEstimator:
    def __init__(self, models_dir):
        haar = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        onnx = os.path.join(models_dir, 'emotion-ferplus-8.onnx')
        self.det = cv2.CascadeClassifier(haar)
        if self.det.empty():
            raise RuntimeError('No se pudo cargar Haarcascade. Revisa la ruta.')
        self.net = cv2.dnn.readNetFromONNX(onnx)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # historial para estabilizar (modo)
        self.hist = deque(maxlen=6)
        self.last_faces = []  # para dibujar

    def analyze(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        self.last_faces = faces

        dom_emo = 'neutral'
        # Tomamos la emoción dominante del rostro más grande
        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
            roi = gray[y:y+h, x:x+w]
            # FER+: 64x64, normalizado 0..1
            blob = cv2.dnn.blobFromImage(roi, scalefactor=1/255.0, size=(64,64), mean=(0,0,0), swapRB=False, crop=True)
            self.net.setInput(blob)
            out = self.net.forward()  # (1,8)
            scores = out.flatten()
            # softmax
            e = np.exp(scores - scores.max())
            probs = e / np.sum(e)
            idx = int(np.argmax(probs))
            dom_emo = FERPLUS_LABELS[idx]

        # estabilizar
        self.hist.append(dom_emo)
        counts = {}
        for e in self.hist:
            counts[e] = counts.get(e, 0) + 1
        stable = max(counts.items(), key=lambda kv: kv[1])[0]
        return stable

    def draw(self, frame):
        # Top-3 por rostro (opcional simple: solo dibujamos box; top3 global sería otra pasada)
        for (x,y,w,h) in self.last_faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
        return frame

# =========================
# ========= MAIN ==========
# =========================

def put_label(img, text, x=12, y=28):
    cv2.putText(img, text, (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description='Latencia Afectiva — Emociones → DMX')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--fps', type=float, default=CAM_FPS_DEFAULT)
    ap.add_argument('--width', type=int, default=DISPLAY_W)
    ap.add_argument('--height', type=int, default=DISPLAY_H)
    ap.add_argument('--models', type=str, required=True, help='Carpeta con onnx/xml')
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--minsize', type=int, default=70)
    ap.add_argument('--no-gui', action='store_true')
    args = ap.parse_args()

    # Vision
    est = EmotionEstimator(args.models)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la cámara.')
    # captura más ligera y escalamos a 800x480
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not args.no_gui:
        win = 'Latencia — Emociones → DMX'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.width, args.height)
        if args.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # DMX
    dmx = DMXEngine(universe=UNIVERSE, tx_hz=DMX_FPS)
    pat = Pattern()

    # Programar bucles en el wrapper de OLA
    dmx.start()

    cam_interval_ms = max(200, int(1000.0/max(0.1, args.fps)))  # 1 FPS => 1000ms, 0.5 FPS => 2000ms

    last_emo = 'neutral'

    def cam_step():
        nonlocal last_emo
        ok, frame = cap.read()
        if ok:
            emo = est.analyze(frame)
            pat.set_emotion(emo)
            if emo != last_emo:
                last_emo = emo
            # Dibujar
            if not args.no_gui:
                disp = cv2.resize(frame, (args.width, args.height))
                est.draw(disp)
                put_label(disp, f'Emoción: {emo}', 12, 32)
                put_label(disp, f'FPS cam: {args.fps:.2f}', 12, args.height-18)
                cv2.imshow('Latencia — Emociones → DMX', disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    # ESC => parar wrapper
                    dmx._running = False
                    dmx.wrapper.Stop()
                    return
        dmx.wrapper.AddEvent(cam_interval_ms, cam_step)

    def lights_step():
        # Actualizar color actual según patrón lento y volcar a rango 22–56
        rgb = pat.color_now(time.time())
        dmx.set_range_rgb_repeat(DMX_LOW, DMX_HIGH, rgb)
        # Reprograma (sincronizado a la tasa DMX ya que _tick envía)
        dmx.wrapper.AddEvent(200, lights_step)  # 5 Hz para evolución del patrón

    # Lanzar tareas
    dmx.wrapper.AddEvent(0, cam_step)
    dmx.wrapper.AddEvent(0, lights_step)

    # Loop principal (DMX + eventos)
    try:
        dmx.run_forever()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
