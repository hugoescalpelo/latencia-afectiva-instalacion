#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latencia Afectiva — Raspberry Pi 4 + DSI 800x480 + Webcam + DeepFace (emociones) + DMX (Enttec Open)

• Muestra la webcam a pantalla completa en la DSI.
• Analiza emociones con DeepFace cada N frames.
• Según la emoción, ejecuta un patrón luminoso en 8 tiras RGB (2 drivers x 4 tiras)
  controladas por dos drivers RGB (12 canales c/u) via Enttec Open DMX (USB) usando OLA.

Requisitos sistema (Pi OS 64-bit recomendado):
------------------------------------------------
# Dependencias del sistema (video, compilar, OLA, etc.)
sudo apt update && sudo apt install -y \
  python3-venv python3-dev build-essential git pkg-config \
  libatlas-base-dev libopenjp2-7 libtiff6 libilmbase25 libopenexr25 libgstreamer1.0-0 \
  libavcodec58 libavformat58 libswscale5 libgtk-3-0 \
  ola ola-python ola-rdm-tests

# (Opcional) habilita cámara V4L2 si usas CSI; para USB no es necesario
# sudo raspi-config -> Interface Options -> enable camera

# Configura OLA para Enttec Open:
# 1) Conecta el Enttec Open (basado en FT232). 
# 2) sudo ola_dev_info  (verifica que aparezca el dispositivo)
# 3) sudo olad  (servicio daemon) o habilítalo: sudo systemctl enable --now olad
# 4) Ve a http://<pi>:9090 -> Página web de OLA, configura el dispositivo como salida del Universo 0.

Entorno Python y dependencias:
------------------------------
python3 -m venv ~/.venvs/latencia && \
source ~/.venvs/latencia/bin/activate && \
pip install --upgrade pip wheel setuptools && \
# OpenCV + DeepFace (usa TF/Torch internamente; en Pi puede tardar la primera vez)
pip install opencv-python==4.9.0.80 deepface==0.0.93 \
            numpy==1.26.4 \
            pillow \
            ola @ git+https://github.com/OpenLightingProject/ola.git#subdirectory=python/ola

Ejecución:
----------
source ~/.venvs/latencia/bin/activate
python3 latencia_afectiva.py --camera 0 --fullscreen --fps 30

Notas:
------
• Si el rendimiento de DeepFace es limitado, reducir "ANALYZE_EVERY_N_FRAMES" o usar tamaño de frame menor.
• Asegúrate de mapear correctamente las direcciones DMX de tus drivers. Ver CONFIG.
• Salida con tecla ESC o Ctrl+C.
"""

import argparse
import cv2
import numpy as np
import threading
import time
import math
from collections import deque

# DeepFace puede tardar en importar; mejor lazy import en hilo
from deepface import DeepFace  # type: ignore

# OLA (Open Lighting Architecture) — cliente Python
from ola.ClientWrapper import ClientWrapper  # type: ignore

# =========================
# ======= CONFIG ==========
# =========================

# Resolución esperada de la DSI
DISPLAY_W, DISPLAY_H = 800, 480

# Frecuencia de captura y análisis
TARGET_FPS = 30
ANALYZE_EVERY_N_FRAMES = 10   # analiza 1 de cada 10 frames

# Universo OLA a utilizar
OLA_UNIVERSE = 0

# Mapping de drivers RGB -> 4 tiras cada uno (12 canales c/u)
# Ajusta bases según el direccionamiento DMX que hayas puesto en tus drivers.
# Cada tira RGB ocupa 3 canales: R,G,B.
# Driver A (tiras 1-4): canales 1..12  (1:R1,2:G1,3:B1, 4:R2,5:G2,6:B2, ...)
# Driver B (tiras 5-8): canales 13..24 por defecto aquí.
DMX_BASE_DRIVER_A = 1
DMX_BASE_DRIVER_B = 13

# Brillos máximos por emoción (0-255)
BRIGHTNESS_DEFAULT = 180

# Emociones que devuelve DeepFace con su clave típica
EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Paleta de colores de base por emoción (R,G,B) 0-255
EMOTION_COLOR = {
    "happy":    (255, 180, 20),    # ámbar cálido
    "sad":      (40, 90, 255),     # azul
    "angry":    (255, 40, 10),     # rojo
    "fear":     (150, 0, 200),     # púrpura
    "surprise": (255, 255, 255),   # blanco
    "neutral":  (120, 120, 120),   # gris
    "disgust":  (20, 200, 60),     # verde
}

# Velocidades/periodos de patrón por emoción (segundos por ciclo)
EMOTION_SPEED = {
    "happy": 1.2,
    "sad": 3.0,
    "angry": 0.25,
    "fear": 0.8,
    "surprise": 0.4,
    "neutral": 5.0,
    "disgust": 1.2,
}

# =========================
# ==== DMX Controller =====
# =========================

class DMXController:
    def __init__(self, universe=0):
        self.universe = universe
        self.wrapper = ClientWrapper()
        self.client = self.wrapper.Client()
        self.dmx = bytearray(512)  # 512 canales
        self.lock = threading.Lock()
        # hilo de envío periódico
        self._running = True
        self._thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._thread.start()

    def _tx_once(self):
        with self.lock:
            data = bytes(self.dmx)
        self.client.SendDmx(self.universe, data, lambda state: None)

    def _tx_loop(self):
        # envía a ~30 Hz para mantener la salida viva
        while self._running:
            self._tx_once()
            # Procesa eventos de OLA
            self.wrapper.RunOnce()
            time.sleep(1.0 / 30.0)

    def set_rgb_strip(self, base_addr: int, strip_index: int, rgb_tuple):
        """
        base_addr: canal base del driver (1-indexed)
        strip_index: 0..3 (cuatro tiras por driver)
        rgb_tuple: (R,G,B) 0-255
        """
        r, g, b = [max(0, min(255, int(x))) for x in rgb_tuple]
        # cálculo de canales (convertir a 0-indexed para el array)
        ch_r = base_addr - 1 + strip_index * 3
        ch_g = ch_r + 1
        ch_b = ch_r + 2
        with self.lock:
            self.dmx[ch_r] = r
            self.dmx[ch_g] = g
            self.dmx[ch_b] = b

    def blackout_all(self):
        with self.lock:
            for i in range(512):
                self.dmx[i] = 0

    def stop(self):
        self._running = False
        # manda un último blackout
        self.blackout_all()
        self._tx_once()
        # da tiempo a cerrar
        time.sleep(0.1)

# =========================
# ==== Pattern Engine =====
# =========================

class PatternEngine:
    """Genera valores RGB para 8 tiras según emoción y tiempo."""
    def __init__(self):
        self.current_emotion = "neutral"
        self.last_switch_ts = time.time()
        self.brightness = BRIGHTNESS_DEFAULT
        # buffer estado
        self.num_strips = 8

    def set_emotion(self, emo: str):
        emo = emo if emo in EMOTION_KEYS else "neutral"
        if emo != self.current_emotion:
            self.current_emotion = emo
            self.last_switch_ts = time.time()

    def _ease(self, x):
        # suavizado cosenoidal (0..1)
        return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, x)))

    def _apply_brightness(self, rgb):
        scale = self.brightness / 255.0
        return tuple(int(c * scale) for c in rgb)

    def _color_lerp(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def colors_for_time(self, now_s: float):
        emo = self.current_emotion
        base = EMOTION_COLOR.get(emo, EMOTION_COLOR["neutral"])
        speed = EMOTION_SPEED.get(emo, 2.0)
        t = (now_s % speed) / max(1e-6, speed)

        cols = [(0, 0, 0)] * self.num_strips

        if emo == "happy":
            # Chase cálido de 4 fases
            for i in range(self.num_strips):
                phase = (t + i / self.num_strips) % 1.0
                amp = self._ease(phase)
                cols[i] = self._apply_brightness(
                    self._color_lerp((10, 10, 10), base, amp)
                )
        elif emo == "sad":
            # Breathing azul
            amp = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(2 * math.pi * t))
            c = self._apply_brightness((int(base[0] * 0.3), int(base[1] * amp), base[2]))
            cols = [c] * self.num_strips
        elif emo == "angry":
            # Estrobo rojo alternando pares/impares
            on = 1 if t < 0.5 else 0
            for i in range(self.num_strips):
                active = on if (i % 2 == 0) else 1 - on
                cols[i] = self._apply_brightness(base if active else (0, 0, 0))
        elif emo == "fear":
            # Pulso hacia los extremos
            center = 3.5
            for i in range(self.num_strips):
                dist = abs(i - center)
                amp = max(0.0, 1.0 - dist / 4.0)
                amp *= (0.5 + 0.5 * math.sin(2 * math.pi * t))
                cols[i] = self._apply_brightness(
                    self._color_lerp((0, 0, 0), base, amp)
                )
        elif emo == "surprise":
            # Flash total con decaimiento rápido
            amp = 1.0 if t < 0.15 else max(0.0, 1.0 - (t - 0.15) / 0.35)
            c = self._apply_brightness(self._color_lerp((40, 40, 40), base, amp))
            cols = [c] * self.num_strips
        elif emo == "disgust":
            # Ola verde desplazada
            for i in range(self.num_strips):
                phase = (t + i / 4.0) % 1.0
                amp = 0.3 + 0.7 * self._ease(phase)
                cols[i] = self._apply_brightness(
                    (int(base[0] * 0.3), int(base[1] * amp), int(base[2] * 0.3))
                )
        else:  # neutral
            # Fade muy lento gris -> base -> gris
            cycle = 0.5 - 0.5 * math.cos(2 * math.pi * t)
            c = self._apply_brightness(self._color_lerp((30, 30, 30), base, cycle))
            cols = [c] * self.num_strips

        return cols

# =========================
# === Emotion Analyzer ====
# =========================

class EmotionAnalyzer:
    def __init__(self):
        # usa un deque para suavizar la emoción detectada
        self.history = deque(maxlen=8)
        self.current = "neutral"
        self._lock = threading.Lock()

    def analyze(self, frame_bgr):
        # DeepFace espera RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
            # DeepFace >=0.0.93 retorna dict o lista; normalizamos
            if isinstance(result, list):
                result = result[0]
            emotions = result.get('emotion') or result.get('emotions')
            if emotions:
                # toma la emoción con mayor score
                emo = max(emotions.items(), key=lambda kv: kv[1])[0]
            else:
                emo = "neutral"
        except Exception:
            emo = "neutral"

        with self._lock:
            self.history.append(emo if emo in EMOTION_KEYS else "neutral")
            # emoción dominante en la historia (modo)
            counts = {k: 0 for k in EMOTION_KEYS}
            for e in self.history:
                counts[e] += 1
            self.current = max(counts.items(), key=lambda kv: kv[1])[0]

    def get_current(self):
        with self._lock:
            return self.current

# =========================
# ========= MAIN ==========
# =========================

def draw_label(img, text, x=12, y=32):
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)


def map_strips_to_dmx(dmx: DMXController, colors):
    """Aplica lista de 8 colores [(R,G,B),...] a los dos drivers."""
    # strips 0..3 -> Driver A
    for i in range(4):
        dmx.set_rgb_strip(DMX_BASE_DRIVER_A, i, colors[i])
    # strips 4..7 -> Driver B
    for i in range(4, 8):
        dmx.set_rgb_strip(DMX_BASE_DRIVER_B, i-4, colors[i])


def main():
    parser = argparse.ArgumentParser(description="Latencia Afectiva — Webcam + DeepFace + DMX")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara V4L2/USB (por defecto 0)")
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--width", type=int, default=DISPLAY_W)
    parser.add_argument("--height", type=int, default=DISPLAY_H)
    parser.add_argument("--flip", action="store_true", help="Voltea horizontalmente la vista")
    parser.add_argument("--no-gui", action="store_true", help="No mostrar ventana (solo DMX)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifica /dev/video* y permisos.")

    # Ajusta resolución si la cámara lo soporta
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not args.no_gui:
        cv2.namedWindow("Latencia Afectiva", cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty("Latencia Afectiva", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow("Latencia Afectiva", args.width, args.height)

    dmx = DMXController(universe=OLA_UNIVERSE)
    patterns = PatternEngine()
    analyzer = EmotionAnalyzer()

    frame_id = 0
    last_emo = "neutral"
    last_analysis_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.flip:
                frame = cv2.flip(frame, 1)

            # reduce tamaño para análisis (rendimiento)
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            # analiza cada N frames (~ fps/ANALYZE_EVERY_N_FRAMES)
            if frame_id % ANALYZE_EVERY_N_FRAMES == 0:
                analyzer.analyze(small)
                last_analysis_ts = time.time()

            current_emo = analyzer.get_current()
            patterns.set_emotion(current_emo)

            # genera colores actuales
            now = time.time()
            colors = patterns.colors_for_time(now)
            map_strips_to_dmx(dmx, colors)

            # overlay
            if not args.no_gui:
                disp = cv2.resize(frame, (args.width, args.height))
                draw_label(disp, f"Emoción: {current_emo}")
                draw_label(disp, f"FPS objetivo: {args.fps}", 12, args.height-16)
                cv2.imshow("Latencia Afectiva", disp)

            key = cv2.waitKey(1) & 0xFF if not args.no_gui else 255
            if key == 27:  # ESC
                break

            frame_id += 1
    except KeyboardInterrupt:
        pass
    finally:
        dmx.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
