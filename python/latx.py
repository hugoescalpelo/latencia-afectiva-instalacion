#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latx.py — Latencia Afectiva (cam + emoción + DMX)
Corregido:
- DMXOut.start() arranca envío continuo (44 Hz) y set_* fuerza frame inmediato.
- Anti-neutral: si hay cara y 'neutral' gana por poco, elegimos la 2ª emoción.
- Sin caras: mantenemos emoción previa (no caemos a neutral).
"""

import os, time, math, argparse, threading
from collections import deque
from array import array
import cv2, numpy as np
from ola.ClientWrapper import ClientWrapper

UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56
DMX_FPS = 44

FERPLUS_LABELS = [
    "neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"
]

EMO_COLOR = {
    "happiness": (255,180, 20),
    "sadness"  : ( 40, 90,255),
    "anger"    : (255, 40, 10),
    "fear"     : (150,  0,200),
    "surprise" : (255,255,255),
    "neutral"  : (120,120,120),
    "disgust"  : ( 20,200, 60),
    "contempt" : (255,120, 60),
}
EMO_SPEED = {
    "happiness": 2.0, "sadness": 4.0, "anger": 0.7, "fear": 1.5,
    "surprise": 1.0, "neutral": 5.0, "disgust": 2.0, "contempt": 3.0,
}

def clamp01(x): return max(0.0, min(1.0, x))
def put_label(img, text, x, y):
    cv2.putText(img, text, (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

# ---------------- DMX ----------------

class DMXOut:
    """Envío continuo a 44 Hz + push inmediato en cada set_*."""
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
        # OLA acepta array('B'); no usar bytes (evita .tobytes en OlaClient antiguo).
        with self._lock:
            payload = self.dmx
        try:
            self.client.SendDmx(self.universe, payload, lambda s: None)
        except Exception:
            pass

    def set_range_constant(self, lo, hi, value):
        v = 0 if value < 0 else 255 if value > 255 else int(value)
        with self._lock:
            for ch in range(lo-1, hi):
                self.dmx[ch] = v
        self._send_now()

    def set_rgb_color_over_range(self, lo, hi, rgb):
        r,g,b = [max(0,min(255,int(c))) for c in rgb]
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
        # Bucle de envío sostenido para mantener vivo el decodificador
        next_t = time.time()
        while not self._stop.is_set():
            self._send_now()
            next_t += self.period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(min(sleep, self.period))
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

# ------------- Emoción ---------------

class EmotionEstimator:
    """Haar + FER+ ONNX con anti-neutral y memoria corta."""
    def __init__(self, models_dir):
        haar = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        onnx = os.path.join(models_dir, 'emotion-ferplus-8.onnx')
        self.det = cv2.CascadeClassifier(haar)
        if self.det.empty(): raise RuntimeError('No se pudo cargar Haarcascade.')
        self.net = cv2.dnn.readNetFromONNX(onnx)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.hist = deque(maxlen=6)
        self.last_faces = []
        self.last_emo = 'neutral'
        self.last_top3 = [('neutral',1.0), ('happiness',0.0), ('sadness',0.0)]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _face_probs(self, gray_roi):
        eq = self.clahe.apply(gray_roi)
        blob = cv2.dnn.blobFromImage(eq, scalefactor=1/255.0, size=(64,64),
                                     mean=(0,), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward().reshape(-1)
        out = np.exp(out - out.max())
        probs = out / np.sum(out)
        return probs

    def analyze(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(gray, scaleFactor=1.045, minNeighbors=3, minSize=(60,60))
        self.last_faces = faces

        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
            probs = self._face_probs(gray[y:y+h, x:x+w])
            idxs = np.argsort(-probs)
            top = [(FERPLUS_LABELS[i], float(probs[i])) for i in idxs[:3]]
            self.last_top3 = top

            dom, p1 = top[0]
            # Anti-neutral: si neutral gana por poco, toma segunda mejor
            if dom == 'neutral' and (top[1][1] >= 0.28 or (p1 - top[1][1]) <= 0.07):
                dom = top[1][0]

            self.hist.append(dom)
            counts = {e:self.hist.count(e) for e in set(self.hist)}
            self.last_emo = max(counts.items(), key=lambda kv: kv[1])[0]

        # si no hay caras, NO cambiamos emoción (se mantiene)
        return self.last_emo

    def draw(self, frame):
        for (x,y,w,h) in self.last_faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
        cv2.putText(frame, f"caras: {len(self.last_faces)}", (12,56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,255,40), 2, cv2.LINE_AA)
        y0 = 84
        for emo,p in self.last_top3:
            cv2.putText(frame, f"{emo}: {int(p*100)}%", (12,y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,220,70), 2, cv2.LINE_AA)
            y0 += 24
        return frame

# ------------- Patrones ---------------

class LatencyPatterns:
    def __init__(self):
        self.current = 'neutral'
    def set_emotion(self, emo):
        if emo in EMO_COLOR:
            self.current = emo
    def color_for_now(self, now_s):
        emo = self.current
        base = EMO_COLOR.get(emo, EMO_COLOR['neutral'])
        speed = EMO_SPEED.get(emo, 2.0)
        t = (now_s % speed) / max(1e-6, speed)

        if emo == 'happiness':
            amp = 0.3 + 0.7*(0.5 - 0.5*math.cos(2*math.pi*t))
        elif emo == 'sadness':
            amp = 0.2 + 0.5*(0.5 - 0.5*math.cos(2*math.pi*t))
        elif emo == 'anger':
            amp = 1.0 if t < 0.15 else 0.15
        elif emo == 'fear':
            amp = 0.25 + 0.6*(0.5 + 0.5*math.sin(2*math.pi*t))
        elif emo == 'surprise':
            amp = 1.0 if t < 0.2 else 0.4
        elif emo == 'disgust':
            amp = 0.35 + 0.5*(0.5 - 0.5*math.cos(2*math.pi*t))
        elif emo == 'contempt':
            amp = 0.3 + 0.4*(0.5 + 0.5*math.sin(2*math.pi*t))
        else:  # neutral
            amp = 0.25 + 0.35*(0.5 - 0.5*math.cos(2*math.pi*t))

        r = int(base[0] * clamp01(amp))
        g = int(base[1] * clamp01(amp))
        b = int(base[2] * clamp01(amp))
        return (r,g,b)

# ------------- Main -------------------

def open_capture(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir /dev/video{index}")
    return cap

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
    est = EmotionEstimator(args.models)

    dmx = DMXOut(universe=UNIVERSE, fps=DMX_FPS)
    dmx.start()  # *** IMPORTANTE *** arranca envío continuo
    patterns = LatencyPatterns()

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
                    emo = est.analyze(frame)
                    patterns.set_emotion(emo)
                    disp = cv2.resize(frame, (args.width, args.height))
                    est.draw(disp)
                    put_label(disp, f'Emoción: {emo}', 12, 32)
                    put_label(disp, f'Cam FPS: {args.fps:.2f}', 12, args.height-18)
                    cv2.imshow("Latencia Afectiva", disp)

            # Actualiza DMX constantemente con el patrón actual
            rgb = patterns.color_for_now(time.time())
            dmx.set_rgb_color_over_range(DMX_LOW, DMX_HIGH, rgb)

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
