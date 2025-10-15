#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latencia Afectiva — C922 a 1 FPS, fullscreen 800x480
Detección facial con YuNet (ONNX) + wireframe 68 puntos (ONNX)
Fallback automático a Haar (solo bbox) si faltan modelos ONNX.

Teclas:
  F -> fullscreen on/off
  G -> overlays on/off
  ESC -> salir
"""
import cv2, time, argparse, os, sys
import numpy as np

DISPLAY_W, DISPLAY_H = 800, 480

def draw_label(img, text, x=12, y=28, scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def load_detectors(models_dir):
    paths = {
        "haar": os.path.join(models_dir, "haarcascade_frontalface_default.xml"),
        "yunet": os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
        "lmk68": os.path.join(models_dir, "face_landmark_68.onnx"),
    }
    # Haar (fallback)
    haar = None
    if os.path.exists(paths["haar"]):
        haar = cv2.CascadeClassifier(paths["haar"])
        if haar.empty():
            haar = None

    # YuNet detector
    yunet_net = None
    if os.path.exists(paths["yunet"]):
        try:
            yunet_net = cv2.dnn.readNet(paths["yunet"])
        except Exception as e:
            print("⚠️ No se pudo cargar YuNet ONNX:", e)
            yunet_net = None

    # Landmarks 68
    lmk_net = None
    if os.path.exists(paths["lmk68"]):
        try:
            lmk_net = cv2.dnn.readNet(paths["lmk68"])
        except Exception as e:
            print("⚠️ No se pudo cargar face_landmark_68 ONNX:", e)
            lmk_net = None

    return haar, yunet_net, lmk_net, paths

def yunet_detect(net, frame_bgr, conf_th=0.8):
    """
    Infiere con YuNet (entrada 320x320). Devuelve lista de rects (x,y,w,h).
    Simplificado para 1 FPS: no ajustamos escala por cara, funciona bien en prácticas.
    """
    h, w = frame_bgr.shape[:2]
    inp = cv2.resize(frame_bgr, (320, 320))
    blob = cv2.dnn.blobFromImage(inp, 1.0/255.0, (320, 320), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()  # [1, N, 15], ver doc YuNet
    rects = []
    for det in out[0]:
        score = det[14]
        if score < conf_th:
            continue
        # caja en coords 320x320
        x, y, w_box, h_box = det[0], det[1], det[2], det[3]
        # volver a tamaño original
        sx, sy = w / 320.0, h / 320.0
        x, y, w_box, h_box = int(x*sx), int(y*sy), int(w_box*sx), int(h_box*sy)
        if w_box > 20 and h_box > 20:
            rects.append((x, y, w_box, h_box))
    return rects

def lmk68_infer(net, frame_bgr, rect):
    """
    Inferir landmarks 68 usando ONNX del OpenCV Zoo.
    Este modelo espera cara recortada + preprocesado estándar.
    Para simplificar, extraemos ROI, redimensionamos a 160x160, normalizamos [0,1].
    """
    x, y, w, h = rect
    h_img, w_img = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x+w), min(h_img, y+h)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    inp = cv2.resize(roi, (160, 160))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp.astype(np.float32) / 255.0
    # NCHW
    blob = np.transpose(inp, (2,0,1))[np.newaxis, ...]
    net.setInput(blob)
    out = net.forward()  # forma [1,136] -> 68 pares (x,y) normalizados 0..1
    pts = out.reshape(-1, 2)
    # mapear de 160x160 al rect original
    pts[:,0] = x0 + pts[:,0] * (x1 - x0)
    pts[:,1] = y0 + pts[:,1] * (y1 - y0)
    return pts.astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--width", type=int, default=DISPLAY_W)
    ap.add_argument("--height", type=int, default=DISPLAY_H)
    ap.add_argument("--models", type=str, default=os.path.expanduser("~/Documents/GitHub/latencia-afectiva-instalacion/models"))
    args = ap.parse_args()

    haar, yunet_net, lmk_net, paths = load_detectors(args.models)

    if yunet_net is None and haar is None:
        print("❌ No hay detector disponible. Falta YuNet ONNX y también Haar.", file=sys.stderr)
        print("   Coloca los modelos en:", args.models, file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara", file=sys.stderr); sys.exit(1)
    # Logitech C922, una resolución estable:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, max(1.0, args.fps))

    win = "Latencia Afectiva — Cámara"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, args.width, args.height)

    overlay_on = True
    fullscreen = args.fullscreen
    period = 1.0 / max(0.1, args.fps)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            # Detecta caras
            if yunet_net is not None:
                rects = yunet_detect(yunet_net, frame, conf_th=0.8)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects_np = haar.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (80,80))
                rects = [tuple(map(int, r)) for r in rects_np]

            vis = frame.copy()
            # Dibuja bbox y landmarks si están
            for (x,y,w,h) in rects:
                cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,255), 2)
                if lmk_net is not None:
                    try:
                        pts = lmk68_infer(lmk_net, frame, (x,y,w,h))
                        if pts is not None:
                            for (px,py) in pts:
                                cv2.circle(vis, (int(px), int(py)), 1, (0,200,255), -1)
                    except Exception as e:
                        # Si algo falla en landmarks, sigue con bbox sin cortar el loop
                        pass

            disp = cv2.resize(vis, (args.width, args.height))

            if overlay_on:
                mode = "YuNet+68" if yunet_net is not None and lmk_net is not None else \
                       ("YuNet (bbox)" if yunet_net is not None else "Haar (bbox)")
                draw_label(disp, f"Latencia Afectiva — 1 FPS — {mode}", 12, 28, 0.8, (255,220,140))
                draw_label(disp, "F: fullscreen  |  G: overlays  |  ESC: salir", 12, 56, 0.6, (200,255,200))

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if not fullscreen:
                    cv2.resizeWindow(win, args.width, args.height)
            elif key in (ord('g'), ord('G')):
                overlay_on = not overlay_on

            # regula ~1 FPS
            time.sleep(max(0, period - (time.time() - t0)))
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    