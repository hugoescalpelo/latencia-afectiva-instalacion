#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Top-3 (FER+) + Face boxes
- Abre C922 a pantalla completa (800x480) con V4L2/MJPG (fallbacks)
- Detecta rostros (Haarcascade)
- Para cada rostro muestra: rectángulo + 3 emociones más probables con %
- Diseñado para 1 FPS (tema: latencia)
Teclas:
  ESC = salir
  F   = alternar fullscreen/ventana
"""
import os, time, argparse
import cv2
import numpy as np

DISPLAY_W, DISPLAY_H = 800, 480
EMO_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

def scale_cover(frame, target_w, target_h):
    h, w = frame.shape[:2]
    s = max(target_w / w, target_h / h)
    nw, nh = int(w*s), int(h*s)
    r = cv2.resize(frame, (nw, nh))
    x0, y0 = (nw-target_w)//2, (nh-target_h)//2
    return r[y0:y0+target_h, x0:x0+target_w]

def fourcc_str(val):
    try:
        return "".join([chr(int(val) >> (8*i) & 0xFF) for i in range(4)])
    except Exception:
        return str(val)

def open_cam_v4l2(index, w, h, fps, fourcc='MJPG'):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, max(1.0, fps))
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def open_cam_gst(index, w, h, fps):
    dev = f"/dev/video{index}"
    pipe = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,framerate={int(fps)}/1,width={w},height={h} ! "
        f"jpegdec ! videoconvert ! appsink drop=1 sync=false"
    )
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None

def describe_cap(cap, tag=""):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r = cap.get(cv2.CAP_PROP_FPS)
    four = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "?"
    print(f"[OK] {tag} backend={backend} size={w}x{h} fps~{fps_r:.1f} fourcc={four}")

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def load_models(models_dir):
    haar = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
    onnx = os.path.join(models_dir, "emotion-ferplus-8.onnx")

    face_cascade = cv2.CascadeClassifier(haar)
    if face_cascade.empty():
        raise SystemExit(f"❌ No se pudo cargar Haarcascade en {haar}")

    net = cv2.dnn.readNetFromONNX(onnx)
    return face_cascade, net

def infer_emotion(net, face_bgr):
    # FER+ espera 64x64 GRAYSCALE, rango 0..1
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
    face = face.astype(np.float32) / 255.0
    blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(64,64), mean=(0,), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(-1)  # (8,)
    probs = softmax(out)
    idx = np.argsort(-probs)[:3]
    return [(EMO_LABELS[i], float(probs[i])) for i in idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=DISPLAY_W)
    parser.add_argument("--height", type=int, default=DISPLAY_H)
    parser.add_argument("--models", type=str, default=os.path.expanduser("~/Documents/GitHub/latencia-afectiva-instalacion/models"))
    args = parser.parse_args()

    # cargar modelos
    face_cascade, emo_net = load_models(args.models)

    # abrir cámara (mismo orden robusto del visor)
    cap = open_cam_v4l2(args.camera, 1280, 720, args.fps, 'MJPG')
    if cap:
        describe_cap(cap, "V4L2 MJPG 1280x720")
    else:
        print("[!] Falló V4L2 MJPG 1280x720, probando YUYV…")
        cap = open_cam_v4l2(args.camera, 1280, 720, args.fps, 'YUYV')
        if cap:
            describe_cap(cap, "V4L2 YUYV 1280x720")
        else:
            print("[!] Falló V4L2 YUYV 1280x720, probando 800x480 MJPG…")
            cap = open_cam_v4l2(args.camera, 800, 480, args.fps, 'MJPG')
            if cap:
                describe_cap(cap, "V4L2 MJPG 800x480")
            else:
                print("[!] Falló V4L2 800x480. Intento por GStreamer explícito…")
                cap = open_cam_gst(args.camera, 1280, 720, max(1, int(args.fps)))
                if cap:
                    describe_cap(cap, "GStreamer MJPG 1280x720")
                else:
                    raise SystemExit("❌ No se pudo abrir la cámara. Prueba --camera 1 o revisa v4l2-ctl --list-formats-ext")

    win = "Emotion Top-3"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True

    period = 1.0 / max(0.1, args.fps)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            # detección de rostros en versión reducida para rendimiento
            frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            gray_small  = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.1, minNeighbors=5,
                minSize=(60,60), flags=cv2.CASCADE_SCALE_IMAGE
            )

            # dibujar resultados sobre una copia escalada a 800x480
            disp = scale_cover(frame, args.width, args.height)

            # factor de correspondencia small->original->display
            # reconstruimos bbox en coords del original:
            for (x, y, w, h) in faces:
                X, Y, W, H = int(x*2), int(y*2), int(w*2), int(h*2)
                # recorte seguro del rostro:
                X0, Y0 = max(0, X), max(0, Y)
                X1, Y1 = min(frame.shape[1], X+W), min(frame.shape[0], Y+H)
                face_roi = frame[Y0:Y1, X0:X1]
                if face_roi.size == 0:
                    continue

                top3 = infer_emotion(emo_net, face_roi)

                # proyectar a coords de 'disp' (cover)
                # para evitar cálculos largos, dibujamos directamente sobre 'frame' y luego re-escalamos:
                cv2.rectangle(frame, (X0, Y0), (X1, Y1), (0, 255, 255), 2)
                # texto (tres líneas)
                base_y = Y0 - 10 if Y0 - 10 > 20 else Y1 + 20
                for i, (lab, p) in enumerate(top3):
                    txt = f"{lab}: {p*100:.1f}%"
                    cv2.putText(frame, txt, (X0, base_y + i*18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(frame, txt, (X0, base_y + i*18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # ahora escalamos con cover y mostramos
            disp = scale_cover(frame, args.width, args.height)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                )
                if not fullscreen:
                    cv2.resizeWindow(win, args.width, args.height)

            # 1 FPS (latencia)
            dt = time.time() - t0
            rest = period - dt
            if rest > 0:
                time.sleep(rest)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # en Wayland, usar XCB para la ventana de OpenCV si no está definido
    if os.environ.get("QT_QPA_PLATFORM", "") == "":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
