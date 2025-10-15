#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Cámara C922 a 1 FPS, pantalla completa 800x480, SOLO BBOX (sin cv2.face)
import cv2, time, argparse, sys, os

DISPLAY_W, DISPLAY_H = 800, 480

def draw_label(img, text, x=12, y=28, scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--width", type=int, default=DISPLAY_W)
    ap.add_argument("--height", type=int, default=DISPLAY_H)
    ap.add_argument("--models", type=str, default=os.path.expanduser("~/Documents/GitHub/latencia-afectiva-instalacion/models"))
    args = ap.parse_args()

    haar_path = os.path.join(args.models, "haarcascade_frontalface_default.xml")
    if not os.path.exists(haar_path):
        print("❌ Falta haarcascade_frontalface_default.xml en --models", file=sys.stderr); sys.exit(1)

    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        print("❌ No se pudo cargar el Haar cascade", file=sys.stderr); sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara", file=sys.stderr); sys.exit(1)
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (80, 80))

            vis = frame.copy()
            for (x,y,w,h) in faces:
                cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,255), 2)

            disp = cv2.resize(vis, (args.width, args.height))

            if overlay_on:
                draw_label(disp, "Latencia Afectiva — 1 FPS (Haar bbox)", 12, 28, 0.8, (255,220,140))
                draw_label(disp, "F: fullscreen  |  G: overlays  |  ESC: salir", 12, 56, 0.6, (200,255,200))

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if not fullscreen:
                    cv2.resizeWindow(win, args.width, args.height)
            elif key in (ord('g'), ord('G')):
                overlay_on = not overlay_on

            time.sleep(max(0, period - (time.time()-t0)))
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
PY