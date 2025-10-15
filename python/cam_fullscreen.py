#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cámara fullscreen para DSI 800x480 (Raspberry Pi)
- Muestra la C922 en pantalla completa a 800x480
- Escalado tipo "cover": llena la pantalla recortando lo mínimo sin deformar
- 1 FPS por defecto (tema: latencia); ajustable con --fps
Teclas:
  ESC = salir
  F   = alterna fullscreen/ventana
"""
import cv2, argparse, time

DISPLAY_W, DISPLAY_H = 800, 480

def scale_cover(frame, target_w, target_h):
    """Escala conservando proporción para LLenar (cover) y recorta centro."""
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    # recorte centrado
    x0 = (nw - target_w) // 2
    y0 = (nh - target_h) // 2
    return resized[y0:y0+target_h, x0:x0+target_w]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Índice de cámara (C922 suele ser 0)")
    ap.add_argument("--fps", type=float, default=1.0, help="FPS de captura/actualización")
    ap.add_argument("--width", type=int, default=DISPLAY_W)
    ap.add_argument("--height", type=int, default=DISPLAY_H)
    args = ap.parse_args()

    # Intenta pedir una resolución que la C922 maneja bien (720p) y luego recortamos a 800x480
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("❌ No se pudo abrir la cámara. Prueba --camera 1 o revisa v4l2-ctl --list-devices")

    # Ajustes razonables para C922
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 720p nativo suele dar mejor nitidez
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, max(1.0, args.fps))

    win = "Cam Fullscreen 800x480"
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

            # espejo (opcional, se ve más natural)
            frame = cv2.flip(frame, 1)

            # escalar a "cover" y ajustar a 800x480
            disp = scale_cover(frame, args.width, args.height)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
                if not fullscreen:
                    cv2.resizeWindow(win, args.width, args.height)

            # regula FPS (latencia visible ❤️)
            dt = time.time() - t0
            sleep_left = period - dt
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
