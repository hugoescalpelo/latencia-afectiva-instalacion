#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, argparse, time, os

DISPLAY_W, DISPLAY_H = 800, 480

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

    # Pide FOURCC deseado
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, max(1.0, fps))

    # Calienta con un read
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def open_cam_gst(index, w, h, fps):
    # Pipeline equivalente a tu ffplay (mjpeg -> jpegdec -> appsink)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=DISPLAY_W)
    ap.add_argument("--height", type=int, default=DISPLAY_H)
    args = ap.parse_args()

    # === Secuencia de intentos (como ffplay) ===
    cap = None

    # 1) V4L2 + MJPG @ 1280x720
    cap = open_cam_v4l2(args.camera, 1280, 720, args.fps, 'MJPG')
    if cap:
        describe_cap(cap, "V4L2 MJPG 1280x720")
    else:
        print("[!] Falló V4L2 MJPG 1280x720, probando YUYV…")
        # 2) V4L2 + YUYV @ 1280x720
        cap = open_cam_v4l2(args.camera, 1280, 720, args.fps, 'YUYV')
        if cap:
            describe_cap(cap, "V4L2 YUYV 1280x720")
        else:
            print("[!] Falló V4L2 YUYV 1280x720, probando 800x480 MJPG…")
            # 3) V4L2 + MJPG @ 800x480
            cap = open_cam_v4l2(args.camera, 800, 480, args.fps, 'MJPG')
            if cap:
                describe_cap(cap, "V4L2 MJPG 800x480")
            else:
                print("[!] Falló V4L2 800x480. Intento por GStreamer explícito…")
                # 4) GStreamer explícito (solo si tienes GStreamer bien)
                cap = open_cam_gst(args.camera, 1280, 720, max(1, int(args.fps)))
                if cap:
                    describe_cap(cap, "GStreamer MJPG 1280x720")
                else:
                    raise SystemExit("❌ No se pudo abrir la cámara con ningún método. Prueba --camera 1 o revisa v4l2-ctl --list-formats-ext")

    win = "Cam Fullscreen 800x480"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Abrimos en fullscreen de una
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

            dt = time.time() - t0
            rest = period - dt
            if rest > 0:
                time.sleep(rest)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Recomendación en Wayland: usar XCB para la ventana OpenCV
    if os.environ.get("QT_QPA_PLATFORM", "") == "":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
