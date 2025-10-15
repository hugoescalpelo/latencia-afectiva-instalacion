# ...
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    # Forzar backend V4L2 (evita GStreamer) y MJPG (C922 lo hace muy bien)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("❌ No se pudo abrir la cámara. Prueba --camera 1.")

    # Pedir MJPG explícito
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Pide 1280x720 (se ve mejor) y escalamos/recortamos a 800x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, max(1.0, args.fps))
# ...
