"""
Araç ve plaka tespiti — teknofest_test videolarına uygular.
Çıktı: results/ klasörüne annotated video + JSON sonuç dosyası.
"""

import json
import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE = Path(__file__).parent

VEHICLE_MODEL_PATH = BASE / "training_vehicle_model/runs/vehicle_final/yolov8m_final_run/weights/best.pt"
PLATE_MODEL_PATH   = BASE / "training_plate_model/runs/plate/yolov8s_plate_run/weights/best.pt"
VIDEO_DIR          = BASE / "teknofest_test"
OUTPUT_DIR         = BASE / "results"

CONF_VEHICLE = 0.25
CONF_PLATE   = 0.30
IOU          = 0.45
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Renk paleti (BGR)
COLORS = {
    "car":          (0,   200, 50),
    "motorcycle":   (50,  150, 255),
    "truck":        (0,   100, 255),
    "bus":          (200, 50,  255),
    "license_plate":(0,   220, 220),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def process_video(video_path: Path, vehicle_model: YOLO, plate_model: YOLO, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [HATA] Video açılamadı: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = out_dir / f"{video_path.stem}_annotated.mp4"
    out_json_path  = out_dir / f"{video_path.stem}_results.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    all_frames = []
    frame_idx  = 0
    t_start    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Her iki modeli aynı frame üzerinde çalıştır
        v_results = vehicle_model.predict(frame, conf=CONF_VEHICLE, iou=IOU,
                                          device=DEVICE, verbose=False)[0]
        p_results = plate_model.predict(frame, conf=CONF_PLATE, iou=IOU,
                                        device=DEVICE, verbose=False)[0]

        frame_detections = {"frame": frame_idx, "vehicles": [], "plates": []}

        for box in v_results.boxes:
            cls_id = int(box.cls[0])
            label  = vehicle_model.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color  = COLORS.get(label, DEFAULT_COLOR)
            draw_box(frame, x1, y1, x2, y2, label, conf, color)
            frame_detections["vehicles"].append({
                "class": label, "conf": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

        for box in p_results.boxes:
            cls_id = int(box.cls[0])
            label  = plate_model.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color  = COLORS.get(label, DEFAULT_COLOR)
            draw_box(frame, x1, y1, x2, y2, label, conf, color)
            frame_detections["plates"].append({
                "class": label, "conf": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

        # Frame bilgisi sol üst köşeye
        info = (f"Frame {frame_idx}/{total} | "
                f"Arac: {len(frame_detections['vehicles'])} | "
                f"Plaka: {len(frame_detections['plates'])}")
        cv2.putText(frame, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame)
        all_frames.append(frame_detections)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"    {frame_idx}/{total} frame işlendi  ({fps_proc:.1f} fps)")

    cap.release()
    writer.release()

    # JSON kaydet
    summary = {
        "video": video_path.name,
        "total_frames": frame_idx,
        "fps": fps,
        "resolution": [width, height],
        "vehicle_model": str(VEHICLE_MODEL_PATH),
        "plate_model": str(PLATE_MODEL_PATH),
        "frames": all_frames,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"  Tamamlandi: {frame_idx} frame, {elapsed:.1f}s")
    print(f"  Video  -> {out_video_path}")
    print(f"  JSON   -> {out_json_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Cihaz: {DEVICE.upper()}")
    print("Modeller yukleniyor...")
    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    plate_model   = YOLO(str(PLATE_MODEL_PATH))
    print("  Arac modeli  :", VEHICLE_MODEL_PATH.name)
    print("  Plaka modeli :", PLATE_MODEL_PATH.name)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"[UYARI] {VIDEO_DIR} icinde .mp4 dosyasi bulunamadi.")
        return

    for video_path in videos:
        print(f"\n--- {video_path.name} isleniyor ---")
        process_video(video_path, vehicle_model, plate_model, OUTPUT_DIR)

    print("\nTum videolar tamamlandi. Sonuclar -> results/")


if __name__ == "__main__":
    main()
