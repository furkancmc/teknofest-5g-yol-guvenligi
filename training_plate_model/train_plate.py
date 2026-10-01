"""
train_plate.py
--------------
YOLOv8s ile plaka tespiti eğitimi — tek komutla çalışır

Dataset : merged_plate_dataset  (7.941 görsel, 1 sınıf)
Model   : YOLOv8s
Epochs  : 100  (patience=15)
Batch   : 20

Beklenen : mAP50 ~0.88-0.94
Süre     : ~50-70 dakika (RTX 4060 Laptop)

Kullanım:
  python train_plate.py           # normal eğitim
  python train_plate.py --resume  # kaldığı yerden devam
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "merged_plate_dataset"
RUNS_DIR    = BASE_DIR / "runs"
LOGS_DIR    = BASE_DIR / "logs"
LOG_FILE    = LOGS_DIR / "train_plate_log.txt"

CLASS_NAMES = {0: "license_plate"}

TRAIN_CFG = dict(
    imgsz         = 640,
    epochs        = 100,
    batch         = 20,           # s modeli daha az VRAM; OOM olursa 16'ya düşür
    device        = 0,
    workers       = 4,
    cache         = False,
    patience      = 15,           # 100 epoch'ta erken durma için biraz gevşek
    project       = str(RUNS_DIR / "plate"),
    name          = "yolov8s_plate_run",
    exist_ok      = True,
    verbose       = True,
    # ── Optimizer ────────────────────────────────────────────────
    optimizer     = "AdamW",
    lr0           = 0.001,
    lrf           = 0.01,
    cos_lr        = True,
    warmup_epochs = 3,
    weight_decay  = 0.0005,
    # ── Augmentation — plakaya özel ──────────────────────────────
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    degrees       = 15,           # plakaların eğik/dönük durması
    translate     = 0.1,
    scale         = 0.6,          # plakalar değişken boyutta görünür
    shear         = 2.0,
    perspective   = 0.001,        # araç açısından kaynaklanan perspektif bozulma
    flipud        = 0.0,
    fliplr        = 0.5,
    mosaic        = 1.0,
    mixup         = 0.05,         # tek sınıfta yüksek mixup gereksiz
    copy_paste    = 0.1,
    close_mosaic  = 10,
)


# ──────────────────────────────────────────────────────────────────
def log(msg=""):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def preflight(data_yaml):
    import torch
    log("=" * 60)
    log("  ÖN KONTROL")
    log("=" * 60)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"[OK] GPU      : {name}  ({mem:.1f} GB)")
        log(f"[OK] CUDA     : {torch.version.cuda}")
    else:
        log("[UYARI] CUDA bulunamadı — CPU ile çok yavaş olur!")

    # Dataset kontrol
    for split in ("train", "valid", "test"):
        img_dir = DATASET_DIR / "images" / split
        n = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        log(f"[OK] {split:5s}    : {n:,} görsel")

    log(f"[OK] Data     : {data_yaml}")
    log(f"[OK] Model    : YOLOv8s")
    log(f"[OK] Epochs   : {TRAIN_CFG['epochs']}  (patience={TRAIN_CFG['patience']})")
    log(f"[OK] Batch    : {TRAIN_CFG['batch']}  (OOM olursa 16'ya düşür)")
    log(f"[OK] imgsz    : {TRAIN_CFG['imgsz']}")
    log("")


def train(resume=False):
    from ultralytics import YOLO

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume ──────────────────────────────────────────────────
    if resume:
        last_pt = Path(TRAIN_CFG["project"]) / TRAIN_CFG["name"] / "weights" / "last.pt"
        if not last_pt.exists():
            log(f"[HATA] last.pt bulunamadı: {last_pt}")
            sys.exit(1)
        log(f"Kaldığı yerden devam: {last_pt}")
        YOLO(str(last_pt)).train(resume=True)
        return

    # ── Dataset kontrol ─────────────────────────────────────────
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        log(f"[HATA] data.yaml bulunamadı: {data_yaml}")
        log("       Önce plate_merger.py çalıştır.")
        sys.exit(1)

    preflight(data_yaml)

    # ── Model yükle ─────────────────────────────────────────────
    model_pt = BASE_DIR.parent / "training_vehicle_model" / "yolov8s.pt"
    if not model_pt.exists():
        model_pt = BASE_DIR.parent / "yolov8s.pt"
    if not model_pt.exists():
        log("yolov8s.pt bulunamadı, Ultralytics'ten indiriliyor...")
        model_pt = "yolov8s.pt"

    log(f"Eğitim başlatılıyor (YOLOv8s)...\n")

    model = YOLO(str(model_pt))
    cfg   = {**TRAIN_CFG, "data": str(data_yaml)}
    model.train(**cfg)

    # ── Tamamlandı ──────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  EĞİTİM TAMAMLANDI")
    log("=" * 60)

    run_dir = Path(TRAIN_CFG["project"]) / TRAIN_CFG["name"]
    best_pt = run_dir / "weights" / "best.pt"
    log(f"best.pt : {best_pt}")

    # ── Validation ──────────────────────────────────────────────
    log("\nValidation başlatılıyor...")
    val_model = YOLO(str(best_pt))

    val_res = val_model.val(
        data    = str(data_yaml),
        imgsz   = TRAIN_CFG["imgsz"],
        device  = TRAIN_CFG["device"],
        split   = "val",
        verbose = False,
    )

    log("\n── Validation Metrikleri ─────────────────────────────────")
    try:
        for k, v in val_res.results_dict.items():
            log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    except Exception:
        pass

    log("\n── Sınıf Bazlı mAP50 ────────────────────────────────────")
    try:
        for i, m in enumerate(val_res.box.maps):
            log(f"  [{i}] {CLASS_NAMES.get(i,'?'):15s}: {m:.4f}")
    except Exception:
        pass

    # ── Test ────────────────────────────────────────────────────
    test_dir = DATASET_DIR / "images" / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        log("\nTest seti çalıştırılıyor...")
        test_res = val_model.val(
            data    = str(data_yaml),
            imgsz   = TRAIN_CFG["imgsz"],
            device  = TRAIN_CFG["device"],
            split   = "test",
            verbose = False,
        )
        log("\n── Test Metrikleri ───────────────────────────────────────")
        try:
            for k, v in test_res.results_dict.items():
                log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        except Exception:
            pass

    log(f"\nLog    : {LOG_FILE}")
    log(f"Klasör : {run_dir}")
    log("Tamamlandı!")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8s Plaka Tespiti Eğitimi")
    parser.add_argument("--resume", action="store_true", help="Kaldığı yerden devam")
    args = parser.parse_args()
    train(resume=args.resume)
