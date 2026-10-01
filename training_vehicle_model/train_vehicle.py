"""
train_vehicle.py
-----------------
RTX 4060 (8 GB VRAM) için optimize edilmiş YOLOv8 araç tespiti eğitim scripti.

Kullanım:
  python train_vehicle.py           # İlk eğitim
  python train_vehicle.py --resume  # Kaldığı yerden devam
"""

import sys
import argparse
import torch
import shutil
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# Yollar
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_YAML   = BASE_DIR.parent / "merged_vehicle_dataset" / "data.yaml"
RUNS_DIR    = BASE_DIR / "runs"
LOGS_DIR    = BASE_DIR / "logs"
LOG_FILE    = LOGS_DIR / "training_log.txt"

# ─────────────────────────────────────────────────────────────────
# Eğitim Hiperparametreleri
# ─────────────────────────────────────────────────────────────────
TRAIN_CONFIG = dict(
    model        = "yolov8s.pt",
    data         = str(DATA_YAML),
    imgsz        = 640,
    epochs       = 50,
    batch        = 8,          # RTX 4060 8GB için güvenli; RAM sorunu için 16'dan düşürüldü
    device       = 0,          # RTX 4060 = GPU 0
    workers      = 4,          # RAM yetersizliği nedeniyle 8'den düşürüldü
    cache        = False,       # 77.8 GB RAM gerekiyor ama 39 GB RAM var, cache kapatıldı
    patience     = 15,
    project      = str(RUNS_DIR / "vehicle_training"),
    name         = "yolov8s_vehicle_first_run",
    exist_ok     = True,
    verbose      = True,
    # ─── Augmentation ──────────────────────────────────────────
    hsv_h        = 0.015,
    hsv_s        = 0.5,
    hsv_v        = 0.4,
    degrees      = 5,
    translate    = 0.1,
    scale        = 0.5,
    shear        = 2.0,
    perspective  = 0.0005,
    flipud       = 0.0,        # Dikey flip YOK (araç verisinde mantıksız)
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.05,
    copy_paste   = 0.0,
    close_mosaic = 10,
)


# ─────────────────────────────────────────────────────────────────
# Loglama yardımcısı
# ─────────────────────────────────────────────────────────────────
def log(msg="", also_file=True):
    ts = datetime.now().strftime("%H:%M:%S")
    full = f"[{ts}] {msg}"
    print(full)
    if also_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full + "\n")


# ─────────────────────────────────────────────────────────────────
# Ön kontroller
# ─────────────────────────────────────────────────────────────────
def preflight_checks():
    log("=" * 60)
    log("  YOLOv8 ARAÇ TESPİTİ EĞİTİMİ — ÖN KONTROL")
    log("=" * 60)

    # data.yaml
    if not DATA_YAML.exists():
        log(f"[HATA] data.yaml bulunamadı: {DATA_YAML}")
        sys.exit(1)
    log(f"[OK] data.yaml bulundu   : {DATA_YAML}")

    # GPU / CUDA
    if not torch.cuda.is_available():
        log("[UYARI] CUDA bulunamadı! Eğitim CPU üzerinde çalışacak (çok yavaş).")
    else:
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log(f"[OK] GPU bulundu          : {gpu_name}")
        log(f"[OK] VRAM                 : {gpu_mem_gb:.1f} GB")
        log(f"[OK] CUDA sürümü          : {torch.version.cuda}")

    # Train görsel sayısı
    train_img_dir = DATA_YAML.parent / "images" / "train"
    n_train = len(list(train_img_dir.glob("*.*"))) if train_img_dir.exists() else 0
    log(f"[OK] Train görsel sayısı  : {n_train:,}")

    valid_img_dir = DATA_YAML.parent / "images" / "valid"
    n_valid = len(list(valid_img_dir.glob("*.*"))) if valid_img_dir.exists() else 0
    log(f"[OK] Valid görsel sayısı  : {n_valid:,}")

    log(f"\n  Model           : {TRAIN_CONFIG['model']}")
    log(f"  Epochs          : {TRAIN_CONFIG['epochs']}")
    log(f"  Batch size      : {TRAIN_CONFIG['batch']}")
    log(f"  Image size      : {TRAIN_CONFIG['imgsz']}")
    log(f"  Patience        : {TRAIN_CONFIG['patience']}")
    log(f"  Workers         : {TRAIN_CONFIG['workers']}")
    log(f"  Çıktı klasörü   : {TRAIN_CONFIG['project']}/{TRAIN_CONFIG['name']}")
    log("")


# ─────────────────────────────────────────────────────────────────
# Ana eğitim fonksiyonu
# ─────────────────────────────────────────────────────────────────
def train(resume=False):
    from ultralytics import YOLO

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    preflight_checks()

    log("Eğitim başlatılıyor...")
    log(f"Augmentation ayarları: flipud={TRAIN_CONFIG['flipud']} (Dikey flip KAPALI)")
    log("")

    model = YOLO(TRAIN_CONFIG["model"])

    if resume:
        last_pt = (Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "last.pt")
        if last_pt.exists():
            log(f"Eğitim kaldığı yerden devam ediyor: {last_pt}")
            model = YOLO(str(last_pt))
            results = model.train(resume=True)
        else:
            log(f"[HATA] last.pt bulunamadı: {last_pt}")
            sys.exit(1)
    else:
        cfg = {k: v for k, v in TRAIN_CONFIG.items() if k != "model"}
        results = model.train(**cfg)

    # ─── Eğitim tamamlandı ─────────────────────────────────────
    log("")
    log("=" * 60)
    log("  EĞİTİM TAMAMLANDI")
    log("=" * 60)

    run_dir = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"]
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    log(f"[OK] best.pt  : {best_pt}")
    log(f"[OK] last.pt  : {last_pt}")

    # ─── Otomatik validation ───────────────────────────────────
    log("\nOtomatik validation başlatılıyor...")
    val_model = YOLO(str(best_pt))
    val_results = val_model.val(
        data=str(DATA_YAML),
        imgsz=TRAIN_CONFIG["imgsz"],
        device=TRAIN_CONFIG["device"],
    )

    log("\n── Validation Sonuçları ──────────────────────────────")
    metrics = val_results.results_dict
    for k, v in metrics.items():
        log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ─── Sınıf bazlı mAP ──────────────────────────────────────
    CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "truck", 3: "bus", 4: "van"}
    try:
        class_map50 = val_results.box.maps  # her sınıf için mAP50
        log("\n── Sınıf Bazlı mAP50 ─────────────────────────────────")
        for i, m in enumerate(class_map50):
            log(f"  [{i}] {CLASS_NAMES.get(i, '?'):12s}: {m:.4f}")
    except Exception:
        pass

    # ─── Sonuç dosyaları ──────────────────────────────────────
    log("\n── Çıktı Dosyaları ───────────────────────────────────")
    for fname in ["confusion_matrix.png", "PR_curve.png", "results.png", "results.csv"]:
        p = run_dir / fname
        if p.exists():
            log(f"  {p}")

    # ─── Test seti ─────────────────────────────────────────────
    test_img_dir = DATA_YAML.parent / "images" / "test"
    if test_img_dir.exists() and any(test_img_dir.iterdir()):
        log("\nTest seti tespit edildi, test başlatılıyor...")
        test_results = val_model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=TRAIN_CONFIG["imgsz"],
            device=TRAIN_CONFIG["device"],
        )
        log("Test tamamlandı.")

    log(f"\nLog dosyası: {LOG_FILE}")
    log("İşlem tamamlandı!")


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Araç Tespiti Eğitimi")
    parser.add_argument("--resume", action="store_true", help="Eğitimi kaldığı yerden devam ettir")
    args = parser.parse_args()
    train(resume=args.resume)
