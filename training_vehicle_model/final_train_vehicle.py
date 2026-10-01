"""
final_train_vehicle.py
----------------------
Van→Car merge + YOLOv8m eğitimi — tek komutla her şey

Değişiklikler:
  • Van (4) → Car (0) merge: 4 sınıfa düşürüldü
  • YOLOv8m (medium) — s'den ~2.75x daha güçlü
  • copy_paste=0.2 — motorcycle/truck/bus için sentetik artırma
  • batch=12, patience=10

Beklenen: mAP50 ~0.78-0.84
Süre: ~16-18 saat (gece bırak)

Kullanım:
  python final_train_vehicle.py            # 23k subset, YOLOv8m
  python final_train_vehicle.py --full     # 47k tam dataset (~30-35 saat)
  python final_train_vehicle.py --resume   # kaldığı yerden devam
"""

import sys
import random
import argparse
import yaml
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "merged_vehicle_dataset"
RUNS_DIR    = BASE_DIR / "runs"
LOGS_DIR    = BASE_DIR / "logs"
LOG_FILE    = LOGS_DIR / "final_train_log.txt"
SUBSET_DIR  = BASE_DIR / "subset_4class"

TARGET_TRAIN  = 23_000
RANDOM_SEED   = 42

# 4 sınıf (van çıkarıldı)
CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "truck", 3: "bus"}

CLASS_BUDGET = {
    3: 5_000,   # bus
    1: 5_000,   # motorcycle
    2: 5_000,   # truck
    0: 8_000,   # car (van'lar da artık car)
}

TRAIN_CFG = dict(
    imgsz         = 640,
    epochs        = 50,
    batch         = 12,          # YOLOv8m için; OOM olursa 8'e düşür
    device        = 0,
    workers       = 4,
    cache         = False,
    patience      = 10,
    project       = str(RUNS_DIR / "vehicle_final"),
    name          = "yolov8m_final_run",
    exist_ok      = True,
    verbose       = True,
    # ── Optimizer ────────────────────────────────────────────────
    optimizer     = "AdamW",
    lr0           = 0.001,
    lrf           = 0.01,
    cos_lr        = True,
    warmup_epochs = 3,
    weight_decay  = 0.0005,
    # ── Augmentation ─────────────────────────────────────────────
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    degrees       = 10,
    translate     = 0.1,
    scale         = 0.5,
    shear         = 2.0,
    perspective   = 0.0005,
    flipud        = 0.0,
    fliplr        = 0.5,
    mosaic        = 1.0,
    mixup         = 0.1,
    copy_paste    = 0.2,         # azınlık sınıflar için kritik
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


def _img_path(stem, image_dir):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = image_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────────
# ADIM 1 — Van → Car merge
# ──────────────────────────────────────────────────────────────────
def merge_van_to_car():
    """
    Tüm split'lerde van (4) → car (0) olarak değiştirir.
    data.yaml'ı nc=4'e günceller.
    Zaten merge edilmişse atlar.
    """
    data_yaml = DATASET_DIR / "data.yaml"
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Zaten 4 sınıflıysa atla
    if cfg.get("nc", 5) == 4:
        log("[ATLA] Van→Car merge zaten yapılmış (nc=4).")
        return

    log("=" * 60)
    log("  VAN → CAR MERGE BAŞLIYOR")
    log("=" * 60)

    total_changed = 0
    for split in ("train", "valid", "test"):
        label_dir = DATASET_DIR / "labels" / split
        if not label_dir.exists():
            continue
        files = list(label_dir.glob("*.txt"))
        changed_files = 0
        for lf in files:
            if lf.name in ("classes.txt", "labels.txt"):
                continue
            try:
                lines = lf.read_text(encoding="utf-8").splitlines()
                new_lines = []
                modified = False
                for line in lines:
                    parts = line.strip().split()
                    if parts and parts[0] == "4":
                        parts[0] = "0"
                        new_lines.append(" ".join(parts))
                        modified = True
                    else:
                        new_lines.append(line)
                if modified:
                    lf.write_text("\n".join(new_lines), encoding="utf-8")
                    changed_files += 1
            except Exception:
                continue
        log(f"  [{split:5s}] {len(files):,} dosya tarandı, {changed_files:,} güncellendi")
        total_changed += changed_files

    log(f"\nToplam güncellenen label: {total_changed:,}")

    # Cache dosyalarını sil (stale olur)
    for cache in [
        DATASET_DIR / "labels" / "train.cache",
        DATASET_DIR / "labels" / "valid.cache",
        DATASET_DIR / "labels" / "test.cache",
    ]:
        if cache.exists():
            cache.unlink()
            log(f"  Silindi: {cache.name}")

    # data.yaml güncelle
    new_cfg = {
        "path" : str(DATASET_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val"  : "images/valid",
        "test" : "images/test",
        "nc"   : 4,
        "names": ["car", "motorcycle", "truck", "bus"],
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(new_cfg, f, allow_unicode=True, sort_keys=False)
    log(f"data.yaml güncellendi (nc=4, van kaldırıldı)")
    log("Van→Car merge tamamlandı!\n")


# ──────────────────────────────────────────────────────────────────
# ADIM 2 — 4 sınıflı dengeli subset
# ──────────────────────────────────────────────────────────────────
def build_4class_subset():
    log("=" * 60)
    log("  4 SINIFLI SUBSET OLUŞTURULUYOR")
    log("=" * 60)

    label_dir = DATASET_DIR / "labels" / "train"
    image_dir = DATASET_DIR / "images" / "train"
    log("Label taranıyor... (1-2 dk)")

    class_to_stems = defaultdict(list)
    for lf in label_dir.glob("*.txt"):
        if lf.name in ("classes.txt", "labels.txt"):
            continue
        seen = set()
        try:
            for line in lf.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if cls in CLASS_NAMES:
                        seen.add(cls)
        except Exception:
            continue
        for cls in seen:
            class_to_stems[cls].append(lf.stem)

    log("Sınıf başına görsel:")
    for cls, name in CLASS_NAMES.items():
        log(f"  [{cls}] {name:12s}: {len(class_to_stems[cls]):,}")

    random.seed(RANDOM_SEED)
    selected = set()

    log("\nSeçim (öncelik: bus > motorcycle > truck > car):")
    for cls in [3, 1, 2, 0]:
        stems     = class_to_stems[cls].copy()
        random.shuffle(stems)
        available = [s for s in stems if s not in selected]
        take      = available[:CLASS_BUDGET[cls]]
        before    = len(selected)
        selected.update(take)
        log(f"  [{cls}] {CLASS_NAMES[cls]:12s}: +{len(selected)-before:,}  → toplam {len(selected):,}")

    valid_paths = []
    for stem in selected:
        p = _img_path(stem, image_dir)
        if p:
            valid_paths.append(str(p.resolve()).replace("\\", "/"))
    log(f"\nDoğrulanan görsel: {len(valid_paths):,}")

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    train_txt = SUBSET_DIR / "train.txt"
    train_txt.write_text("\n".join(valid_paths), encoding="utf-8")

    data_yaml = SUBSET_DIR / "data.yaml"
    content = {
        "path" : str(DATASET_DIR.resolve()).replace("\\", "/"),
        "train": str(train_txt.resolve()).replace("\\", "/"),
        "val"  : "images/valid",
        "test" : "images/test",
        "nc"   : 4,
        "names": ["car", "motorcycle", "truck", "bus"],
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)

    log(f"train.txt → {train_txt}")
    log(f"data.yaml  → {data_yaml}\n")
    return data_yaml


# ──────────────────────────────────────────────────────────────────
# ADIM 3 — Eğitim
# ──────────────────────────────────────────────────────────────────
def preflight(data_yaml, mode):
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
        log("[UYARI] CUDA yok!")
    log(f"[OK] Mod      : {mode}")
    log(f"[OK] Data     : {data_yaml}")
    log(f"[OK] Model    : YOLOv8m")
    log(f"[OK] Epochs   : {TRAIN_CFG['epochs']}  (patience={TRAIN_CFG['patience']})")
    log(f"[OK] Batch    : {TRAIN_CFG['batch']}  (OOM olursa 8'e düşür)")
    log(f"[OK] copy_paste: {TRAIN_CFG['copy_paste']}")
    log("")


def train(use_subset=True, resume=False):
    from ultralytics import YOLO

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        last_pt = Path(TRAIN_CFG["project"]) / TRAIN_CFG["name"] / "weights" / "last.pt"
        if not last_pt.exists():
            log(f"[HATA] last.pt bulunamadı: {last_pt}")
            sys.exit(1)
        log(f"Kaldığı yerden devam: {last_pt}")
        YOLO(str(last_pt)).train(resume=True)
        return

    # Adım 1: merge
    merge_van_to_car()

    # Adım 2: data seç
    if use_subset:
        data_yaml = build_4class_subset()
        mode = f"4-sınıf subset (~{TARGET_TRAIN:,} görsel)"
    else:
        data_yaml = DATASET_DIR / "data.yaml"
        mode = "4-sınıf TAM DATASET (47k)"

    preflight(data_yaml, mode)

    log("Eğitim başlatılıyor (YOLOv8m)...\n")

    model_pt = BASE_DIR / "yolov8m.pt"
    if not model_pt.exists():
        log("yolov8m.pt bulunamadı, Ultralytics'ten indiriliyor...")
        model_pt = "yolov8m.pt"

    model   = YOLO(str(model_pt))
    cfg     = {**TRAIN_CFG, "data": str(data_yaml)}
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
    full_yaml = DATASET_DIR / "data.yaml"

    val_res = val_model.val(
        data=str(full_yaml), imgsz=TRAIN_CFG["imgsz"],
        device=TRAIN_CFG["device"], split="val", verbose=False,
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
            log(f"  [{i}] {CLASS_NAMES.get(i,'?'):12s}: {m:.4f}")
    except Exception:
        pass

    # ── Test ────────────────────────────────────────────────────
    test_dir = DATASET_DIR / "images" / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        log("\nTest seti çalıştırılıyor...")
        test_res = val_model.val(
            data=str(full_yaml), imgsz=TRAIN_CFG["imgsz"],
            device=TRAIN_CFG["device"], split="test", verbose=False,
        )
        log("\n── Test Metrikleri ───────────────────────────────────────")
        try:
            for k, v in test_res.results_dict.items():
                log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        except Exception:
            pass

    log(f"\nLog : {LOG_FILE}")
    log(f"Klasör : {run_dir}")
    log("Tamamlandı!")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8m Final Araç Tespiti")
    parser.add_argument("--full",   action="store_true", help="Tam 47k dataset (~30 saat)")
    parser.add_argument("--resume", action="store_true", help="Kaldığı yerden devam")
    args = parser.parse_args()
    train(use_subset=not args.full, resume=args.resume)
