"""
fast_train_vehicle.py
---------------------
RTX 4060 (8 GB VRAM) için ~2-3 saatte tamamlanan yüksek doğruluklu YOLOv8s eğitimi.

Strateji
  • Training setinden sınıf dengeli ~12 000 görsel alt-kümesi
    (van / bus / motorcycle / truck az temsil edildiği için öncelikli)
  • batch=16, AdamW optimizer, cosine LR decay → daha hızlı yakınsama
  • 30 epoch, patience=8 erken durdurma
  • Validation tam dataset (7 087 görsel) üzerinde → gerçekçi mAP

Kullanım
  python fast_train_vehicle.py            # alt-küme ile hızlı eğitim
  python fast_train_vehicle.py --full     # tam dataset (yavaş ama maksimum)
  python fast_train_vehicle.py --resume   # kaldığı yerden devam
"""

import sys
import random
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
# Yollar
# ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "merged_vehicle_dataset"
RUNS_DIR    = BASE_DIR / "runs"
LOGS_DIR    = BASE_DIR / "logs"
LOG_FILE    = LOGS_DIR / "fast_train_log.txt"
SUBSET_DIR  = BASE_DIR / "subset"

# ──────────────────────────────────────────────────────────────────
# Subset parametreleri
# ──────────────────────────────────────────────────────────────────
TARGET_TRAIN  = 20_000
RANDOM_SEED   = 42
CLASS_NAMES   = {0: "car", 1: "motorcycle", 2: "truck", 3: "bus", 4: "van"}

# None → tüm görselleri al; sayı → o kadarla sınırla
CLASS_BUDGET  = {
    4: None,   # van        — tümünü al (~1 658)
    3: 4_500,  # bus        — max 7 219 mevcut
    1: 4_500,  # motorcycle — max 7 183 mevcut
    2: 4_500,  # truck      — max 9 686 mevcut
    0: 8_000,  # car (fill) — gerisi
}

# ──────────────────────────────────────────────────────────────────
# Eğitim konfigürasyonu
# ──────────────────────────────────────────────────────────────────
TRAIN_CFG = dict(
    imgsz         = 640,
    epochs        = 50,
    batch         = 16,
    device        = 0,           # RTX 4060 = GPU 0
    workers       = 4,
    cache         = False,       # cache="disk" SSD varsa daha hızlı
    patience      = 8,
    project       = str(RUNS_DIR / "vehicle_fast"),
    name          = "yolov20ks_fast_run",
    exist_ok      = True,
    verbose       = True,
    # ── Optimizer ────────────────────────────────────────────────
    optimizer     = "AdamW",     # SGD'ye göre ~20% daha hızlı yakınsama
    lr0           = 0.001,
    lrf           = 0.01,        # final_lr = lr0 * lrf
    cos_lr        = True,        # cosine decay
    warmup_epochs = 2,
    weight_decay  = 0.0005,
    # ── Augmentation ─────────────────────────────────────────────
    hsv_h         = 0.015,
    hsv_s         = 0.5,
    hsv_v         = 0.4,
    degrees       = 5,
    translate     = 0.1,
    scale         = 0.5,
    shear         = 2.0,
    perspective   = 0.0005,
    flipud        = 0.0,         # dikey flip yok (araç verisi)
    fliplr        = 0.5,
    mosaic        = 1.0,
    mixup         = 0.05,
    close_mosaic  = 5,
)


# ──────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────
def log(msg=""):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _img_path(stem: str, image_dir: Path):
    """Bir stem için gerçek resim yolunu döndürür (jpg/png vb.)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = image_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────────
# Dengeli subset oluşturucu
# ──────────────────────────────────────────────────────────────────
def build_balanced_subset() -> Path:
    """
    Train etiket klasörünü tarar, sınıf dengeli ~TARGET_TRAIN görsel seçer.
    subset/train.txt ve subset/data.yaml dosyalarını oluşturur.
    Döner: subset data.yaml yolu
    """
    log("=" * 60)
    log("  DENGELI ALT-KÜME OLUŞTURULUYOR")
    log("=" * 60)

    label_dir = DATASET_DIR / "labels" / "train"
    image_dir = DATASET_DIR / "images" / "train"

    log(f"Label klasörü taranıyor: {label_dir}")
    log("(Bu işlem 1-2 dakika sürebilir...)")

    class_to_stems: dict[int, list] = defaultdict(list)

    label_files = list(label_dir.glob("*.txt"))
    for lf in label_files:
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

    log(f"\nToplam taranan label: {len(label_files):,}")
    log("Sınıf başına görsel (raw):")
    for cls, name in CLASS_NAMES.items():
        log(f"  [{cls}] {name:12s}: {len(class_to_stems[cls]):,}")

    # ── Seçim: azdan çoğa (van önce, car son) ───────────────────
    random.seed(RANDOM_SEED)
    selected: set[str] = set()

    log("\nSeçim başlıyor (öncelik: van > bus > motorcycle > truck > car):")
    for cls in [4, 3, 1, 2, 0]:
        stems    = class_to_stems[cls].copy()
        random.shuffle(stems)
        budget   = CLASS_BUDGET[cls]
        if budget is None:
            take = stems
        else:
            available = [s for s in stems if s not in selected]
            take = available[:budget]
        before = len(selected)
        selected.update(take)
        added  = len(selected) - before
        log(f"  [{cls}] {CLASS_NAMES[cls]:12s}: +{added:,}  →  toplam {len(selected):,}")

    # ── Geçerli image yollarını topla ───────────────────────────
    valid_paths: list[str] = []
    for stem in selected:
        p = _img_path(stem, image_dir)
        if p:
            valid_paths.append(str(p.resolve()).replace("\\", "/"))

    log(f"\nDoğrulanan görsel: {len(valid_paths):,}")

    # ── train.txt yaz ───────────────────────────────────────────
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    train_txt = SUBSET_DIR / "train.txt"
    train_txt.write_text("\n".join(valid_paths), encoding="utf-8")
    log(f"train.txt → {train_txt}")

    # ── data.yaml yaz ───────────────────────────────────────────
    data_yaml = SUBSET_DIR / "data.yaml"
    content = {
        "path" : str(DATASET_DIR.resolve()).replace("\\", "/"),
        "train": str(train_txt.resolve()).replace("\\", "/"),
        "val"  : "images/valid",
        "test" : "images/test",
        "nc"   : 5,
        "names": [CLASS_NAMES[i] for i in range(5)],
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    log(f"data.yaml  → {data_yaml}")

    # ── Subset dağılımı doğrula ──────────────────────────────────
    _verify_distribution(selected, label_dir)
    return data_yaml


def _verify_distribution(stems: set[str], label_dir: Path):
    """Seçilen subset'in nesne dağılımını gösterir."""
    counts: dict[int, int] = defaultdict(int)
    for stem in stems:
        lf = label_dir / (stem + ".txt")
        if not lf.exists():
            continue
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in CLASS_NAMES:
                    counts[cls] += 1

    total = sum(counts.values())
    log(f"\nSubset nesne dağılımı (toplam: {total:,}):")
    for cls, name in CLASS_NAMES.items():
        pct = counts[cls] / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        log(f"  [{cls}] {name:12s}: {counts[cls]:6,}  ({pct:5.1f}%)  {bar}")


# ──────────────────────────────────────────────────────────────────
# Ön kontrol
# ──────────────────────────────────────────────────────────────────
def preflight(data_yaml: Path, subset_mode: bool):
    import torch
    log("=" * 60)
    log("  ÖN KONTROL")
    log("=" * 60)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"[OK] GPU      : {name}  ({mem:.1f} GB VRAM)")
        log(f"[OK] CUDA     : {torch.version.cuda}")
    else:
        log("[UYARI] CUDA bulunamadı — CPU çok yavaş olacak!")

    mode = f"Alt-küme (~{TARGET_TRAIN:,} görsel)" if subset_mode else "Tam dataset (47 380 görsel)"
    log(f"[OK] Mod      : {mode}")
    log(f"[OK] Data     : {data_yaml}")
    log(f"[OK] Model    : yolov8s.pt")
    log(f"[OK] Epochs   : {TRAIN_CFG['epochs']}  (patience={TRAIN_CFG['patience']})")
    log(f"[OK] Batch    : {TRAIN_CFG['batch']}")
    log(f"[OK] imgsz    : {TRAIN_CFG['imgsz']}")
    log(f"[OK] Optimizer: {TRAIN_CFG['optimizer']}  lr0={TRAIN_CFG['lr0']}")
    log(f"[OK] cos_lr   : {TRAIN_CFG['cos_lr']}")
    log("")


# ──────────────────────────────────────────────────────────────────
# Ana eğitim
# ──────────────────────────────────────────────────────────────────
def train(use_subset: bool = True, resume: bool = False):
    from ultralytics import YOLO

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume modu ─────────────────────────────────────────────
    if resume:
        last_pt = Path(TRAIN_CFG["project"]) / TRAIN_CFG["name"] / "weights" / "last.pt"
        if not last_pt.exists():
            log(f"[HATA] last.pt bulunamadı: {last_pt}")
            sys.exit(1)
        log(f"Kaldığı yerden devam: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
        return

    # ── Data seç ────────────────────────────────────────────────
    if use_subset:
        data_yaml = build_balanced_subset()
    else:
        data_yaml = DATASET_DIR / "data.yaml"

    preflight(data_yaml, use_subset)

    log("Eğitim başlatılıyor...\n")
    model_path = BASE_DIR / "yolov8s.pt"
    if not model_path.exists():
        log("[UYARI] yolov8s.pt bulunamadı, Ultralytics'ten indirilecek...")
        model_path = "yolov8s.pt"

    model  = YOLO(str(model_path))
    cfg    = {**TRAIN_CFG, "data": str(data_yaml)}
    results = model.train(**cfg)

    # ── Eğitim tamamlandı ───────────────────────────────────────
    log("\n" + "=" * 60)
    log("  EĞİTİM TAMAMLANDI")
    log("=" * 60)

    run_dir = Path(TRAIN_CFG["project"]) / TRAIN_CFG["name"]
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    log(f"best.pt : {best_pt}")
    log(f"last.pt : {last_pt}")

    # ── Validation (tam dataset) ─────────────────────────────────
    log("\nValidation başlatılıyor (tam validation seti — 7 087 görsel)...")
    val_model  = YOLO(str(best_pt))
    full_yaml  = DATASET_DIR / "data.yaml"

    val_res = val_model.val(
        data   = str(full_yaml),
        imgsz  = TRAIN_CFG["imgsz"],
        device = TRAIN_CFG["device"],
        split  = "val",
        verbose= False,
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
            log(f"  [{i}] {CLASS_NAMES.get(i, '?'):12s}: {m:.4f}")
    except Exception:
        pass

    # ── Test seti ────────────────────────────────────────────────
    test_dir = DATASET_DIR / "images" / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        log("\nTest seti çalıştırılıyor...")
        test_res = val_model.val(
            data   = str(full_yaml),
            imgsz  = TRAIN_CFG["imgsz"],
            device = TRAIN_CFG["device"],
            split  = "test",
            verbose= False,
        )
        log("\n── Test Metrikleri ───────────────────────────────────────")
        try:
            for k, v in test_res.results_dict.items():
                log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        except Exception:
            pass

    log(f"\nLog dosyası : {LOG_FILE}")
    log(f"Sonuç klasörü: {run_dir}")
    log("İşlem tamamlandı!")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8s Hızlı Araç Tespiti Eğitimi")
    parser.add_argument("--full",   action="store_true",
                        help="Alt-küme yerine tam dataseti kullan (yavaş)")
    parser.add_argument("--resume", action="store_true",
                        help="Eğitimi kaldığı yerden devam ettir")
    args = parser.parse_args()
    train(use_subset=not args.full, resume=args.resume)
