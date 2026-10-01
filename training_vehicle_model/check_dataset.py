"""
check_dataset.py
-----------------
Eğitim öncesi dataset doğrulama scripti.
- data.yaml kontrolü
- Train/valid/test image-label sayısı eşleşmesi
- Boş label dosyası tespiti
- Hatalı YOLO format tespiti
- Class ID aralığı kontrolü (0-4)
- Koordinat aralığı kontrolü (0-1)
- Sınıf bazlı nesne sayımı
- Raporu logs/dataset_check_report.txt dosyasına yaz
"""

import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# Yollar
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "merged_vehicle_dataset"
YAML_PATH   = DATASET_DIR / "data.yaml"
LOGS_DIR    = BASE_DIR / "logs"
REPORT_PATH = LOGS_DIR / "dataset_check_report.txt"

VALID_CLASS_IDS = {0, 1, 2, 3, 4}
SPLITS          = ["train", "valid", "test"]

# ─────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────
lines_output = []

def log(msg=""):
    print(msg)
    lines_output.append(msg)

def section(title):
    bar = "=" * 60
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

# ─────────────────────────────────────────────────────────────────
# 1. data.yaml kontrolü
# ─────────────────────────────────────────────────────────────────
def check_yaml():
    section("1. DATA.YAML KONTROLÜ")
    if not YAML_PATH.exists():
        log(f"[HATA] data.yaml bulunamadı: {YAML_PATH}")
        sys.exit(1)

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log(f"Dosya   : {YAML_PATH}")
    log(f"train   : {cfg.get('train', 'YOK')}")
    log(f"val     : {cfg.get('val', 'YOK')}")
    log(f"test    : {cfg.get('test', 'YOK')}")
    log(f"nc      : {cfg.get('nc', cfg.get('names'))}")
    log(f"names   : {cfg.get('names', 'YOK')}")

    names = cfg.get("names", {})
    if isinstance(names, dict):
        name_set = set(names.keys())
    else:
        name_set = set(range(len(names)))

    if name_set == VALID_CLASS_IDS:
        log("[OK] Class ID'ler (0-4) yaml ile uyuşuyor.")
    else:
        log(f"[UYARI] Yaml'daki class ID'ler: {sorted(name_set)}, Beklenen: {sorted(VALID_CLASS_IDS)}")

    return cfg

# ─────────────────────────────────────────────────────────────────
# 2. Split bazlı kontroller
# ─────────────────────────────────────────────────────────────────
def check_split(split):
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split

    result = {
        "split": split,
        "images": 0,
        "labels": 0,
        "empty_labels": [],
        "bad_format": [],
        "invalid_class": [],
        "out_of_range_coords": [],
        "missing_label": [],
        "missing_image": [],
        "class_counts": defaultdict(int),
    }

    if not images_dir.exists():
        log(f"[UYARI] Klasör bulunamadı: {images_dir}")
        return result
    if not labels_dir.exists():
        log(f"[UYARI] Klasör bulunamadı: {labels_dir}")
        return result

    img_files = {f.stem: f for f in images_dir.iterdir()
                 if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
    lbl_files = {f.stem: f for f in labels_dir.iterdir() if f.suffix == ".txt"}

    result["images"] = len(img_files)
    result["labels"] = len(lbl_files)

    # Eşleşme kontrolü
    for stem in img_files:
        if stem not in lbl_files:
            result["missing_label"].append(stem)
    for stem in lbl_files:
        if stem not in img_files:
            result["missing_image"].append(stem)

    # Her label dosyasını detaylı incele
    for stem, lbl_path in lbl_files.items():
        with open(lbl_path, "r") as f:
            raw_lines = f.readlines()

        filtered = [l.strip() for l in raw_lines if l.strip()]

        if len(filtered) == 0:
            result["empty_labels"].append(lbl_path.name)
            continue

        for line in filtered:
            parts = line.split()
            # Format: class cx cy w h (5 değer)
            if len(parts) != 5:
                result["bad_format"].append(f"{lbl_path.name}: '{line}'")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                result["bad_format"].append(f"{lbl_path.name}: '{line}'")
                continue

            if cls_id not in VALID_CLASS_IDS:
                result["invalid_class"].append(f"{lbl_path.name}: class={cls_id}")
            else:
                result["class_counts"][cls_id] += 1

            for val, name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
                if not (0.0 <= val <= 1.0):
                    result["out_of_range_coords"].append(
                        f"{lbl_path.name}: {name}={val:.4f}"
                    )
                    break

    return result

# ─────────────────────────────────────────────────────────────────
# 3. Rapor
# ─────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "truck", 3: "bus", 4: "van"}

def print_split_report(r):
    split = r["split"].upper()
    log(f"\n── {split} ──────────────────────────────")
    log(f"  Görsel sayısı : {r['images']}")
    log(f"  Label sayısı  : {r['labels']}")

    match = (r["images"] == r["labels"] and
             len(r["missing_label"]) == 0 and
             len(r["missing_image"]) == 0)
    log(f"  Eşleşme durumu: {'✓ BAŞARILI' if match else '✗ HATA'}")

    if r["missing_label"]:
        log(f"  [HATA] Label'sız görsel ({len(r['missing_label'])}): {r['missing_label'][:3]}...")
    if r["missing_image"]:
        log(f"  [HATA] Görselsiz label ({len(r['missing_image'])}): {r['missing_image'][:3]}...")
    if r["empty_labels"]:
        log(f"  [UYARI] Boş label dosyası ({len(r['empty_labels'])}): {r['empty_labels'][:3]}")
    if r["bad_format"]:
        log(f"  [HATA] Hatalı format ({len(r['bad_format'])}): {r['bad_format'][:3]}")
    if r["invalid_class"]:
        log(f"  [HATA] Geçersiz class ID ({len(r['invalid_class'])}): {r['invalid_class'][:3]}")
    if r["out_of_range_coords"]:
        log(f"  [HATA] Koordinat aralık dışı ({len(r['out_of_range_coords'])}): {r['out_of_range_coords'][:3]}")

    if r["class_counts"]:
        log(f"  Sınıf bazlı nesne sayıları:")
        for cid in sorted(r["class_counts"]):
            log(f"    [{cid}] {CLASS_NAMES.get(cid,'?'):12s}: {r['class_counts'][cid]:,}")


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Dataset Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Dataset yolu: {DATASET_DIR}")

    cfg = check_yaml()

    section("2. SPLIT BAZLI KONTROLLER")

    all_results = []
    total_class_counts = defaultdict(int)
    overall_ok = True

    for split in SPLITS:
        r = check_split(split)
        print_split_report(r)
        all_results.append(r)
        for cid, cnt in r["class_counts"].items():
            total_class_counts[cid] += cnt

        has_error = (
            r["missing_label"] or r["missing_image"] or
            r["bad_format"] or r["invalid_class"] or r["out_of_range_coords"]
        )
        if has_error:
            overall_ok = False

    # Genel toplam
    section("3. GENEL ÖZET")
    total_images = sum(r["images"] for r in all_results)
    total_labels = sum(r["labels"] for r in all_results)
    log(f"  Toplam görsel  : {total_images:,}")
    log(f"  Toplam label   : {total_labels:,}")
    log(f"  Toplam nesne (tüm split): {sum(total_class_counts.values()):,}")
    log(f"\n  Tüm split sınıf toplamları:")
    for cid in sorted(total_class_counts):
        log(f"    [{cid}] {CLASS_NAMES.get(cid,'?'):12s}: {total_class_counts[cid]:,}")

    log(f"\n{'─'*60}")
    if overall_ok:
        log("  SONUÇ: Dataset EĞİTİME HAZIR ✓")
    else:
        log("  SONUÇ: Hatalar tespit edildi, eğitim öncesi düzeltme önerilir ✗")
    log(f"{'─'*60}")

    # Raporu kaydet
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_output))
    log(f"\nRapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
