"""
fix_dataset.py
---------------
Dataset düzeltme scripti:
1. OBB / hatalı formatlı satırları temizle (5 değer olmayanları sil)
2. Test setinde van (class 4) yoksa valid'den rastgele örnekle ekle
3. Temizlendikten sonra hiç nesne kalmayan dosyaları rejected'a taşı
4. Rapor üret
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "merged_vehicle_dataset"
LOGS_DIR    = BASE_DIR / "training_vehicle_model" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LOGS_DIR / "fix_report.txt"

REJECTED_DIR = DATASET_DIR / "rejected"
REJECTED_DIR.mkdir(exist_ok=True)

CLASS_NAMES  = {0: "car", 1: "motorcycle", 2: "truck", 3: "bus", 4: "van"}
VALID_IDS    = set(CLASS_NAMES.keys())
SPLITS       = ["train", "valid", "test"]

lines_out = []

def log(msg=""):
    print(msg)
    lines_out.append(str(msg))

# ─────────────────────────────────────────────────────────────────
# 1. OBB / Hatalı Format Temizleme
# ─────────────────────────────────────────────────────────────────
def fix_bad_format_lines():
    log("\n[1] Hatalı format satırları temizleniyor...")
    fixed_files  = 0
    removed_rows = 0
    rejected     = 0

    for split in SPLITS:
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir = DATASET_DIR / "images" / split
        if not lbl_dir.exists():
            continue

        for lbl_path in lbl_dir.glob("*.txt"):
            with open(lbl_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()

            good_lines = []
            removed    = 0
            for line in raw_lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                if len(parts) != 5:
                    log(f"  [KALDIRILDI] {lbl_path.name} — '{line.strip()[:80]}'")
                    removed += 1
                    removed_rows += 1
                    continue
                # class id kontrolü
                try:
                    cls_id = int(parts[0])
                    if cls_id not in VALID_IDS:
                        log(f"  [KALDIRILDI] {lbl_path.name} — geçersiz class={cls_id}")
                        removed += 1
                        removed_rows += 1
                        continue
                    float(parts[1]); float(parts[2]); float(parts[3]); float(parts[4])
                except ValueError:
                    log(f"  [KALDIRILDI] {lbl_path.name} — değer hatası: '{line.strip()[:60]}'")
                    removed += 1
                    removed_rows += 1
                    continue
                good_lines.append(line)

            if removed > 0:
                fixed_files += 1
                if len(good_lines) == 0:
                    # Tüm satırlar silinmiş → rejected
                    stem = lbl_path.stem
                    img_files = list(img_dir.glob(f"{stem}.*"))
                    if img_files:
                        shutil.move(str(img_files[0]), REJECTED_DIR / img_files[0].name)
                    shutil.move(str(lbl_path), REJECTED_DIR / lbl_path.name)
                    log(f"  [REJECTED] {lbl_path.name} — tüm satırlar silindi, dosya reddedildi")
                    rejected += 1
                else:
                    with open(lbl_path, "w", encoding="utf-8") as f:
                        f.writelines(good_lines)

    log(f"\n  Toplam düzeltilen dosya : {fixed_files}")
    log(f"  Toplam kaldırılan satır  : {removed_rows}")
    log(f"  Rejected'a taşınan dosya : {rejected}")

# ─────────────────────────────────────────────────────────────────
# 2. Test setinde van kontrolü ve yeniden örnekleme
# ─────────────────────────────────────────────────────────────────
def fix_test_van():
    log("\n[2] Test setindeki van (class 4) durumu kontrol ediliyor...")

    test_lbl_dir = DATASET_DIR / "labels" / "test"
    test_img_dir = DATASET_DIR / "images" / "test"

    # Test'te van var mı?
    van_count = 0
    for lbl_path in test_lbl_dir.glob("*.txt"):
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5 and int(parts[0]) == 4:
                    van_count += 1

    log(f"  Test setinde mevcut van nesnesi sayısı: {van_count}")

    if van_count > 0:
        log("  Test setinde zaten van var, işlem gerekmez.")
        return

    # Valid'den van içeren dosyaları bul
    log("  Test setinde van YOK. Valid'den van içeren örnekler test'e taşınıyor...")

    valid_lbl_dir = DATASET_DIR / "labels" / "valid"
    valid_img_dir = DATASET_DIR / "images" / "valid"

    van_candidates = []
    for lbl_path in valid_lbl_dir.glob("*.txt"):
        with open(lbl_path, "r", encoding="utf-8") as f:
            content = f.read()
        # En az bir van içeren dosyaları bul
        for line in content.strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5 and int(parts[0]) == 4:
                van_candidates.append(lbl_path)
                break

    log(f"  Valid'de van içeren {len(van_candidates)} dosya bulundu.")

    # Maksimum 200 tane örnekle (ya da hepsini)
    sample_size = min(200, len(van_candidates))
    selected    = random.sample(van_candidates, sample_size)

    moved = 0
    for lbl_path in selected:
        stem      = lbl_path.stem
        img_files = list(valid_img_dir.glob(f"{stem}.*"))
        if not img_files:
            continue
        img_path = img_files[0]

        dest_lbl = test_lbl_dir / lbl_path.name
        dest_img = test_img_dir / img_path.name

        if not dest_lbl.exists() and not dest_img.exists():
            shutil.move(str(lbl_path), dest_lbl)
            shutil.move(str(img_path), dest_img)
            moved += 1

    log(f"  {moved} dosya valid'den test'e taşındı.")

# ─────────────────────────────────────────────────────────────────
# 3. Son istatistik
# ─────────────────────────────────────────────────────────────────
def final_stats():
    log("\n[3] Final istatistikler:")
    log(f"{'─'*50}")
    total_objs = defaultdict(int)

    for split in SPLITS:
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir = DATASET_DIR / "images" / split
        if not lbl_dir.exists():
            continue

        n_imgs = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt")))
        cls_cnt = defaultdict(int)

        for lbl_path in lbl_dir.glob("*.txt"):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            c = int(parts[0])
                            cls_cnt[c] += 1
                            total_objs[c] += 1
                        except ValueError:
                            pass

        match = "✓" if n_imgs == n_lbls else "✗"
        log(f"\n  [{split.upper()}] Görsel: {n_imgs:,}  Label: {n_lbls:,}  {match}")
        for cid in sorted(cls_cnt):
            log(f"    [{cid}] {CLASS_NAMES.get(cid,'?'):12s}: {cls_cnt[cid]:,}")

    log(f"\n  Tüm split genel toplam:")
    for cid in sorted(total_objs):
        log(f"    [{cid}] {CLASS_NAMES.get(cid,'?'):12s}: {total_objs[cid]:,}")

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    log(f"Fix Dataset — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Dataset: {DATASET_DIR}")

    fix_bad_format_lines()
    fix_test_van()
    final_stats()

    log(f"\n{'='*50}")
    log("Düzeltme tamamlandı.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    log(f"Rapor: {REPORT_PATH}")


if __name__ == "__main__":
    main()
