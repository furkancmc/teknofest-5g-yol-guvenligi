"""
monitor_training.py
--------------------
Eğitim sırasında results.csv'yi okuyarak anlık ilerlemeyi takip eder.
Ayrı bir terminalde çalıştırın:

  python monitor_training.py

Her 30 saniyede bir güncellenir. Ctrl+C ile durdurulabilir.
"""

import time
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR    = Path(__file__).parent
RESULTS_CSV = BASE_DIR / "runs" / "vehicle_final" / "yolov8m_final_run" / "results.csv"


def read_results(csv_path: Path):
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def display(rows):
    if not rows:
        return
    last = rows[-1]

    # Hangi keylerin mevcut olduğunu bul (ultralytics sürüme göre değişebilir)
    epoch       = last.get("epoch", "?")
    box_loss    = last.get("train/box_loss", last.get("box_loss", "?"))
    cls_loss    = last.get("train/cls_loss", last.get("cls_loss", "?"))
    dfl_loss    = last.get("train/dfl_loss", last.get("dfl_loss", "?"))
    map50       = last.get("metrics/mAP50(B)", last.get("mAP50", "?"))
    map50_95    = last.get("metrics/mAP50-95(B)", last.get("mAP50-95", "?"))
    precision   = last.get("metrics/precision(B)", last.get("precision", "?"))
    recall      = last.get("metrics/recall(B)", last.get("recall", "?"))

    print(f"\n{'─'*60}")
    print(f"  Güncelleme: {datetime.now().strftime('%H:%M:%S')}  |  Epoch: {epoch}")
    print(f"{'─'*60}")
    print(f"  box_loss    : {box_loss}")
    print(f"  cls_loss    : {cls_loss}")
    print(f"  dfl_loss    : {dfl_loss}")
    print(f"  mAP50       : {map50}")
    print(f"  mAP50-95    : {map50_95}")
    print(f"  Precision   : {precision}")
    print(f"  Recall      : {recall}")

    # Genel ilerleme (son 5 epoch)
    if len(rows) >= 2:
        print(f"\n  Son epoch trendi (box_loss):")
        for r in rows[-5:]:
            ep  = r.get("epoch", "?")
            bl  = r.get("train/box_loss", r.get("box_loss", "?"))
            m50 = r.get("metrics/mAP50(B)", r.get("mAP50", "?"))
            print(f"    Epoch {ep:>3}: box_loss={bl:>8}  mAP50={m50:>8}")


def main():
    print("YOLOv8 Eğitim Monitörü başlatıldı.")
    print(f"Takip edilen dosya: {RESULTS_CSV}")
    print("Ctrl+C ile durdurun.\n")

    prev_epoch = None
    try:
        while True:
            rows = read_results(RESULTS_CSV)
            if rows is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Eğitim henüz başlamadı veya results.csv bulunamadı...")
            else:
                last_epoch = rows[-1].get("epoch", None)
                if last_epoch != prev_epoch:
                    display(rows)
                    prev_epoch = last_epoch
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Yeni epoch yok, bekleniyor...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitör durduruldu.")


if __name__ == "__main__":
    main()
