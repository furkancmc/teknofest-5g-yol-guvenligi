"""
predict_test_samples.py
------------------------
Test setinden rastgele 50 görsel seçerek best.pt ile tahmin yapar.
Sonuçları runs/predictions_sample/ klasörüne kaydeder.

Kullanım:
  python predict_test_samples.py
"""

import random
import shutil
from pathlib import Path

BASE_DIR     = Path(__file__).parent
DATASET_DIR  = BASE_DIR.parent / "merged_vehicle_dataset"
TEST_IMG_DIR = DATASET_DIR / "images" / "test"
BEST_PT      = BASE_DIR / "runs" / "vehicle_final" / "yolov8m_final_run" / "weights" / "best.pt"
OUTPUT_DIR   = BASE_DIR / "runs" / "predictions_sample"
SAMPLE_COUNT = 50


def main():
    from ultralytics import YOLO

    if not BEST_PT.exists():
        print(f"[HATA] best.pt bulunamadı: {BEST_PT}")
        print("Önce eğitimi tamamlayın: python train_vehicle.py")
        return

    if not TEST_IMG_DIR.exists():
        print(f"[HATA] Test klasörü bulunamadı: {TEST_IMG_DIR}")
        return

    # Tüm test görsellerini listele
    all_imgs = [f for f in TEST_IMG_DIR.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    if not all_imgs:
        print("Test klasörü boş!")
        return

    # Rastgele örnekle
    sample_count = min(SAMPLE_COUNT, len(all_imgs))
    selected = random.sample(all_imgs, sample_count)
    print(f"Test setinden {sample_count} görsel seçildi.")

    # Geçici klasör: model liste yerine klasör alıyor
    tmp_dir = BASE_DIR / "runs" / "_tmp_predict_input"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for img in selected:
        shutil.copy2(img, tmp_dir / img.name)

    # Tahmin yap
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BEST_PT))
    results = model.predict(
        source=str(tmp_dir),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(BASE_DIR / "runs"),
        name="predictions_sample",
        exist_ok=True,
    )

    # Geçici klasörü temizle
    shutil.rmtree(tmp_dir)

    print(f"\nTahminler tamamlandı!")
    print(f"Sonuçlar: {BASE_DIR / 'runs' / 'predictions_sample'}")
    print(f"Toplam görsel: {len(results)}")


if __name__ == "__main__":
    main()
