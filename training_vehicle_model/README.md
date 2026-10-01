# 🚗 Araç Tespit Modeli — Eğitim

Bu klasör, **YOLOv8m** tabanlı final araç tespit modelinin (ve ona giden ön denemelerin) eğitim betiklerini içerir. Proje geneli için [ana README](../README.md)'ye bakın.

## Model

| Parametre | Değer |
|---|---|
| Mimari | YOLOv8m (COCO ön eğitimli `yolov8m.pt`) |
| Sınıflar | `car`, `motorcycle`, `truck`, `bus` (van → car birleştirildi) |
| Çözünürlük / epoch / batch | 640×640 / 50 / 12 |
| Optimizer | AdamW, lr0=0,001, cosine, weight decay 0,0005, warmup 3 |
| Ağırlık | `runs/vehicle_final/yolov8m_final_run/weights/best.pt` |

## Dataset yapısı

`vehicle_merger.py` çıktısı (kök dizinde `merged_vehicle_dataset/`):

```
merged_vehicle_dataset/
├── data.yaml
├── images/{train,valid,test}/
├── labels/{train,valid,test}/   ← YOLO formatı: sınıf cx cy w h (normalize)
└── rejected/                    ← geçersiz etiketli görseller
```

Ham datasetler kök dizindeki `vehicle/` klasörüne, orijinal Roboflow klasör adlarıyla konulmalıdır (adlar `vehicle_merger.py` içindeki `mappings` ile eşleşir).

## Adımlar

```bash
python ../vehicle_merger.py        # datasetleri birleştir, sınıfları eşle
python check_dataset.py            # etiket/format kontrolü → logs/dataset_check_report.txt
python fix_dataset.py              # bozuk satırları temizle → logs/fix_report.txt
python ../verify_dataset.py        # son doğrulama

python final_train_vehicle.py            # FİNAL: van→car + YOLOv8m, dengeli ~23k alt küme
python final_train_vehicle.py --full     # tüm eğitim seti (47.380 görsel), çok daha uzun sürer
python final_train_vehicle.py --resume   # kaldığı yerden devam

python monitor_training.py         # ayrı terminalde: results.csv'den canlı takip
python predict_test_samples.py     # örnek test görselleri üzerinde tahmin
```

## Betikler

| Betik | Açıklama |
|---|---|
| `final_train_vehicle.py` | **Final eğitim.** Van→car birleştirme, 4 sınıflı dengeli alt küme, YOLOv8m, Copy-Paste/MixUp. |
| `fast_train_vehicle.py` | YOLOv8s ile hızlı deneme (sınıf dengeli alt küme). Deney tablosundaki küçük model denemeleri. |
| `train_vehicle.py` | İlk YOLOv8s eğitimi (batch 8, 50 epoch). Tarihsel referans. |
| `check_dataset.py` | Dataset doğrulama ve sınıf istatistikleri. |
| `fix_dataset.py` | Hatalı formatlı satırları temizler, boşalan dosyaları `rejected/` altına taşır. |
| `monitor_training.py` | Eğitim ilerlemesini `results.csv` üzerinden izler. |
| `predict_test_samples.py` | `best.pt` ile örnek tahminler üretir (`runs/predictions_sample/`). |

## Notlar

- GPU belleği yetmezse `final_train_vehicle.py` içinde `batch` değerini 8'e düşürün.
- Eğitim çıktıları (`runs/`, `logs/`, `subset*/`) ve ağırlıklar `.gitignore` ile depo dışında tutulur.
- Sonuçlar, deney tablosu ve grafikler için ana README'nin [Sonuçlar](../README.md#-sonuçlar) bölümüne bakın.
