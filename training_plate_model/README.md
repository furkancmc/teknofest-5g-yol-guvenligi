# 🔢 Plaka Tespit Modeli — Eğitim

Bu klasör, **YOLOv8s** tabanlı bağımsız plaka tespit modelinin eğitim betiğini içerir. Proje geneli için [ana README](../README.md)'ye bakın.

## Model

| Parametre | Değer |
|---|---|
| Mimari | YOLOv8s |
| Sınıf | 1 (`license_plate`) |
| Veri | 7.941 görsel (licenseplate2 + Vehicle Registration Plates.v2) |
| Çözünürlük / epoch / batch | 640×640 / 100 / 20 (patience 15) |
| Optimizer | AdamW, lr0=0,001, cosine, weight decay 0,0005, warmup 3 |
| Plakaya özel artırma | degrees=15, scale=0,6, perspective=0,001, flipud kapalı |
| Ağırlık | `runs/plate/yolov8s_plate_run/weights/best.pt` |
| Sonuç | mAP@50 = 0,994, Precision/Recall ≈ 0,99 |

## Dataset yapısı

`plate_merger.py` çıktısı (kök dizinde `merged_plate_dataset/`):

```
merged_plate_dataset/
├── data.yaml
├── images/{train,valid,test}/
└── labels/{train,valid,test}/   ← YOLO formatı: sınıf cx cy w h (normalize)
```

Ham datasetler kök dizindeki `plate/` klasörüne, orijinal Roboflow klasör adlarıyla konulmalıdır.

## Adımlar

```bash
python ../plate_merger.py        # plaka datasetlerini birleştir
python train_plate.py            # eğitimi başlat
python train_plate.py --resume   # kaldığı yerden devam
```

## Notlar

- Plaka tespiti araç takibinden **bağımsızdır**: pipeline'da her karede ayrı `predict()` çağrısı yapılır.
- Plaka tespiti için hafif bir model yeterli olduğundan YOLOv8s seçilmiştir; toplam işlem hızı artar.
- Pipeline'daki güven eşiği 0,33, IoU eşiği 0,45'tir.
- `OOM` alırsanız `batch` değerini 16'ya düşürün.
