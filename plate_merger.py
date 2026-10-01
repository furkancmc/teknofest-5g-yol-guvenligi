import os
import shutil
from pathlib import Path
from collections import defaultdict

# Veri seti yolu
_ROOT = Path(__file__).parent
base_dir   = _ROOT / "plate"
output_dir = _ROOT / "merged_plate_dataset"

# Yeni Sınıf İsimleri
target_names = {
    0: "license_plate"
}

# Mapping: dataset_name -> {old_class_id: new_class_id}
# -1 means DROP
mappings = {
    "Vehicle Registration Plates.v2-licenseplatedatasetv1.yolov8": {
        0: 0  # License_Plate -> license_plate
    },
    "licenseplate2.v1i.yolov8": {
        0: 0  # LicensePlate -> license_plate
    }
}

# Çıktı klasörlerini oluştur
splits = ["train", "valid", "test"]
for split in splits:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
(output_dir / "rejected").mkdir(parents=True, exist_ok=True)

# İstatistikler
stats = {
    "total_images": 0,
    "total_labels": 0,
    "rejected_images": 0,
    "class_counts": defaultdict(int),
    "dataset_image_counts": defaultdict(int)
}

print("Plaka veri setleri birleştiriliyor...")

for dataset_path in base_dir.iterdir():
    if not dataset_path.is_dir() or dataset_path.name not in mappings:
        continue

    dataset_name = dataset_path.name
    dataset_mapping = mappings[dataset_name]

    print(f"İşleniyor: {dataset_name}")

    # Her split (train, valid, val, test) için dön
    for old_split_name in ["train", "valid", "val", "test"]:
        split_path = dataset_path / old_split_name
        if not split_path.exists() or not split_path.is_dir():
            continue

        labels_dir = split_path / "labels"
        images_dir = split_path / "images"

        if not labels_dir.exists() or not images_dir.exists():
            continue

        # validasyon klasörünü standartlaştır
        new_split_name = "valid" if old_split_name == "val" else old_split_name

        if new_split_name not in splits:
            continue

        # Label txt dosyalarını oku
        for label_file in labels_dir.glob("*.txt"):
            stem = label_file.stem

            # Eşleşen görseli bul (jpg, png, jpeg olabilir)
            img_files = list(images_dir.glob(f"{stem}.*"))
            if not img_files:
                continue

            img_file = img_files[0]

            # Label dosyasını oku ve dönüştür
            new_lines = []
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    old_cls_id = int(parts[0])
                    if old_cls_id in dataset_mapping:
                        new_cls_id = dataset_mapping[old_cls_id]
                        if new_cls_id != -1:
                            new_line = f"{new_cls_id} " + " ".join(parts[1:]) + "\n"
                            new_lines.append(new_line)
                            stats["class_counts"][target_names[new_cls_id]] += 1
                            stats["total_labels"] += 1

            new_stem = f"{dataset_name}_{stem}"
            new_label_name = f"{new_stem}.txt"
            new_img_name = f"{new_stem}{img_file.suffix}"

            # Eğer yeni nesne yoksa rejected klasörüne gönder
            if len(new_lines) == 0:
                shutil.copy2(img_file, output_dir / "rejected" / new_img_name)
                shutil.copy2(label_file, output_dir / "rejected" / new_label_name)
                stats["rejected_images"] += 1
            else:
                # Görseli kopyala
                shutil.copy2(img_file, output_dir / "images" / new_split_name / new_img_name)

                # Yeni label txt dosyasını yaz
                with open(output_dir / "labels" / new_split_name / new_label_name, "w") as f:
                    f.writelines(new_lines)

                stats["total_images"] += 1
                stats["dataset_image_counts"][dataset_name] += 1

# Yeni data.yaml oluştur
yaml_content = """path: merged_plate_dataset
train: images/train
val: images/valid
test: images/test

nc: 1
names:
  0: license_plate
"""

with open(output_dir / "data.yaml", "w") as f:
    f.write(yaml_content)

print("\n--- İŞLEM RAPORU ---")
print(f"Toplam görsel sayısı (Kabul edilen): {stats['total_images']}")
print(f"Toplam etiket (nesne) sayısı: {stats['total_labels']}")
print(f"Reddedilen (hiç nesne kalmayan) görsel sayısı: {stats['rejected_images']}\n")

print("Sınıf bazlı nesne sayıları:")
for cls_name, count in sorted(stats['class_counts'].items()):
    print(f"- {cls_name}: {count}")

print("\nDataset bazlı alınan görsel sayıları:")
for ds_name, count in sorted(stats['dataset_image_counts'].items()):
    print(f"- {ds_name}: {count}")

print("\nVeri seti başarıyla oluşturuldu: merged_plate_dataset")

# Raporu bir txt dosyasına da yazdıralım
with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
    f.write("--- İŞLEM RAPORU ---\n")
    f.write(f"Toplam görsel sayısı (Kabul edilen): {stats['total_images']}\n")
    f.write(f"Toplam etiket (nesne) sayısı: {stats['total_labels']}\n")
    f.write(f"Reddedilen (hiç nesne kalmayan) görsel sayısı: {stats['rejected_images']}\n\n")

    f.write("Sınıf bazlı nesne sayıları:\n")
    for cls_name, count in sorted(stats['class_counts'].items()):
        f.write(f"- {cls_name}: {count}\n")

    f.write("\nDataset bazlı alınan görsel sayıları:\n")
    for ds_name, count in sorted(stats['dataset_image_counts'].items()):
        f.write(f"- {ds_name}: {count}\n")

    f.write("\nEski Sınıfların Yeni Sınıflara Eşlenme Tablosu:\n")
    f.write("Vehicle Registration Plates dataset: License_Plate (0) -> license_plate (0)\n")
    f.write("licenseplate2 dataset: LicensePlate (0) -> license_plate (0)\n")
