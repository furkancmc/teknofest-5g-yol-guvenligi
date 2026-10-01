import os
from pathlib import Path

def verify_dataset(dataset_path):
    print(f"Veri Seti Kontrol Ediliyor: {dataset_path}")
    print("-" * 50)
    
    splits = ["train", "valid", "test"]
    all_good = True
    
    for split in splits:
        images_dir = Path(dataset_path) / "images" / split
        labels_dir = Path(dataset_path) / "labels" / split
        
        # Eğer klasörler yoksa uyar
        if not images_dir.exists():
            print(f"UYARI: {images_dir} bulunamadı!")
            continue
        if not labels_dir.exists():
            print(f"UYARI: {labels_dir} bulunamadı!")
            continue
            
        # Dosya sayılarını al
        images_count = len([f for f in images_dir.iterdir() if f.is_file()])
        labels_count = len([f for f in labels_dir.iterdir() if f.is_file()])
        
        print(f"[{split.upper()}]")
        print(f"Images sayısı: {images_count}")
        print(f"Labels sayısı: {labels_count}")
        
        if images_count == labels_count:
            print(f"-> DURUM: BAŞARILI (Eşleşiyor)\n")
        else:
            print(f"-> DURUM: HATA! (Sayılar eşleşmiyor! Fark: {abs(images_count - labels_count)})\n")
            all_good = False
            
            # Eksik olanları bul (opsiyonel hata ayıklama)
            img_stems = set(f.stem for f in images_dir.iterdir() if f.is_file())
            lbl_stems = set(f.stem for f in labels_dir.iterdir() if f.is_file())
            
            missing_labels = img_stems - lbl_stems
            missing_images = lbl_stems - img_stems
            
            if missing_labels:
                print(f"   Label'ı eksik olan ilk 5 görsel: {list(missing_labels)[:5]}")
            if missing_images:
                print(f"   Görseli eksik olan ilk 5 label: {list(missing_images)[:5]}")
                
    print("-" * 50)
    if all_good:
        print("SONUÇ: Tüm klasörlerde image ve label sayıları BİREBİR EŞLEŞİYOR! ✓")
    else:
        print("SONUÇ: Eşleşmeyen klasörler tespit edildi. Lütfen yukarıdaki detayları inceleyin. ✗")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "merged_vehicle_dataset"),
    )
    args = parser.parse_args()
    verify_dataset(args.dataset)
