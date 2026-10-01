"""
5G Akıllı Trafik İzleme — GENEL Pipeline
=========================================
Kamera açısından bağımsız (MOBESE / kavşak / dashcam / garaj) çalışacak
şekilde tasarlanmıştır. Hiçbir geometri test videolarına sabitlenmemiştir;
ölçek (piksel→metre) her videoda araç boyutlarından otomatik tahmin edilir.

Pipeline:
  Video
   → YOLOv8m Araç + Plaka Tespiti
   → NMS / Confidence Filtreleme
   → ByteTrack (ID atama / takip)
   → Kalman Filtre (piksel düzleminde düzeltme — 8 boyutlu: cx,cy,w,h,vx,vy,vw,vh)
   → Polygon ROI (yol bölgesi filtresi, adaptif)
   → Otomatik Ölçek / Homografi (araç boyutundan metre)
   → Optical Flow (hareket yönü)
   → Hız Tahmini (+ Moving Average Filter)
   → Araçlar Arası Mesafe
   → Time-to-Collision (TTC)
   → Şerit İhlali (hibrit: boyalı şerit veya rota sapması)
   → Risk Skoru (çok faktörlü, çarpan etkili)
   → 5G Uyarı Çıktısı (kalıcı durum + banner + soğuma)
   → Event Logging (zaman damgalı olay kaydı + risk + latency)

Kullanım:
  python traffic_pipeline.py                      # teknofest_test/ tüm videolar
  python traffic_pipeline.py --input yol/video.mp4
  python traffic_pipeline.py --input klasor/ --output cikti/
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════
# KONFİGÜRASYON  (geometri DEĞİL — sadece evrensel sabitler)
# ═══════════════════════════════════════════════════════════════
BASE = Path(__file__).parent

VEHICLE_MODEL_PATH = BASE / "training_vehicle_model/runs/vehicle_final/yolov8m_final_run/weights/best.pt"
PLATE_MODEL_PATH   = BASE / "training_plate_model/runs/plate/yolov8s_plate_run/weights/best.pt"
DEFAULT_INPUT      = BASE / "teknofest_test"
DEFAULT_OUTPUT     = BASE / "results_pipeline"

CONF_VEHICLE = 0.28
CONF_PLATE   = 0.33
IOU_THRESH   = 0.45
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PROC_W, PROC_H = 1280, 720      # işlem çözünürlüğü

# Gerçek dünya araç boyutları (metre) — ölçek otomatik tahmini için referans
REAL_WIDTH_M = {
    "car": 1.82, "motorcycle": 0.80, "truck": 2.50, "bus": 2.55,
}
REAL_HEIGHT_M = {
    "car": 1.50, "motorcycle": 1.10, "truck": 3.20, "bus": 3.10,
}
DEFAULT_REAL_WIDTH = 1.82
DEFAULT_REAL_HEIGHT = 1.50

# Makul ölçek sınırları (metre / piksel) — saçma tahminleri kırpar
SCALE_MIN, SCALE_MAX = 0.01, 0.15

# Hız sınırları araç türüne göre (km/h)
MAX_SPEED_BY_CLASS = {
    "car": 250.0, "motorcycle": 200.0, "truck": 120.0, "bus": 110.0,
}

# Risk / uyarı eşikleri
SPEED_LIMIT_KMH   = 90.0    # genel referans hız sınırı
DIST_CRITICAL_M   = 8.0     # kritik mesafe (metre)
DIST_WARNING_M    = 18.0    # uyarı mesafesi (metre)
TTC_CRITICAL_S    = 2.0     # kritik TTC (saniye)
TTC_WARNING_S     = 5.0     # uyarı TTC (saniye)
MIN_MOVE_SPEED    = 5.0     # km/h — altında "durağan" sayılır
RISK_WARN_THRESH  = 40.0    # orta risk eşiği
RISK_5G_THRESH    = 60.0    # 5G uyarı eşiği
RISK_CRIT_THRESH  = 80.0    # kritik risk eşiği

# Track sürekliliği
KALMAN_MAX_AGE     = 15     # ölçümsüz kaç kare track yaşar
TRACK_MERGE_DIST   = 60.0   # piksel — ID birleştirme eşiği

# Renk paleti (BGR)
C = {
    "car": (90, 200, 90), "motorcycle": (60, 160, 255),
    "truck": (40, 120, 255), "bus": (200, 80, 255),
    "license_plate": (0, 225, 225), "id_tag": (255, 205, 40),
    "arrow": (255, 230, 80), "arrow_viol": (40, 60, 255),
    "dist_ok": (0, 220, 220), "dist_warn": (40, 140, 255), "dist_crit": (30, 30, 255),
    "lane": (90, 230, 90), "viol": (30, 30, 255), "roi": (180, 180, 60),
    "risk_low": (90, 210, 90), "risk_med": (40, 200, 240),
    "risk_high": (30, 110, 255), "risk_crit": (20, 20, 230),
    "white": (255, 255, 255), "black": (0, 0, 0),
    "panel": (28, 28, 28), "ok_green": (70, 200, 70),
    "warn_orange": (30, 170, 250), "crit_red": (25, 25, 235),
}


# ═══════════════════════════════════════════════════════════════
# Moving Average Filter — kayan pencere ortalaması
# ═══════════════════════════════════════════════════════════════
class MovingAverage:
    def __init__(self, window: int = 7):
        self.window = window
        self.buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, key: int, value: float) -> float:
        self.buffers[key].append(value)
        return float(np.mean(self.buffers[key]))

    def clear(self, key: int):
        self.buffers.pop(key, None)


# ═══════════════════════════════════════════════════════════════
# Kalman Filtre — 8 boyutlu state: [cx, cy, w, h, vx, vy, vw, vh]
#   - Hem konum hem kutu boyutu takip edilir → bbox titremesi ortadan kalkar
#   - Sabit hız (constant velocity) modeli
#   - Joseph form kovaryans güncellemesi (sayısal kararlılık)
# ═══════════════════════════════════════════════════════════════
class KalmanBox:
    """8-state Kalman filtresi: [cx, cy, w, h, vx, vy, vw, vh]
    Konum ve kutu boyutlarını birlikte düzleştirir.
    """
    def __init__(self, cx: float, cy: float, w: float, h: float):
        # State transition matrix (8x8) — sabit hız modeli
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # w  += vw
        self.F[3, 7] = 1.0  # h  += vh

        # Observation matrix (4x8) — ölçüm: [cx, cy, w, h]
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Proses gürültüsü: konum düşük, boyut çok düşük, hız orta
        # Bu değerler kutu titremesini bastırırken gerçek hareketi yakalayacak
        self.Q = np.diag([
            1.0,   # cx proses gürültüsü
            1.0,   # cy
            0.5,   # w  (boyut yavaş değişir)
            0.5,   # h
            4.0,   # vx (hız daha belirsiz)
            4.0,   # vy
            0.25,  # vw (boyut hızı çok yavaş)
            0.25,  # vh
        ]).astype(np.float64)

        # Ölçüm gürültüsü: algılayıcı (YOLO bbox) hatasını modelliyor
        self.R = np.diag([
            4.0,   # cx ölçüm gürültüsü
            4.0,   # cy
            6.0,   # w  (bbox genişliği daha gürültülü)
            6.0,   # h
        ]).astype(np.float64)

        # Başlangıç kovaryansı
        self.P = np.eye(8, dtype=np.float64)
        self.P[0, 0] = 10.0
        self.P[1, 1] = 10.0
        self.P[2, 2] = 10.0
        self.P[3, 3] = 10.0
        self.P[4, 4] = 100.0   # hız başta belirsiz
        self.P[5, 5] = 100.0
        self.P[6, 6] = 25.0
        self.P[7, 7] = 25.0

        # Başlangıç state
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def predict(self) -> np.ndarray:
        """Bir adım ileri tahmin. Dönen: [cx, cy, w, h]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Boyutların negatif olmasını engelle
        self.x[2] = max(self.x[2], 4.0)
        self.x[3] = max(self.x[3], 4.0)
        return self.x[:4].copy()

    def update(self, cx: float, cy: float, w: float, h: float) -> np.ndarray:
        """Ölçümle güncelle. Joseph form kovaryans güncellemesi. Dönen: [cx, cy, w, h]."""
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x                          # inovasyon
        S = self.H @ self.P @ self.H.T + self.R          # inovasyon kovaryansı
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman kazancı
        self.x = self.x + K @ y

        # Joseph form: (I - KH)P(I - KH)^T + KRK^T — sayısal olarak daha kararlı
        I_KH = np.eye(8, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Simetri düzeltmesi
        self.P = (self.P + self.P.T) / 2.0

        # Boyutların negatif olmasını engelle
        self.x[2] = max(self.x[2], 4.0)
        self.x[3] = max(self.x[3], 4.0)
        return self.x[:4].copy()

    @property
    def velocity(self) -> Tuple[float, float]:
        """Piksel hızı (vx, vy) kare başına."""
        return float(self.x[4]), float(self.x[5])

    @property
    def state_box(self) -> Tuple[float, float, float, float]:
        """Mevcut düzleştirilmiş (cx, cy, w, h)."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])


class KalmanBank:
    """Track ID başına Kalman filtresi yönetir.
    - step(): ölçümle günceller
    - predict_only(): ölçüm gelmeyen karelerde sadece tahmin (iz sürekliliği)
    - age takibi: kaç kare ölçümsüz geçtiğini tutar
    """
    def __init__(self, max_age: int = KALMAN_MAX_AGE):
        self.filters: Dict[int, KalmanBox] = {}
        self.ages: Dict[int, int] = {}           # ölçümsüz kare sayısı
        self.max_age = max_age

    def step(self, tid: int, cx: float, cy: float, w: float, h: float
             ) -> Tuple[float, float, float, float]:
        """Ölçümle güncelle. Döndüren: düzleştirilmiş (cx, cy, w, h)."""
        if tid not in self.filters:
            self.filters[tid] = KalmanBox(cx, cy, w, h)
            self.ages[tid] = 0
            return cx, cy, w, h
        kf = self.filters[tid]
        kf.predict()
        scx, scy, sw, sh = kf.update(cx, cy, w, h)
        self.ages[tid] = 0
        return float(scx), float(scy), float(sw), float(sh)

    def predict_only(self, tid: int) -> Optional[Tuple[float, float, float, float]]:
        """Ölçüm gelmediğinde sadece tahmin. None dönerse track ölmüş."""
        if tid not in self.filters:
            return None
        self.ages[tid] = self.ages.get(tid, 0) + 1
        if self.ages[tid] > self.max_age:
            self.remove(tid)
            return None
        kf = self.filters[tid]
        pred = kf.predict()
        return float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])

    def velocity(self, tid: int) -> Tuple[float, float]:
        """Piksel hızı (vx, vy) kare başına."""
        return self.filters[tid].velocity if tid in self.filters else (0.0, 0.0)

    def has(self, tid: int) -> bool:
        return tid in self.filters

    def remove(self, tid: int):
        self.filters.pop(tid, None)
        self.ages.pop(tid, None)

    def cleanup(self, active_ids: set):
        """Aktif olmayan ve ölmüş track'leri temizle."""
        dead = [tid for tid in self.filters
                if tid not in active_ids and self.ages.get(tid, 0) > self.max_age]
        for tid in dead:
            self.remove(tid)


# ═══════════════════════════════════════════════════════════════
# Perspektif Ölçek Modeli — piksel → metre (kalibrasyonsuz, KARARLI)
#   Ölçeği tekil araç bbox'ından DEĞİL, görüntü satırına (foot_y)
#   bağlı küresel bir modelden okur. Birçok kareden / araçtan toplanan
#   örneklerle her satır bandı için medyan metre/piksel tutulur; böylece
#   bir aracın dönmesi/bbox titremesi mesafeyi bozmaz. Perspektif doğal
#   olarak yakalanır: alt satırlar (yakın) küçük m/px, üst (uzak) büyük.
#
# DÜZELTMELER:
#   - Minimum bbox genişliği 20 px (küçük/titreyen kutular hariç)
#   - IQR tabanlı outlier rejection
#   - Hem genişlik hem yükseklik referansı (anizotropik)
#   - Warm-up koruması: yeterli örnek toplanmadan güvenli varsayılan ölçek
# ═══════════════════════════════════════════════════════════════
class PerspectiveScaleModel:
    def __init__(self, img_h: int, nbins: int = 14, win: int = 300):
        self.h = img_h
        self.nbins = nbins
        self.bin_h = img_h / nbins
        self.bins: List[deque] = [deque(maxlen=win) for _ in range(nbins)]
        self.curve: Optional[np.ndarray] = None     # bin başına yumuşatılmış m/px
        self.all_samples: deque = deque(maxlen=800)
        self._warmup_count = 0
        self._MIN_SAMPLES_FOR_CURVE = 15   # eğri oluşturmak için minimum örnek (daha hızlı warm-up)

    def add_sample(self, foot_y: float, bbox_w_px: float, bbox_h_px: float, label: str):
        """Araç bbox genişliği ve yüksekliğinden ölçek örneği ekle."""
        # Çok küçük kutular güvenilmez — atla
        if bbox_w_px < 20 or bbox_h_px < 20:
            return

        real_w = REAL_WIDTH_M.get(label, DEFAULT_REAL_WIDTH)
        real_h = REAL_HEIGHT_M.get(label, DEFAULT_REAL_HEIGHT)

        # Genişlik ve yükseklikten ayrı ölçek tahminleri
        mpp_w = real_w / bbox_w_px
        mpp_h = real_h / bbox_h_px

        # İki tahminin ağırlıklı ortalaması (genişlik daha güvenilir, perspektiften az etkilenir)
        mpp = 0.65 * mpp_w + 0.35 * mpp_h

        # Fiziksel sınırlar dışındakileri reddet
        if mpp < SCALE_MIN or mpp > SCALE_MAX:
            return

        b = int(np.clip(foot_y // self.bin_h, 0, self.nbins - 1))
        self.bins[b].append(mpp)
        self.all_samples.append(mpp)
        self._warmup_count += 1

    def refresh(self):
        """Bin medyanlarını hesapla, outlier'ları ele, boş binleri doldur, eğriyi yumuşat."""
        if self._warmup_count < self._MIN_SAMPLES_FOR_CURVE:
            self.curve = None
            return

        vals = []
        for b in self.bins:
            if len(b) >= 3:
                arr = np.array(b, dtype=np.float64)
                # IQR tabanlı outlier rejection
                q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
                iqr = q3 - q1
                if iqr > 1e-6:
                    mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
                    cleaned = arr[mask]
                    vals.append(float(np.median(cleaned)) if len(cleaned) > 0 else np.nan)
                else:
                    vals.append(float(np.median(arr)))
            else:
                vals.append(np.nan)

        arr = np.array(vals, dtype=np.float64)
        known = ~np.isnan(arr)
        if not known.any():
            self.curve = None
            return

        idx = np.arange(self.nbins)
        # Boş binleri komşu bilinen değerlerden enterpolasyon/extrapolasyonla doldur
        arr = np.interp(idx, idx[known], arr[known]).astype(np.float64)

        # 3-noktalı yumuşatma (perspektif eğrisini düzleştir)
        if self.nbins >= 3:
            k = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            smoothed = np.convolve(arr, k, mode="same")
            smoothed[0] = 0.25 * smoothed[0] + 0.75 * smoothed[1]      # kenar düzeltmesi
            smoothed[-1] = 0.25 * smoothed[-1] + 0.75 * smoothed[-2]
            arr = smoothed

        # Eğrinin monotonik artışını zorla (perspektif: yukarıda büyük ölçek)
        # Alt satırlar yakın = küçük m/px, üst satırlar uzak = büyük m/px
        # Bu zorlama olmasa da çalışır ama monotonik eğri fiziksel olarak daha doğru
        self.curve = arr

    def scale_at(self, y: float) -> float:
        """Belirli bir y koordinatındaki metre/piksel ölçeği."""
        if self.curve is None:
            return self._global()
        # Bin merkezleri arasında lineer enterpolasyon (sürekli/pürüzsüz)
        centers = (np.arange(self.nbins) + 0.5) * self.bin_h
        return float(np.clip(
            np.interp(np.clip(y, 0, self.h - 1), centers, self.curve),
            SCALE_MIN, SCALE_MAX
        ))

    def between(self, ya: float, yb: float) -> float:
        """İki y koordinatı arasındaki ortalama ölçek."""
        return self.scale_at((ya + yb) / 2.0)

    def _global(self) -> float:
        """Küresel medyan ölçek (warm-up veya fallback)."""
        if len(self.all_samples) < 5:
            return 0.035   # güvenli varsayılan: ~3.5 cm/px (orta mesafe)
        arr = np.array(self.all_samples, dtype=np.float64)
        # IQR ile outlier temizle
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        if iqr > 1e-6:
            mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
            cleaned = arr[mask]
            return float(np.median(cleaned)) if len(cleaned) > 0 else 0.04
        return float(np.median(arr))


# ═══════════════════════════════════════════════════════════════
# Optical Flow — araç kutusu içindeki seyrek LK akışı ile hareket yönü
# ═══════════════════════════════════════════════════════════════
class OpticalFlowDirection:
    LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None

    def set_frame(self, gray: np.ndarray):
        self.prev_gray = gray

    def flow_in_box(self, cur_gray: np.ndarray,
                    box: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        if self.prev_gray is None:
            return None
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(cur_gray.shape[1] - 1, x2)
        y2 = min(cur_gray.shape[0] - 1, y2)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None
        roi_prev = self.prev_gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(roi_prev, maxCorners=25, qualityLevel=0.05,
                                      minDistance=5, blockSize=7)
        if pts is None or len(pts) < 4:
            return None
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, cur_gray, pts, None, **self.LK_PARAMS)
        if nxt is None:
            return None
        st = st.reshape(-1).astype(bool)
        good_old = pts.reshape(-1, 2)[st]
        good_new = nxt.reshape(-1, 2)[st]
        if len(good_new) < 3:
            return None
        flow = good_new - good_old
        # Aykırı değerleri kırp → medyan vektör
        dx = float(np.median(flow[:, 0]))
        dy = float(np.median(flow[:, 1]))
        return dx, dy


# ═══════════════════════════════════════════════════════════════
# Adaptif Polygon ROI — araçların geçtiği bölgeden yol alanı türetir
# ═══════════════════════════════════════════════════════════════
class AdaptiveROI:
    def __init__(self, warmup: int = 35, pad: int = 45):
        self.points: List[Tuple[int, int]] = []
        self.warmup = warmup
        self.pad = pad
        self.hull: Optional[np.ndarray] = None
        self.frame_count = 0

    def add(self, cx: float, cy: float):
        self.points.append((int(cx), int(cy)))

    def rebuild(self, w: int, h: int):
        self.frame_count += 1
        if len(self.points) < 12:
            self.hull = None
            return
        if self.frame_count % 15 != 0 and self.hull is not None:
            return
        pts = np.array(self.points[-4000:], dtype=np.int32)
        hull = cv2.convexHull(pts)
        # Genişlet (dilate) — kenardaki araçları dışlamamak için
        M = cv2.moments(hull)
        if M["m00"] == 0:
            self.hull = hull
            return
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        expanded = []
        for p in hull.reshape(-1, 2):
            vx, vy = p[0] - cx, p[1] - cy
            d = math.hypot(vx, vy) or 1.0
            expanded.append([p[0] + vx / d * self.pad, p[1] + vy / d * self.pad])
        self.hull = np.array(expanded, dtype=np.int32).reshape(-1, 1, 2)

    def contains(self, cx: float, cy: float) -> bool:
        # Gevşek filtre: yalnızca yol bölgesinin BELİRGİN dışındaki (negatif
        # mesafe > MARGIN) araçları eler; gerçek hareketli araçları asla atmaz.
        if self.hull is None or self.frame_count < self.warmup:
            return True
        return cv2.pointPolygonTest(self.hull, (float(cx), float(cy)), True) >= -self.pad

    def draw(self, frame: np.ndarray):
        # Görsel kirliliği önlemek için ROI poligonu ÇİZİLMEZ (yalnızca filtre).
        pass


# ═══════════════════════════════════════════════════════════════
# Hız Tahmini — Kalman piksel hızı × otomatik ölçek + Moving Average
#   DÜZELTMELER:
#   - Araç türüne göre fiziksel hız sınırı
#   - Ani sıçrama koruması (jump rejection)
#   - Minimum hareket eşiği (2 px/kare altı → sıfır hız)
#   - Genişletilmiş moving average penceresi (12 kare)
# ═══════════════════════════════════════════════════════════════
class SpeedEstimator:
    def __init__(self, fps: float):
        self.fps = fps
        self.ma = MovingAverage(window=8)            # daha reaktif pencere (12→8)
        self.prev_speed: Dict[int, float] = {}       # jump rejection için
        self.MIN_PX_MOVE = 1.0                        # piksel/kare minimum hareket (1.5→1.0)
        self.MAX_SPEED_JUMP = 50.0                    # km/h — tek karedeki max sıçrama (35→50)

    def estimate(self, tid: int, px_vx: float, px_vy: float,
                 scale_m_px: float, label: str = "car") -> float:
        """Hız kestirimi. Dönen: km/h."""
        px_per_frame = math.hypot(px_vx, px_vy)

        # Minimum hareket eşiği: gürültü kaynaklı mikro hareketleri sıfırla
        if px_per_frame < self.MIN_PX_MOVE:
            speed_kmh = 0.0
        else:
            m_per_frame = px_per_frame * scale_m_px
            speed_ms = m_per_frame * self.fps
            speed_kmh = speed_ms * 3.6

        # Araç türüne göre fiziksel üst sınır
        max_speed = MAX_SPEED_BY_CLASS.get(label, 250.0)
        speed_kmh = min(speed_kmh, max_speed)

        # Ani sıçrama koruması: önceki hıza göre çok büyük fark varsa bastır
        prev = self.prev_speed.get(tid, speed_kmh)
        if abs(speed_kmh - prev) > self.MAX_SPEED_JUMP and prev > 0:
            # Sıçramayı yumuşat — önceki hıza doğru çek
            speed_kmh = prev + np.sign(speed_kmh - prev) * self.MAX_SPEED_JUMP * 0.3

        speed_kmh = max(0.0, speed_kmh)
        smoothed = self.ma.update(tid, speed_kmh)
        self.prev_speed[tid] = smoothed
        return round(smoothed, 1)

    def clear(self, tid: int):
        self.ma.clear(tid)
        self.prev_speed.pop(tid, None)


# ═══════════════════════════════════════════════════════════════
# Araçlar Arası Mesafe (metre) — Düzeltilmiş
#   - Kalman-düzeltilmiş foot point kullanılır
#   - Minimum mesafe filtresi (overlap/çift tespit eleme)
#   - Sadece anlamlı mesafeler döndürülür
# ═══════════════════════════════════════════════════════════════
class DistanceCalculator:
    MIN_VALID_DIST_M = 0.5     # 0.5 m altı mesafeler fiziksel olarak imkansız

    @staticmethod
    def pairwise(foot_px: Dict[int, Tuple[float, float]],
                 scale_est: "PerspectiveScaleModel") -> Dict[Tuple[int, int], float]:
        """Tüm araç çiftleri arasındaki mesafeyi hesapla (metre)."""
        ids = list(foot_px.keys())
        out: Dict[Tuple[int, int], float] = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ax, ay = foot_px[a]
                bx, by = foot_px[b]
                px_d = math.hypot(ax - bx, ay - by)

                # Çok yakın pikseller (overlap) — çift tespit olabilir, atla
                if px_d < 5:
                    continue

                # Ölçek, iki aracın yer-temas noktalarının orta satırından (perspektif)
                scale = scale_est.between(ay, by)
                dist_m = round(px_d * scale, 2)

                # Fiziksel minimum mesafe filtresi
                if dist_m < DistanceCalculator.MIN_VALID_DIST_M:
                    continue

                out[tuple(sorted((a, b)))] = dist_m
        return out

    @staticmethod
    def nearest(tid: int, dists: Dict[Tuple[int, int], float]) -> Tuple[Optional[int], float]:
        """En yakın aracı ve mesafesini bul."""
        best_id, best_d = None, float("inf")
        for (a, b), d in dists.items():
            other = None
            if a == tid:
                other = b
            elif b == tid:
                other = a
            if other is not None and d < best_d:
                best_d, best_id = d, other
        return best_id, (round(best_d, 2) if best_id is not None else float("inf"))


# ═══════════════════════════════════════════════════════════════
# Time-to-Collision — öndeki araca yaklaşma süresi
#   Yön, aracın kendi hareket vektörüyle belirlenir (açıdan bağımsız).
# ═══════════════════════════════════════════════════════════════
class TTCCalculator:
    def __init__(self, fps: float):
        self.fps = fps
        self.prev_dist: Dict[int, float] = {}      # track → öne mesafe (m)
        self.prev_leader: Dict[int, int] = {}

    def compute(self, tid: int,
                centroids_px: Dict[int, Tuple[float, float]],
                motion: Dict[int, Tuple[float, float]],
                scale_est: "PerspectiveScaleModel") -> Tuple[Optional[int], float]:
        """Hareket yönündeki en yakın öndeki aracı ve TTC'yi (saniye) döndürür."""
        if tid not in centroids_px or tid not in motion:
            return None, float("inf")
        ox, oy = centroids_px[tid]
        mvx, mvy = motion[tid]
        mmag = math.hypot(mvx, mvy)
        if mmag < 0.5:                              # neredeyse durağan → TTC yok (1.0→0.5)
            return None, float("inf")
        ux, uy = mvx / mmag, mvy / mmag             # birim hareket vektörü

        leader, leader_d = None, float("inf")
        for other, (px, py) in centroids_px.items():
            if other == tid:
                continue
            rx, ry = px - ox, py - oy
            forward = rx * ux + ry * uy             # ileri izdüşüm (px)
            lateral = abs(rx * (-uy) + ry * ux)     # yanal sapma (px)
            if forward <= 0:
                continue
            if lateral > forward * 0.7 + 50:        # şerit/koridor dışı — daha geniş koridor
                continue
            if forward < leader_d:
                leader_d, leader = forward, other

        if leader is None:
            self.prev_leader.pop(tid, None)
            return None, float("inf")

        lpx, lpy = centroids_px[leader]
        scale = scale_est.between(oy, lpy)
        dist_m = leader_d * scale

        ttc = float("inf")
        if self.prev_leader.get(tid) == leader and tid in self.prev_dist:
            range_rate = (dist_m - self.prev_dist[tid]) * self.fps   # m/s (+ uzaklaşıyor)
            if range_rate < -0.3:                                    # yaklaşıyor
                ttc = dist_m / (-range_rate)
        self.prev_dist[tid] = dist_m
        self.prev_leader[tid] = leader
        return leader, round(ttc, 2) if ttc != float("inf") else float("inf")


# ═══════════════════════════════════════════════════════════════
# Şerit İhlali — Hibrit
#   (1) Boyalı şerit net tespit edilirse → çizgiye göre
#   (2) Edilemezse → aracın kendi düzgün rotasından yanal sapma
# ═══════════════════════════════════════════════════════════════
class LaneAnalyzer:
    def __init__(self, smooth_n: int = 30):
        self.left_hist = deque(maxlen=smooth_n)  # daha stabil şerit geçmişi (20→30)
        self.right_hist = deque(maxlen=smooth_n)
        self.lane_conf = 0.0                    # 0..1 boyalı şerit güveni
        self.viol_count: Dict[int, int] = defaultdict(int)
        self.trail: Dict[int, deque] = defaultdict(lambda: deque(maxlen=22))
        self.VIOL_THRESH = 12                   # daha az yanlış pozitif (9→12)

    # ---- Boyalı şerit tespiti (varsa) ----------------------------
    def detect_painted(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        roi_y = int(h * 0.55)
        roi = frame[roi_y:h, :]
        rh = h - roi_y
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(gray, 60, 160)

        left_lines, right_lines = [], []
        for x_lo, x_hi, sign, bucket in [
            (0.02, 0.48, -1, left_lines), (0.52, 0.98, +1, right_lines)
        ]:
            mask = np.zeros_like(edges)
            poly = np.array([[
                (int(w * x_lo), rh), (int(w * (x_lo + 0.12)), int(rh * 0.08)),
                (int(w * (x_hi - 0.12)), int(rh * 0.08)), (int(w * x_hi), rh),
            ]], dtype=np.int32)
            cv2.fillPoly(mask, poly, 255)
            ls = cv2.HoughLinesP(cv2.bitwise_and(edges, mask), 1, np.pi / 180,
                                 32, minLineLength=35, maxLineGap=120)  # daha toleranslı Hough
            if ls is None:
                continue
            for l in ls:
                x1, y1, x2, y2 = l[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if sign * slope < 0.2 or abs(slope) > 5.0:  # daha geniş açı aralığı
                    continue
                length = math.hypot(x2 - x1, y2 - y1)
                bucket.append((x1, y1 + roi_y, x2, y2 + roi_y, length))

        left = self._fit(left_lines, h)
        right = self._fit(right_lines, h)
        if left is not None:
            self.left_hist.append(left)
        if right is not None:
            self.right_hist.append(right)

        lf = np.mean(self.left_hist, axis=0) if self.left_hist else None
        rf = np.mean(self.right_hist, axis=0) if self.right_hist else None
        # Güven: her iki çizgi de son karelerde tutarlı bulunduysa yüksek
        have_both = lf is not None and rf is not None
        if have_both and (rf[0] - lf[0]) > 70:        # daha küçük şerit genişliği kabul
            self.lane_conf = min(1.0, self.lane_conf + 0.10)  # daha hızlı güven artışı
        elif lf is not None or rf is not None:         # tek şerit görüldüğünde yavaş artır
            self.lane_conf = min(0.4, self.lane_conf + 0.03)
        else:
            self.lane_conf = max(0.0, self.lane_conf - 0.04)  # daha yavaş düşüş
        return lf, rf

    @staticmethod
    def _fit(lines, h):
        if not lines:
            return None
        xs, ys, ws = [], [], []
        for x1, y1, x2, y2, ln in lines:
            xs += [x1, x2]; ys += [y1, y2]; ws += [ln, ln]
        try:
            a, b = np.polyfit(ys, xs, 1, w=ws)      # x = a*y + b
        except Exception:
            return None
        yb, yt = float(h), float(h * 0.60)
        return np.array([a * yb + b, yb, a * yt + b, yt], dtype=np.float32)

    @staticmethod
    def _x_at(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return None
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

    # ---- İhlal kararı (hibrit) -----------------------------------
    def check(self, tid: int, cx: float, cy: float, speed_kmh: float,
              left, right) -> Tuple[bool, str]:
        if speed_kmh < MIN_MOVE_SPEED:               # durağan → ihlal yok
            self.trail.pop(tid, None)
            self.viol_count[tid] = max(0, self.viol_count[tid] - 2)
            return False, "ok"

        self.trail[tid].append((cx, cy))
        raw_viol = False
        mode = "ok"

        if self.lane_conf >= 0.35 and left is not None and right is not None:  # eşik 0.5→0.35
            lx, rx = self._x_at(left, cy), self._x_at(right, cy)
            if lx is not None and rx is not None and (rx - lx) > 70:
                m = (rx - lx) * 0.09   # biraz daha geniş tolerans marjı
                raw_viol = cx < (lx - m) or cx > (rx + m)
                mode = "painted"
        else:
            # Rota sapması: aracın kendi son rotasına dik mesafe
            raw_viol, mode = self._trajectory_deviation(tid), "trajectory"

        if raw_viol:
            self.viol_count[tid] += 1
        else:
            self.viol_count[tid] = max(0, self.viol_count[tid] - 1)
        return self.viol_count[tid] >= self.VIOL_THRESH, mode

    def _trajectory_deviation(self, tid: int) -> bool:
        tr = self.trail[tid]
        if len(tr) < 12:
            return False
        pts = np.array(tr, dtype=np.float32)
        p0 = pts[0]
        # Baş-kuyruk yönünde beklenen düz rota; son noktanın bu eksene dik sapması
        ref = pts[len(pts) // 2] - p0
        ref_mag = math.hypot(ref[0], ref[1])
        if ref_mag < 12:                             # yeterince ilerlememiş (8→12)
            return False
        ux, uy = ref / ref_mag
        cur = pts[-1] - p0
        lateral = abs(cur[0] * (-uy) + cur[1] * ux)
        forward = cur[0] * ux + cur[1] * uy
        if forward < 15:                             # minimum ilerleme eşiği (10→15)
            return False
        # Yanal sapma ileriye oranla büyükse → savrulma/şerit değiştirme
        # Eşik gevşetildi: 0.35→0.45, lateral>22→30 (virajlarda yanlış alarm azaltıldı)
        return (lateral / forward) > 0.45 and lateral > 30

    def draw(self, frame: np.ndarray, left, right):
        if self.lane_conf >= 0.5 and left is not None and right is not None:
            overlay = frame.copy()
            pts = np.array([
                [int(left[0]), int(left[1])], [int(left[2]), int(left[3])],
                [int(right[2]), int(right[3])], [int(right[0]), int(right[1])],
            ], np.int32)
            cv2.fillPoly(overlay, [pts], (40, 90, 40))
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            for ln in (left, right):
                cv2.line(frame, (int(ln[0]), int(ln[1])),
                         (int(ln[2]), int(ln[3])), C["lane"], 3, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# Risk Skoru — çok faktörlü, çarpan etkili (0–100)
#   5 bileşen: Hız + Yakın Takip + TTC + Şerit İhlali + Yoğunluk
#   Birden fazla risk aynı anda tetiklendiğinde çarpan etkisi uygulanır →
#   gerçekten tehlikeli durumlar kolayca 5G eşiğini (60+) aşar.
# ═══════════════════════════════════════════════════════════════
class RiskScorer:
    @staticmethod
    def _lin(v, lo, hi, inv=False):
        """Lineer normalizasyon [lo, hi] → [0, 100]. inv=True ise ters orantı."""
        if hi == lo:
            return 0.0
        t = (v - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
        return (1 - t) * 100 if inv else t * 100

    def compute(self, speed, dist_m, ttc_s, lane_viol, n_vehicles) -> float:
        """Risk skoru hesapla (0-100). Çarpan etkili."""
        # ── Bireysel risk bileşenleri ──
        # Hız riski: 40 km/h altı → 0, 120+ → 100
        s = self._lin(speed, 40, 120)

        # Yakın takip riski: DIST_CRITICAL_M altı → 100, DIST_WARNING_M üstü → 0
        if dist_m != float("inf") and dist_m < DIST_WARNING_M * 1.5:
            d = self._lin(dist_m, DIST_CRITICAL_M * 0.5, DIST_WARNING_M, inv=True)
        else:
            d = 0.0

        # TTC riski: TTC_CRITICAL_S altı → 100, TTC_WARNING_S üstü → 0
        if ttc_s != float("inf") and ttc_s < TTC_WARNING_S * 1.5:
            t = self._lin(ttc_s, TTC_CRITICAL_S * 0.5, TTC_WARNING_S, inv=True)
        else:
            t = 0.0

        # Şerit ihlali riski
        l = 90.0 if lane_viol else 0.0

        # Yoğunluk riski: araç sayısına göre kademeli
        if n_vehicles <= 1:
            y = 0.0
        elif n_vehicles <= 3:
            y = (n_vehicles - 1) * 15.0
        elif n_vehicles <= 6:
            y = 30.0 + (n_vehicles - 3) * 12.0
        else:
            y = min(100.0, 66.0 + (n_vehicles - 6) * 8.0)

        # ── Ağırlıklı toplam ──
        base_score = 0.25 * s + 0.25 * d + 0.25 * t + 0.15 * l + 0.10 * y

        # ── Çarpan etkisi: birden fazla risk faktörü aktifse skor yükselir ──
        active_factors = 0
        if s > 20:
            active_factors += 1
        if d > 20:
            active_factors += 1
        if t > 20:
            active_factors += 1
        if lane_viol:
            active_factors += 1
        if y > 15:
            active_factors += 1

        if active_factors >= 3:
            multiplier = 1.30
        elif active_factors >= 2:
            multiplier = 1.15
        else:
            multiplier = 1.0

        final_score = min(100.0, base_score * multiplier)
        return round(final_score, 1)

    @staticmethod
    def color(score):
        if score < RISK_WARN_THRESH:
            return C["risk_low"]
        if score < RISK_5G_THRESH:
            return C["risk_med"]
        if score < RISK_CRIT_THRESH:
            return C["risk_high"]
        return C["risk_crit"]

    @staticmethod
    def label(score):
        if score < RISK_WARN_THRESH:
            return "DUSUK"
        if score < RISK_5G_THRESH:
            return "ORTA"
        if score < RISK_CRIT_THRESH:
            return "YUKSEK"
        return "KRITIK"


# ═══════════════════════════════════════════════════════════════
# Event Logging — zaman damgalı olay kaydı (CSV + JSON)
#   - Her event'e risk_score ve risk_level alanları eklenir
#   - 5G gecikme simülasyonu: latency_ms alanı
#   - Sahne özeti: her N karede bir genel durum kaydı
# ═══════════════════════════════════════════════════════════════
class EventLogger:
    def __init__(self, video_name: str):
        self.video = video_name
        self.events: List[Dict] = []
        self.last_seen: Dict[Tuple[int, str], int] = {}   # cooldown
        self.COOLDOWN = 25                                # kare

    def log(self, frame_idx, t_sec, ev_type, tid, detail: Dict,
            risk_score: float = 0.0, risk_level: str = ""):
        """Olay kaydı ekle (cooldown'lu)."""
        key = (tid, ev_type)
        if frame_idx - self.last_seen.get(key, -10**9) < self.COOLDOWN:
            return
        self.last_seen[key] = frame_idx
        self.events.append({
            "video": self.video,
            "frame": frame_idx,
            "time_sec": round(t_sec, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": ev_type,
            "track_id": tid,
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level if risk_level else RiskScorer.label(risk_score),
            "latency_ms": 10,      # 5G URLLC düşük gecikme simülasyonu
            **detail,
        })

    def log_scene(self, frame_idx, t_sec, n_vehicles, avg_speed, max_risk,
                  scene_risk_level, total_alerts):
        """Sahne özet kaydı (periyodik)."""
        self.events.append({
            "video": self.video,
            "frame": frame_idx,
            "time_sec": round(t_sec, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": "SCENE_SUMMARY",
            "track_id": -1,
            "risk_score": round(max_risk, 1),
            "risk_level": scene_risk_level,
            "latency_ms": 10,
            "n_vehicles": n_vehicles,
            "avg_speed_kmh": round(avg_speed, 1),
            "total_alerts": total_alerts,
        })

    def save(self, out_dir: Path, stem: str):
        if not self.events:
            return None, None
        jp = out_dir / f"{stem}_events.json"
        cp = out_dir / f"{stem}_events.csv"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
        keys = sorted({k for e in self.events for k in e.keys()})
        with open(cp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(self.events)
        return jp, cp


# ═══════════════════════════════════════════════════════════════
# 5G Uyarı Modülü — kalıcı durum göstergesi + tetikli banner
#   DÜZELTMELER:
#   - Araç bazlı + sahne bazlı risk değerlendirmesi
#   - Kalıcılık: 3+ kare üst üste eşik aşılırsa tetiklenme
#   - Soğuma süresi: uyarı min 2 saniye aktif kalır
#   - Sahne riski: tüm araçların ağırlıklı ortalaması
# ═══════════════════════════════════════════════════════════════
class AlertModule:
    def __init__(self, fps: float):
        self.fps = fps
        self.active: List[Dict] = []
        self.flash = 0
        self.total_alerts = 0
        # Kalıcılık: araç bazında kaç kare üst üste eşik aşıldı
        self.consecutive: Dict[int, int] = defaultdict(int)
        # Soğuma: uyarı tetiklendikten sonraki minimum aktif kalma karesi
        self.cooldown_until: Dict[int, int] = {}
        self.CONFIRM_FRAMES = 2      # kaç kare üst üste eşik aşılmalı
        self.COOLDOWN_FRAMES = max(int(fps * 2), 30)  # 2 saniye soğuma

    def evaluate(self, frame_idx, risks: Dict[int, float], speeds, ttcs, viols):
        """Her kare çağrılır. Aktif uyarıları belirler."""
        self.active = []

        for tid, score in risks.items():
            if score >= RISK_5G_THRESH:
                self.consecutive[tid] += 1
            else:
                # Soğuma süresi kontrolü
                if tid in self.cooldown_until and frame_idx < self.cooldown_until[tid]:
                    # Hâlâ soğuma süresinde — uyarıyı aktif tut
                    self.active.append({
                        "track_id": tid, "risk": score,
                        "speed": speeds.get(tid, 0.0),
                        "ttc": ttcs.get(tid, float("inf")),
                        "lane": tid in viols,
                        "type": "SOĞUMA",
                    })
                    continue
                self.consecutive[tid] = max(0, self.consecutive[tid] - 1)
                continue

            # Kalıcılık kontrolü: yeterli kare üst üste eşik aşıldı mı?
            if self.consecutive[tid] >= self.CONFIRM_FRAMES:
                alert_type = "KRITIK" if score >= RISK_CRIT_THRESH else "YUKSEK"
                self.active.append({
                    "track_id": tid, "risk": score,
                    "speed": speeds.get(tid, 0.0),
                    "ttc": ttcs.get(tid, float("inf")),
                    "lane": tid in viols,
                    "type": alert_type,
                })
                # Soğuma süresini ayarla
                self.cooldown_until[tid] = frame_idx + self.COOLDOWN_FRAMES

        self.total_alerts += len(self.active)

        # Ölü track'lerin soğuma bilgisini temizle
        dead_keys = [k for k in self.cooldown_until if k not in risks]
        for k in dead_keys:
            self.cooldown_until.pop(k, None)
            self.consecutive.pop(k, None)

    def scene_risk(self, risks: Dict[int, float]) -> float:
        """Sahne bazlı risk: tüm araçların ağırlıklı ortalaması (yüksek skorlar ağır)."""
        if not risks:
            return 0.0
        values = list(risks.values())
        # Ağırlıklı: yüksek skorlar daha fazla etkili (kare ağırlık)
        weights = [max(v, 1.0) ** 1.5 for v in values]
        total_w = sum(weights)
        if total_w < 1e-6:
            return 0.0
        weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_w
        # Sahne riski en az max bireysel riskin %70'i kadar olmalı
        max_risk = max(values)
        return max(weighted_avg, max_risk * 0.7)

    def draw(self, frame, fps, global_max_risk):
        h, w = frame.shape[:2]
        # --- Kalıcı 5G durum pili (her zaman görünür) ---
        if global_max_risk >= RISK_5G_THRESH:
            status, scol = "KRITIK", C["crit_red"]
        elif global_max_risk >= RISK_WARN_THRESH:
            status, scol = "DIKKAT", C["warn_orange"]
        else:
            status, scol = "NORMAL", C["ok_green"]
        pill = f"5G DURUM: {status}  (risk {global_max_risk:.0f})"
        (tw, th), _ = cv2.getTextSize(pill, cv2.FONT_HERSHEY_DUPLEX, 0.62, 2)
        px, py = 10, 12
        cv2.rectangle(frame, (px - 6, py - 4), (px + tw + 10, py + th + 12), C["panel"], -1)
        cv2.rectangle(frame, (px - 6, py - 4), (px + tw + 10, py + th + 12), scol, 2)
        cv2.circle(frame, (px + 4, py + th // 2 + 3), 6, scol, -1)
        cv2.putText(frame, pill, (px + 16, py + th + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, C["white"], 1, cv2.LINE_AA)

        # --- Tetikli yanıp sönen banner(lar) ---
        if not self.active:
            self.flash = 0
            return
        self.flash = (self.flash + 1) % max(1, int(fps))
        blink = self.flash < fps / 2
        for i, a in enumerate(self.active[:4]):
            bh = 40
            y0 = py + th + 22 + i * (bh + 4)
            ov = frame.copy()
            cv2.rectangle(ov, (10, y0), (w - 10, y0 + bh), C["crit_red"], -1)
            cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
            bc = C["white"] if blink else C["crit_red"]
            cv2.rectangle(frame, (10, y0), (w - 10, y0 + bh), bc, 2)
            ttc_txt = f"{a['ttc']:.1f}s" if a["ttc"] != float("inf") else "-"
            t1 = f"5G UYARI  #{a['track_id']}  Risk {a['risk']:.0f}  Hiz {a['speed']:.0f}km/h  TTC {ttc_txt}"
            t2 = a["type"] + ("  | SERIT IHLALI" if a["lane"] else "")
            cv2.putText(frame, t1, (18, y0 + 17),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, C["white"], 1, cv2.LINE_AA)
            cv2.putText(frame, t2, (18, y0 + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 200, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# Track ID Birleştirici — ByteTrack ID atlama sorununu çözer
#   Eski track ve yeni track centroid'i yakınsa eski ID korunur.
# ═══════════════════════════════════════════════════════════════
class TrackMerger:
    def __init__(self, merge_dist: float = TRACK_MERGE_DIST):
        self.merge_dist = merge_dist
        self.last_seen: Dict[int, Tuple[float, float, int]] = {}  # tid → (cx, cy, frame_idx)
        self.id_map: Dict[int, int] = {}                          # new_id → old_id

    def resolve(self, tid: int, cx: float, cy: float, frame_idx: int) -> int:
        """Yeni track ID'yi, eşleşen eski ID'yle birleştir veya olduğu gibi döndür."""
        # Eğer bu ID zaten bilinen bir merge varsa onu kullan
        resolved = self.id_map.get(tid, tid)

        # Bu ID'yi daha önce gördük mü?
        if resolved in self.last_seen:
            self.last_seen[resolved] = (cx, cy, frame_idx)
            return resolved

        # Yeni bir ID — ölü track'lerden birine yakın mı?
        best_old = None
        best_dist = float("inf")
        for old_id, (ox, oy, of) in self.last_seen.items():
            # Son 10 karede kaybolmuş track'ler arasında ara
            if frame_idx - of > KALMAN_MAX_AGE:
                continue
            dist = math.hypot(cx - ox, cy - oy)
            if dist < self.merge_dist and dist < best_dist:
                best_dist = dist
                best_old = old_id

        if best_old is not None:
            # Birleştir: yeni ID → eski ID
            self.id_map[tid] = best_old
            self.last_seen[best_old] = (cx, cy, frame_idx)
            return best_old
        else:
            # Gerçekten yeni track
            self.last_seen[tid] = (cx, cy, frame_idx)
            return tid

    def cleanup(self, frame_idx: int):
        """Çok eski track'leri temizle."""
        dead = [k for k, (_, _, f) in self.last_seen.items()
                if frame_idx - f > KALMAN_MAX_AGE * 3]
        for k in dead:
            self.last_seen.pop(k, None)
        dead_map = [k for k, v in self.id_map.items() if v in dead]
        for k in dead_map:
            self.id_map.pop(k, None)


# ═══════════════════════════════════════════════════════════════
# Çizim yardımcıları — Kalman-düzeltilmiş kutularla profesyonel overlay
# ═══════════════════════════════════════════════════════════════
def text_bg(img, txt, org, scale=0.5, color=C["white"], thick=1,
            bg=C["black"], font=cv2.FONT_HERSHEY_SIMPLEX, pad=3):
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), bg, -1)
    cv2.putText(img, txt, org, font, scale, color, thick, cv2.LINE_AA)


def draw_vehicle(frame, box, label, tid, speed, risk, viol, ttc):
    """Araç kutusunu ve bilgilerini çiz. box: Kalman-düzeltilmiş (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = box
    # Kutu rengi: ihlal varsa kırmızı, yoksa sınıf rengi
    border = C["viol"] if viol else C.get(label, C["white"])
    thickness = 3 if viol else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness)

    # Track ID + sınıf etiketi
    id_text = f"#{tid} {label[:3].upper()}"
    text_bg(frame, id_text, (x1, y1 - 2),
            scale=0.46, color=C["id_tag"], bg=(25, 25, 25))

    # Hız göstergesi (sağ üst köşe)
    if speed > 1:
        sc = C["risk_crit"] if speed > SPEED_LIMIT_KMH else C["white"]
        text_bg(frame, f"{speed:.0f}km/h", (max(x1, x2 - 80), y1 - 2),
                scale=0.46, color=sc, bg=(25, 25, 25))

    # Risk barı (kutunun altında) — renk geçişli
    bw = x2 - x1
    if bw > 10:
        fill = int(bw * min(risk, 100) / 100)
        cv2.rectangle(frame, (x1, y2 + 1), (x2, y2 + 7), (45, 45, 45), -1)
        bar_color = RiskScorer.color(risk)
        cv2.rectangle(frame, (x1, y2 + 1), (x1 + fill, y2 + 7), bar_color, -1)
        # Risk değeri yazısı
        risk_label = f"R:{risk:.0f}"
        text_bg(frame, risk_label, (x1, y2 + 22), scale=0.38, color=bar_color, bg=(20, 20, 20))

    # TTC göstergesi
    if ttc != float("inf") and ttc < TTC_WARNING_S:
        tc = C["risk_crit"] if ttc < TTC_CRITICAL_S else C["risk_high"]
        text_bg(frame, f"TTC {ttc:.1f}s", (x1, y2 + 38), scale=0.44, color=tc)

    # Şerit ihlali etiketi
    if viol:
        y_offset = y2 + (54 if ttc != float("inf") and ttc < TTC_WARNING_S else 38)
        text_bg(frame, "SERIT IHLALI", (x1, y_offset),
                scale=0.42, color=C["viol"])


def draw_arrow(frame, box, motion, viol):
    dx, dy = motion
    mag = math.hypot(dx, dy)
    if mag < 1.5:
        return
    x1, y1, x2, y2 = box
    mcx, mcy = (x1 + x2) // 2, (y1 + y2) // 2
    scale = min(70.0, (x2 - x1) * 0.6) / mag
    tip = (int(mcx + dx * scale), int(mcy + dy * scale))
    cv2.arrowedLine(frame, (mcx, mcy), tip,
                    C["arrow_viol"] if viol else C["arrow"], 2,
                    tipLength=0.35, line_type=cv2.LINE_AA)


def draw_distance(frame, ca, cb, dist_m):
    col = C["dist_crit"] if dist_m < DIST_CRITICAL_M else (
          C["dist_warn"] if dist_m < DIST_WARNING_M else C["dist_ok"])
    cv2.line(frame, (int(ca[0]), int(ca[1])), (int(cb[0]), int(cb[1])), col, 1, cv2.LINE_AA)
    mx, my = int((ca[0] + cb[0]) / 2), int((ca[1] + cb[1]) / 2)
    text_bg(frame, f"{dist_m:.1f}m", (mx, my), scale=0.42, color=col)


def draw_panel(frame, frame_idx, total, fps, n_veh, n_alert, scale_mpp, avg_speed, max_risk):
    """Sağ alt bilgi paneli — sahne metrikleri."""
    h, w = frame.shape[:2]
    pw, ph = 250, 130
    x0, y0 = w - pw - 8, h - ph - 8
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + pw, y0 + ph), C["panel"], -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (80, 80, 80), 1)
    rows = [
        f"Kare    : {frame_idx}/{total}",
        f"FPS     : {fps:.1f}",
        f"Arac    : {n_veh}",
        f"Olcek   : {scale_mpp*100:.1f} cm/px",
        f"Ort.Hiz : {avg_speed:.0f} km/h",
        f"MaxRisk : {max_risk:.0f}",
        f"Uyari   : {n_alert}",
    ]
    for i, t in enumerate(rows):
        cv2.putText(frame, t, (x0 + 8, y0 + 19 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["white"], 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# ANA İŞLEM
# ═══════════════════════════════════════════════════════════════
def process_video(video_path: Path, v_model: YOLO, p_model: YOLO, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [HATA] açılamadı: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = video_path.stem
    out_video = out_dir / f"{stem}_pipeline.mp4"
    out_json = out_dir / f"{stem}_pipeline.json"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (PROC_W, PROC_H))

    # Modüller
    kalman = KalmanBank()
    scale_est = PerspectiveScaleModel(PROC_H)
    flow = OpticalFlowDirection()
    roi = AdaptiveROI()
    speed_est = SpeedEstimator(fps)
    dist_calc = DistanceCalculator()
    ttc_calc = TTCCalculator(fps)
    lane = LaneAnalyzer()
    risk_scorer = RiskScorer()
    alert = AlertModule(fps)
    events = EventLogger(video_path.name)
    merger = TrackMerger()

    motion_ema: Dict[int, Tuple[float, float]] = {}   # yön oku yumuşatma

    all_frames = []
    frame_idx = 0
    t0 = time.time()
    print(f"  Giriş {ow}x{oh}@{fps:.0f}fps → işlem {PROC_W}x{PROC_H}")

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame = cv2.resize(raw, (PROC_W, PROC_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_sec = frame_idx / fps

        # 1) Araç tespiti + ByteTrack (NMS/conf YOLO içinde) -------
        vres = v_model.track(frame, persist=True, tracker="bytetrack.yaml",
                             conf=CONF_VEHICLE, iou=IOU_THRESH,
                             device=DEVICE, verbose=False, imgsz=PROC_W)[0]
        # 2) Plaka tespiti -----------------------------------------
        pres = p_model.predict(frame, conf=CONF_PLATE, iou=IOU_THRESH,
                               device=DEVICE, verbose=False)[0]
        # 3) Boyalı şerit (varsa) ----------------------------------
        left, right = lane.detect_painted(frame)

        boxes = vres.boxes
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([])

        foot: Dict[int, Tuple[float, float]] = {}     # yer-temas noktası (mesafe/TTC)
        track_box: Dict[int, Tuple[int, int, int, int]] = {}   # Kalman-düzeltilmiş kutular
        track_label: Dict[int, str] = {}
        speeds: Dict[int, float] = {}
        motions: Dict[int, Tuple[float, float]] = {}
        active_tids: set = set()
        fdet = {"frame": frame_idx, "time_sec": round(t_sec, 2),
                "vehicles": [], "plates": [], "alerts": []}

        # (a) Önce TÜM örnekleri topla → kararlı ölçek modelini güncelle
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = v_model.names[int(box.cls[0])]
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            scale_est.add_sample(float(y2), bbox_w, bbox_h, label)
        scale_est.refresh()

        # (b) Araç başına Kalman, ROI, optical flow, hız ----------
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = v_model.names[cls]
            raw_tid = int(ids[i]) if len(ids) > i else -(i + 1)

            # Track ID birleştirme
            cx_raw, cy_raw = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tid = merger.resolve(raw_tid, cx_raw, cy_raw, frame_idx)

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w_raw = float(x2 - x1)
            h_raw = float(y2 - y1)

            # ROI: yol bölgesi dışındaki araçları ele (gevşek)
            foot_y = float(y2)
            roi.add(cx, foot_y)
            if not roi.contains(cx, foot_y):
                continue

            # ── Kalman düzeltme: konum + kutu boyutu birlikte ──
            scx, scy, sw, sh = kalman.step(tid, cx, cy, w_raw, h_raw)
            active_tids.add(tid)

            # Kalman-düzeltilmiş kutu koordinatları
            kx1 = int(scx - sw / 2)
            ky1 = int(scy - sh / 2)
            kx2 = int(scx + sw / 2)
            ky2 = int(scy + sh / 2)
            # Görüntü sınırlarına kırp
            kx1 = max(0, min(kx1, PROC_W - 1))
            ky1 = max(0, min(ky1, PROC_H - 1))
            kx2 = max(kx1 + 4, min(kx2, PROC_W))
            ky2 = max(ky1 + 4, min(ky2, PROC_H))

            # Kalman piksel hızı
            pvx, pvy = kalman.velocity(tid)

            # Ölçek: KONUMA (foot_y) bağlı kararlı model — bbox titremesinden bağımsız
            scale = scale_est.scale_at(foot_y)

            # Optical flow yönü (varsa) → Kalman hızıyla füzyon → EMA yumuşatma
            # Kalman'a daha fazla güven (0.5/0.5 → 0.3/0.7): optical flow gürültüsünü azalt
            of = flow.flow_in_box(gray, (x1, y1, x2, y2))
            if of is not None and math.hypot(*of) > 0.8:
                mvx = 0.3 * of[0] + 0.7 * pvx
                mvy = 0.3 * of[1] + 0.7 * pvy
            else:
                mvx, mvy = pvx, pvy
            pe = motion_ema.get(tid, (mvx, mvy))
            mvx = 0.35 * mvx + 0.65 * pe[0]   # EMA daha kararlı (0.4/0.6 → 0.35/0.65)
            mvy = 0.35 * mvy + 0.65 * pe[1]
            motion_ema[tid] = (mvx, mvy)
            motions[tid] = (mvx, mvy)

            # Hız (Kalman piksel hızı × konum-ölçeği + moving average)
            speed = speed_est.estimate(tid, pvx, pvy, scale, label)

            foot[tid] = (scx, float(ky2))       # Kalman-düzeltilmiş foot point
            track_box[tid] = (kx1, ky1, kx2, ky2)
            track_label[tid] = label
            speeds[tid] = speed
            fdet["vehicles"].append({
                "track_id": tid, "class": label, "conf": round(conf, 3),
                "bbox": [kx1, ky1, kx2, ky2], "speed_kmh": speed,
                "scale_m_px": round(scale, 5),
            })

        # Kalman cleanup: aktif olmayan track'leri temizle
        kalman.cleanup(active_tids)
        merger.cleanup(frame_idx)

        # Ölü track'lerin motion EMA'sını temizle
        dead_motion = [k for k in motion_ema if k not in active_tids]
        for k in dead_motion:
            motion_ema.pop(k, None)

        # 9) Araçlar arası mesafe (foot noktaları + perspektif ölçek)
        dists = dist_calc.pairwise(foot, scale_est)

        # 10) TTC --------------------------------------------------
        ttcs: Dict[int, float] = {}
        for tid in foot:
            _, ttc = ttc_calc.compute(tid, foot, motions, scale_est)
            ttcs[tid] = ttc

        # 11) Şerit ihlali + 12) Risk ------------------------------
        viols: List[int] = []
        risks: Dict[int, float] = {}
        leaders: Dict[int, int] = {}        # tid → en yakın komşu (çizim için)
        n_veh = len(foot)
        for tid in foot:
            kx1, ky1, kx2, ky2 = track_box[tid]
            cx, cy = (kx1 + kx2) / 2.0, float(ky2)
            is_viol, mode = lane.check(tid, cx, cy, speeds[tid], left, right)
            if is_viol:
                viols.append(tid)
            near_id, nd = dist_calc.nearest(tid, dists)
            if near_id is not None:
                leaders[tid] = near_id
            risk = risk_scorer.compute(speeds[tid], nd, ttcs[tid], is_viol, n_veh)
            risks[tid] = risk
            for v in fdet["vehicles"]:
                if v["track_id"] == tid:
                    v.update({"lane_violation": is_viol, "lane_mode": mode,
                              "nearest_dist_m": None if nd == float("inf") else nd,
                              "ttc_s": None if ttcs[tid] == float("inf") else ttcs[tid],
                              "risk": risk,
                              "risk_level": RiskScorer.label(risk)})

            # Event logging — her event'e risk bilgisi eklenir
            risk_level = RiskScorer.label(risk)
            if speeds[tid] > SPEED_LIMIT_KMH:
                events.log(frame_idx, t_sec, "OVERSPEED", tid,
                           {"speed_kmh": speeds[tid]}, risk, risk_level)
            if nd != float("inf") and nd < DIST_CRITICAL_M:
                events.log(frame_idx, t_sec, "CLOSE_FOLLOW", tid,
                           {"dist_m": nd, "nearest_id": near_id}, risk, risk_level)
            if ttcs[tid] != float("inf") and ttcs[tid] < TTC_CRITICAL_S:
                events.log(frame_idx, t_sec, "LOW_TTC", tid,
                           {"ttc_s": ttcs[tid]}, risk, risk_level)
            if is_viol:
                events.log(frame_idx, t_sec, "LANE_VIOLATION", tid,
                           {"mode": mode}, risk, risk_level)

        # 13) 5G uyarı --------------------------------------------
        alert.evaluate(frame_idx, risks, speeds, ttcs, viols)
        global_max = max(risks.values()) if risks else 0.0
        scene_risk = alert.scene_risk(risks)

        for a in alert.active:
            events.log(frame_idx, t_sec, "RISK_5G", a["track_id"],
                       {"risk": a["risk"], "type": a["type"],
                        "speed_kmh": a["speed"],
                        "ttc_s": a["ttc"] if a["ttc"] != float("inf") else None,
                        "lane_violation": a["lane"]},
                       a["risk"], a["type"])
            fdet["alerts"].append(a.copy())

        # Periyodik sahne özeti (her 50 karede bir)
        if frame_idx % 50 == 0 and n_veh > 0:
            avg_speed = sum(speeds.values()) / len(speeds) if speeds else 0.0
            events.log_scene(frame_idx, t_sec, n_veh, avg_speed, global_max,
                             RiskScorer.label(scene_risk), alert.total_alerts)

        # ---- Çizim (Kalman-düzeltilmiş kutularla) ----------------
        roi.rebuild(PROC_W, PROC_H)
        lane.draw(frame, left, right)

        # Mesafe: sadece her aracın EN YAKIN komşusuna tek çizgi (karmaşa yok)
        drawn_pairs = set()
        for tid, near in leaders.items():
            key = tuple(sorted((tid, near)))
            if key in drawn_pairs:
                continue
            dm = dists.get(key)
            if dm is not None and dm <= DIST_WARNING_M * 2:
                draw_distance(frame, foot[tid], foot[near], dm)
                drawn_pairs.add(key)

        # Araçları çiz (Kalman-düzeltilmiş kutularla — titremeyen)
        for tid in track_box:
            draw_vehicle(frame, track_box[tid], track_label[tid], tid,
                         speeds[tid], risks[tid], tid in viols, ttcs[tid])
            draw_arrow(frame, track_box[tid], motions[tid], tid in viols)

        # Plaka kutularını çiz
        for pb in pres.boxes:
            px1, py1, px2, py2 = map(int, pb.xyxy[0])
            pcf = float(pb.conf[0])
            cv2.rectangle(frame, (px1, py1), (px2, py2), C["license_plate"], 2)
            text_bg(frame, f"Plaka {pcf:.2f}", (px1, py1 - 2), scale=0.4,
                    color=C["license_plate"])
            fdet["plates"].append({"conf": round(pcf, 3), "bbox": [px1, py1, px2, py2]})

        # 5G durumu ve bilgi paneli
        alert.draw(frame, fps, global_max)
        elapsed = time.time() - t0
        pf = frame_idx / elapsed if elapsed > 0 else 0
        avg_speed = sum(speeds.values()) / len(speeds) if speeds else 0.0
        draw_panel(frame, frame_idx, total, pf, n_veh, alert.total_alerts,
                   scale_est._global(), avg_speed, global_max)

        writer.write(frame)
        all_frames.append(fdet)
        flow.set_frame(gray)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    {frame_idx}/{total}  {pf:.1f}fps  arac:{n_veh}  "
                  f"uyari:{alert.total_alerts}  maxrisk:{global_max:.0f}")

    cap.release()
    writer.release()

    ev_json, ev_csv = events.save(out_dir, stem)
    summary = {
        "video": video_path.name, "total_frames": frame_idx, "fps": fps,
        "resolution": [ow, oh], "proc_resolution": [PROC_W, PROC_H],
        "total_5g_alerts": alert.total_alerts, "total_events": len(events.events),
        "frames": all_frames,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Tamamlandı: {frame_idx} kare | {time.time()-t0:.1f}s | "
          f"5G uyarı:{alert.total_alerts} | olay:{len(events.events)}")
    print(f"    Video : {out_video}")
    print(f"    JSON  : {out_json}")
    if ev_json:
        print(f"    Olay  : {ev_json.name}, {ev_csv.name}")


def main():
    ap = argparse.ArgumentParser(description="5G Akıllı Trafik İzleme Pipeline (genel)")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="Video dosyası veya klasör (varsayılan: teknofest_test/)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Çıktı klasörü (varsayılan: results_pipeline/)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cihaz: {DEVICE.upper()}")
    print("Modeller yükleniyor...")
    v_model = YOLO(str(VEHICLE_MODEL_PATH))
    p_model = YOLO(str(PLATE_MODEL_PATH))
    print(f"  Araç : {VEHICLE_MODEL_PATH.name}  |  Plaka : {PLATE_MODEL_PATH.name}")

    if in_path.is_dir():
        videos = sorted([p for p in in_path.glob("*")
                         if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")])
    elif in_path.is_file():
        videos = [in_path]
    else:
        print(f"[HATA] Girdi bulunamadı: {in_path}")
        return

    if not videos:
        print(f"[UYARI] Video bulunamadı: {in_path}")
        return

    for vp in videos:
        print(f"\n{'='*56}\n  {vp.name}\n{'='*56}")
        process_video(vp, v_model, p_model, out_dir)

    print(f"\nTüm videolar tamamlandı → {out_dir}")


if __name__ == "__main__":
    main()
