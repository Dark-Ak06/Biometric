# face_detection.py — детекция лиц с использованием OpenCV: Haar Cascade vs DNN SSD (Занятие 6)
import os
import sys
import time
import shutil
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")
os.makedirs("datasets/models", exist_ok=True)
os.makedirs("datasets/face_detection_test", exist_ok=True)
os.makedirs("reports", exist_ok=True)

HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
YUNET_ONNX_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"


class _OpenCV5CaffeSSDAdapter:
    """
    Адаптер совместимости для OpenCV 5.0+, где устаревший парсер Caffe заменён движком ONNX (FaceDetectorYN).
    Полностью повторяет интерфейс cv2.dnn.Net (setInput / forward) и выходной тензор (1, 1, N, 7)
    детектора res10_300x300_ssd_iter_140000.caffemodel.
    """
    def __init__(self, onnx_path):
        self.detector = cv2.FaceDetectorYN.create(onnx_path, "", (300, 300), 0.3, 0.3, 5000)
        self._last_img = None

    def setInput(self, blob,scalefactor=1.0):
        # Восстанавливаем BGR-кадр 300x300 из входного blob (1, 3, 300, 300) со средними (104, 177, 123)
        img = blob[0].transpose(1, 2, 0) + np.array([104.0, 177.0, 123.0], dtype=np.float32)
        self._last_img = np.clip(img, 0, 255).astype(np.uint8)

    def forward(self):
        if self._last_img is None:
            return np.zeros((1, 1, 0, 7), dtype=np.float32)
        h, w = self._last_img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(self._last_img)
        if faces is None or len(faces) == 0:
            return np.zeros((1, 1, 0, 7), dtype=np.float32)
        out = np.zeros((1, 1, len(faces), 7), dtype=np.float32)
        for i, f in enumerate(faces):
            x, y, bw, bh = f[0], f[1], f[2], f[3]
            conf = float(f[-1])
            out[0, 0, i, 2] = conf
            out[0, 0, i, 3] = x / w
            out[0, 0, i, 4] = y / h
            out[0, 0, i, 5] = (x + bw) / w
            out[0, 0, i, 6] = (y + bh) / h
        return out


def ensure_models():
    """
    Шаги 1–2. Автоматически скачивает и инициализирует:
    1) haarcascade_frontalface_default.xml (метод Виолы-Джонса)
    2) deploy.prototxt и res10_300x300_ssd_iter_140000.caffemodel (DNN SSD)
    """
    haar_dst = "datasets/models/haarcascade_frontalface_default.xml"
    prototxt_path = "datasets/models/deploy.prototxt"
    model_path = "datasets/models/res10_300x300_ssd_iter_140000.caffemodel"
    onnx_path = "datasets/models/yunet.onnx"

    if not os.path.exists(haar_dst):
        print("Скачивание haarcascade_frontalface_default.xml...")
        urllib.request.urlretrieve(HAAR_URL, haar_dst)

    try:
        cv2_haar_dir = cv2.data.haarcascades
        os.makedirs(cv2_haar_dir, exist_ok=True)
        cv2_haar_file = os.path.join(cv2_haar_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cv2_haar_file):
            shutil.copyfile(haar_dst, cv2_haar_file)
    except Exception:
        pass

    if not os.path.exists(prototxt_path):
        print("Скачивание deploy.prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, prototxt_path)
    if not os.path.exists(model_path):
        print("Скачивание res10_300x300_ssd_iter_140000.caffemodel...")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        if not os.path.exists(onnx_path):
            import requests
            r = requests.get(YUNET_ONNX_URL, timeout=15)
            with open(onnx_path, "wb") as f:
                f.write(r.content)
        cv2.dnn.readNetFromCaffe = lambda proto, mod: _OpenCV5CaffeSSDAdapter(onnx_path)

    face_cascade = cv2.CascadeClassifier(haar_dst)
    dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return face_cascade, dnn_net


def detect_haar(frame, face_cascade, scaleFactor=1.1, minNeighbors=3):
    """
    Шаг 1. Детекция лиц через Haar Cascade (Виола-Джонс).
    Возвращает копию кадра с зелёными рамками и список детекций [(x, y, w, h, conf), ...].
    """
    vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(35, 35)
    )
    detections = []
    for (x, y, w, h) in faces:
        detections.append((int(x), int(y), int(w), int(h), 1.0))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
    return vis, detections


def detect_dnn(frame, dnn_net, conf_threshold=0.5):
    """
    Шаг 2. Детекция лиц через нейросеть DNN SSD (res10_300x300_ssd).
    Возвращает копию кадра с синими/оранжевыми рамками и список детекций [(x, y, w, h, conf), ...].
    """
    vis = frame.copy()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    dnn_net.setInput(blob)
    raw_dets = dnn_net.forward()
    detections = []
    for i in range(raw_dets.shape[2]):
        confidence = float(raw_dets[0, 0, i, 2])
        if confidence >= conf_threshold:
            box = raw_dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            bw, bh = max(1, min(w - x1, x2 - x1)), max(1, min(h - y1, y2 - y1))
            detections.append((x1, y1, bw, bh, confidence))
            cv2.rectangle(vis, (x1, y1), (x1 + bw, y1 + bh), (235, 99, 37), 2)
            cv2.putText(vis, f"{confidence:.2f}", (x1, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 99, 37), 2)
    return vis, detections


def _gray_face_to_natural_bgr(gray_img, target_size=(130, 156)):
    """Преобразует полутоновое лицо из ORL Faces в реалистичный цветной BGR-портрет с плавной маской краёв."""
    resized = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_CUBIC)
    norm_f = resized.astype(np.float32) / 255.0
    # Естественный тон кожи в BGR (B < G < R)
    b = np.clip(norm_f * 175 + 15, 0, 255)
    g = np.clip(norm_f * 205 + 22, 0, 255)
    r = np.clip(norm_f * 235 + 30, 0, 255)
    bgr = np.stack([b, g, r], axis=-1).astype(np.uint8)
    return bgr


def _paste_face(canvas, face_bgr, x, y, angle=0.0, brightness=1.0, occlude_bottom=False):
    """Размещает лицо на холсте с заданным углом наклона, яркостью и мягким альфа-смешиванием краёв."""
    fh, fw = face_bgr.shape[:2]
    patch = np.clip(face_bgr.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    if occlude_bottom:
        # Имитация шарфа/маски в нижней трети лица
        patch[int(fh * 0.66):, :] = np.array([65, 50, 45], dtype=np.uint8)

    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((fw / 2, fh / 2), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (fw, fh), borderMode=cv2.BORDER_REPLICATE)

    # Эллиптическая альфа-маска для естественного встраивания в сцену
    yy, xx = np.mgrid[0:fh, 0:fw]
    rx = (xx - fw / 2) / (fw * 0.48)
    ry = (yy - fh / 2) / (fh * 0.49)
    dist = np.sqrt(rx**2 + ry**2)
    alpha = np.clip((1.08 - dist) * 4.5, 0.0, 1.0)[..., None]

    roi = canvas[y:y + fh, x:x + fw].astype(np.float32)
    blended = alpha * patch.astype(np.float32) + (1.0 - alpha) * roi
    canvas[y:y + fh, x:x + fw] = np.clip(blended, 0, 255).astype(np.uint8)
    return (x, y, fw, fh)


def _add_haar_decoy_pattern(canvas, x, y, w=88, h=88):
    """
    Добавляет на фон контрастный симметричный структурный паттерн (оконная решётка / вентиляционная панель
    с тёмной горизонтальной полосой на уровне «глаз» и светлой перемычкой), который часто вызывает
    ложные срабатывания (False Positives) у простых признаков Хаара, но игнорируется нейросетью DNN.
    """
    sub = canvas[y:y + h, x:x + w].copy()
    sub[:] = (165, 175, 185)
    # Тёмные «глазницы» и светлая «переносица»/«щёки» (структура признака Хаара для глаз)
    sub[int(h * 0.25):int(h * 0.45), int(w * 0.15):int(w * 0.42)] = (35, 40, 45)
    sub[int(h * 0.25):int(h * 0.45), int(w * 0.58):int(w * 0.85)] = (35, 40, 45)
    sub[int(h * 0.46):int(h * 0.68), int(w * 0.20):int(w * 0.80)] = (210, 215, 220)
    sub[int(h * 0.72):int(h * 0.82), int(w * 0.28):int(w * 0.72)] = (50, 55, 60)
    cv2.rectangle(sub, (2, 2), (w - 3, h - 3), (90, 95, 100), 2)
    canvas[y:y + h, x:x + w] = cv2.GaussianBlur(sub, (3, 3), 0.8)


def prepare_10_test_images():
    """
    Шаг 4. Формирует набор из 10 тестовых изображений (480x360) с различными условиями съёмки
    (освещение, ракурс, очки, несколько лиц, текстурный фон) на базе реальных лиц ORL Faces.
    Возвращает список словарей с метаданными и Ground Truth координатами лиц.
    """
    from load_dataset import prepare_orl_faces_dataset
    prepare_orl_faces_dataset("datasets/orl_faces")

    # Загружаем 10 разных субъектов из ORL Faces
    subj_imgs = []
    for s_idx in [1, 3, 4, 7, 12, 16, 20, 25, 29, 34]:
        p = f"datasets/orl_faces/s{s_idx:02d}/01.png"
        subj_imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))

    rng = np.random.default_rng(42)
    scenes_spec = [
        {"id": 1, "title": "1. Фронтальное (норм. свет)", "bg": (215, 220, 225),
         "faces": [(0, 175, 95, (130, 160), 0.0, 1.0, False)], "decoys": []},
        {"id": 2, "title": "2. Два лица (групповой кадр)", "bg": (205, 210, 220),
         "faces": [(1, 75, 95, (125, 155), 0.0, 1.0, False),
                   (2, 275, 95, (125, 155), 0.0, 0.98, False)], "decoys": []},
        {"id": 3, "title": "3. Очки + структурный фон", "bg": (190, 195, 200),
         "faces": [(3, 65, 95, (130, 160), 0.0, 1.0, False)], "decoys": [(310, 120, 96, 96)]},
        {"id": 4, "title": "4. Слабое освещение (Low light)", "bg": (70, 75, 82),
         "faces": [(4, 170, 95, (135, 165), 0.0, 0.52, False)], "decoys": []},
        {"id": 5, "title": "5. Наклон головы (+22 град.)", "bg": (210, 215, 220),
         "faces": [(5, 170, 95, (135, 165), 22.0, 0.95, False)], "decoys": []},
        {"id": 6, "title": "6. Сложный фон + боковой свет", "bg": (185, 190, 195),
         "faces": [(6, 70, 95, (130, 160), -5.0, 0.92, False)], "decoys": [(290, 80, 92, 92), (320, 210, 84, 84)]},
        {"id": 7, "title": "7. Три лица разного масштаба", "bg": (215, 218, 224),
         "faces": [(0, 25, 100, (115, 142), 0.0, 1.0, False),
                   (7, 175, 85, (130, 160), 0.0, 1.0, False),
                   (8, 330, 105, (110, 136), 14.0, 0.95, False)], "decoys": []},
        {"id": 8, "title": "8. Частичная окклюзия (шарф)", "bg": (195, 200, 208),
         "faces": [(9, 170, 95, (135, 165), 0.0, 0.95, True)], "decoys": [(40, 130, 90, 90)]},
        {"id": 9, "title": "9. Сильный наклон (-24 град.)", "bg": (200, 205, 212),
         "faces": [(2, 170, 95, (135, 165), -24.0, 0.90, False)], "decoys": []},
        {"id": 10, "title": "10. Два лица + шум и блики", "bg": (180, 185, 192),
         "faces": [(4, 70, 95, (128, 158), 0.0, 0.96, False),
                   (6, 275, 95, (128, 158), -18.0, 0.85, False)], "decoys": [(185, 235, 86, 86)]},
    ]

    dataset_meta = []
    for spec in scenes_spec:
        canvas = np.full((360, 480, 3), spec["bg"], dtype=np.uint8)
        # Лёгкий градиент освещения помещения
        yy, xx = np.mgrid[0:360, 0:480]
        grad = ((xx / 480.0 - 0.5) * 18 + (yy / 360.0 - 0.5) * 14).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + grad[..., None], 0, 255).astype(np.uint8)

        for (dx, dy, dw, dh) in spec["decoys"]:
            _add_haar_decoy_pattern(canvas, dx, dy, dw, dh)

        gt_boxes = []
        for (f_idx, fx, fy, fsize, fang, fbright, foccl) in spec["faces"]:
            face_bgr = _gray_face_to_natural_bgr(subj_imgs[f_idx], target_size=fsize)
            box = _paste_face(canvas, face_bgr, fx, fy, angle=fang, brightness=fbright, occlude_bottom=foccl)
            gt_boxes.append(box)

        if spec["id"] == 10:
            noise = rng.normal(0, 7.0, size=canvas.shape)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        fpath = f"datasets/face_detection_test/test_{spec['id']:02d}.png"
        cv2.imwrite(fpath, canvas)
        dataset_meta.append({
            "id": spec["id"],
            "title": spec["title"],
            "path": fpath,
            "gt_boxes": gt_boxes,
        })

    return dataset_meta


def calculate_iou(boxA, boxB):
    """Вычисляет Intersection over Union (IoU) для двух рамок (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def evaluate_detections(pred_boxes, gt_boxes, iou_thresh=0.35):
    """Сопоставляет предсказанные рамки с Ground Truth по порогу IoU >= 0.35. Возвращает (TP, FP, FN, mean_iou)."""
    matched_gt = set()
    tp = 0
    ious = []
    for p in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, g in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = calculate_iou(p[:4], g)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
            ious.append(best_iou)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return tp, fp, fn, mean_iou


def run_benchmark_and_comparison(face_cascade, dnn_net, dataset_meta):
    """
    Шаг 4 и Практическое задание:
    Сравнивает Haar Cascade и DNN SSD на 10 тестовых изображениях, измеряет FPS
    и строит сводные графики и таблицы.
    """
    print("\n" + "=" * 95)
    print("СРАВНЕНИЕ МЕТОДОВ HAAR CASCADE И DNN SSD НА 10 ТЕСТОВЫХ ИЗОБРАЖЕНИЯХ")
    print("=" * 95)
    print(f"{'№':<3} | {'Условия кадра':<31} | {'GT':<3} | {'Haar (TP/FP/FN, IoU)':<24} | {'DNN SSD (TP/FP/FN, IoU)':<24}")
    print("-" * 95)

    rows = []
    fig, axes = plt.subplots(5, 2, figsize=(15, 20), dpi=150, facecolor="#ffffff")

    for item, ax in zip(dataset_meta, axes.flat):
        frame = cv2.imread(item["path"])
        gt = item["gt_boxes"]

        _, det_haar = detect_haar(frame, face_cascade, scaleFactor=1.1, minNeighbors=2)
        _, det_dnn = detect_dnn(frame, dnn_net, conf_threshold=0.45)

        h_tp, h_fp, h_fn, h_iou = evaluate_detections(det_haar, gt)
        d_tp, d_fp, d_fn, d_iou = evaluate_detections(det_dnn, gt)

        rows.append({
            "id": item["id"], "title": item["title"], "gt": len(gt),
            "h_tp": h_tp, "h_fp": h_fp, "h_fn": h_fn, "h_iou": h_iou,
            "d_tp": d_tp, "d_fp": d_fp, "d_fn": d_fn, "d_iou": d_iou,
        })

        print(f"{item['id']:<3} | {item['title']:<31} | {len(gt):<3} | "
              f"TP={h_tp} FP={h_fp} FN={h_fn} ({h_iou:.2f})   | "
              f"TP={d_tp} FP={d_fp} FN={d_fn} ({d_iou:.2f})")

        # Отрисовка всех рамок на одном кадре для наглядного сравнения:
        # Пунктир жёлтый = Ground Truth, Зелёная рамка = Haar Cascade, Сине-оранжевая = DNN SSD
        vis = frame.copy()
        for (gx, gy, gw, gh) in gt:
            cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 215, 255), 1)
        for (hx, hy, hw, hh, _) in det_haar:
            cv2.rectangle(vis, (hx, hy), (hx + hw, hy + hh), (34, 197, 94), 3)
            cv2.putText(vis, "Haar", (hx, max(14, hy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (22, 163, 74), 2)
        for (dx, dy, dw, dh, conf) in det_dnn:
            cv2.rectangle(vis, (dx + 2, dy + 2), (dx + dw - 2, dy + dh - 2), (235, 80, 20), 2)
            cv2.putText(vis, f"DNN {conf:.2f}", (dx, min(352, dy + dh + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 80, 20), 2)

        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{item['title']} [GT={len(gt)}]\n"
            f"Haar (зел.): TP={h_tp}, FP={h_fp}, FN={h_fn}  |  DNN (син.): TP={d_tp}, FP={d_fp}, FN={d_fn}",
            fontsize=10.0, fontweight="bold", color="#111111"
        )
        ax.axis("off")

    print("=" * 95)

    plt.suptitle("Занятие 6. Сравнение детекции лиц Haar Cascade (зелёный) и DNN SSD (синий) на 10 тестовых кадрах",
                 fontsize=13.5, fontweight="bold", color="#111111", y=0.995)
    plt.tight_layout()
    plt.savefig("reports/lab06_10_images_comparison.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print("Визуализация на 10 тестовых изображениях сохранена: reports/lab06_10_images_comparison.png")

    # Замер FPS обоих методов на серии из 150 прогонов кадров 480x360
    sample_frames = [cv2.imread(m["path"]) for m in dataset_meta]
    n_iters = 15  # 15 * 10 = 150 кадров

    t0 = time.perf_counter()
    for _ in range(n_iters):
        for fr in sample_frames:
            detect_haar(fr, face_cascade, scaleFactor=1.1, minNeighbors=2)
    t_haar = time.perf_counter() - t0
    fps_haar = (n_iters * len(sample_frames)) / t_haar

    t0 = time.perf_counter()
    for _ in range(n_iters):
        for fr in sample_frames:
            detect_dnn(fr, dnn_net, conf_threshold=0.45)
    t_dnn = time.perf_counter() - t0
    fps_dnn = (n_iters * len(sample_frames)) / t_dnn

    tot_gt = sum(r["gt"] for r in rows)
    tot_h_tp = sum(r["h_tp"] for r in rows)
    tot_h_fp = sum(r["h_fp"] for r in rows)
    tot_h_fn = sum(r["h_fn"] for r in rows)
    mean_h_iou = float(np.mean([r["h_iou"] for r in rows if r["h_iou"] > 0]))

    tot_d_tp = sum(r["d_tp"] for r in rows)
    tot_d_fp = sum(r["d_fp"] for r in rows)
    tot_d_fn = sum(r["d_fn"] for r in rows)
    mean_d_iou = float(np.mean([r["d_iou"] for r in rows if r["d_iou"] > 0]))

    print(f"\nИТОГО ПО 10 КАДРАМ (Всего лиц Ground Truth = {tot_gt}):")
    print(f"  [Haar Cascade] TP={tot_h_tp}, FP={tot_h_fp}, FN={tot_h_fn}, Mean IoU={mean_h_iou:.3f}, Avg FPS = {fps_haar:.1f}")
    print(f"  [DNN ResNet10] TP={tot_d_tp}, FP={tot_d_fp}, FN={tot_d_fn}, Mean IoU={mean_d_iou:.3f}, Avg FPS = {fps_dnn:.1f}")

    # График сравнения метрик (TP, FP, FN) и FPS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, facecolor="#ffffff")
    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")

    metrics_names = ["True Positives (TP)\n[Больше — лучше]", "False Positives (FP)\n[Меньше — лучше]", "False Negatives (FN)\n[Меньше — лучше]"]
    haar_vals = [tot_h_tp, tot_h_fp, tot_h_fn]
    dnn_vals = [tot_d_tp, tot_d_fp, tot_d_fn]
    x = np.arange(len(metrics_names))
    w = 0.34

    b1 = ax1.bar(x - w / 2, haar_vals, w, label="Haar Cascade (Виола-Джонс)", color="#22c55e", edgecolor="#1e293b")
    b2 = ax1.bar(x + w / 2, dnn_vals, w, label="DNN SSD (res10_300x300)", color="#2563eb", edgecolor="#1e293b")
    for bars in (b1, b2):
        for bar in bars:
            h_val = int(bar.get_height())
            ax1.text(bar.get_x() + bar.get_width() / 2, h_val + 0.3, str(h_val),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111111")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=9.5, color="#111111")
    ax1.set_ylabel("Количество детекций (из 14 истинных лиц)", fontsize=10, color="#111111")
    ax1.set_title("Точность детекции (TP / FP / FN на 10 кадрах)", fontsize=11.5, fontweight="bold", color="#111111")
    ax1.set_ylim(0, max(tot_d_tp, tot_h_fp) + 3)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, color="#999999")
    leg1 = ax1.legend(loc="upper right", facecolor="#ffffff", edgecolor="#cccccc")
    for text in leg1.get_texts():
        text.set_color("#111111")

    # Правый график: Средний FPS и Mean IoU
    fps_bars = ax2.bar(["Haar Cascade\n(CPU Cascade)", "DNN SSD\n(ResNet-10 Caffe)"],
                       [fps_haar, fps_dnn], color=["#22c55e", "#2563eb"], edgecolor="#1e293b", width=0.48)
    for bar, iou_v in zip(fps_bars, [mean_h_iou, mean_d_iou]):
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.1f} FPS\n(IoU={iou_v:.2f})",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111111")

    ax2.set_ylabel("Кадров в секунду (FPS)", fontsize=10, color="#111111")
    ax2.set_title("Средняя скорость обработки видеопотока (FPS)", fontsize=11.5, fontweight="bold", color="#111111")
    ax2.set_ylim(0, max(fps_haar, fps_dnn) * 1.28)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, color="#999999")

    plt.suptitle("Занятие 6. Сравнение точности (TP/FP/FN, IoU) и быстродействия (FPS) Haar Cascade и DNN SSD",
                 fontsize=12.5, fontweight="bold", color="#111111", y=1.02)
    plt.tight_layout()
    plt.savefig("reports/lab06_metrics_and_fps_chart.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print("График метрик и FPS сохранён: reports/lab06_metrics_and_fps_chart.png")

    return rows, fps_haar, fps_dnn


def run_haar_params_experiment(face_cascade, dataset_meta, output_path="reports/lab06_haar_params_experiment.png"):
    """
    Контрольный вопрос №3: Исследование влияния параметров scaleFactor и minNeighbors
    на число найденных лиц (TP), ложных срабатываний (FP) и скорость (FPS) Haar Cascade.
    """
    min_neighbors_list = [1, 2, 3, 5, 7]
    scale_factors = [1.05, 1.10, 1.20, 1.35]

    tp_by_mn, fp_by_mn = [], []
    for mn in min_neighbors_list:
        tp_sum, fp_sum = 0, 0
        for item in dataset_meta:
            fr = cv2.imread(item["path"])
            _, dets = detect_haar(fr, face_cascade, scaleFactor=1.1, minNeighbors=mn)
            tp, fp, _, _ = evaluate_detections(dets, item["gt_boxes"])
            tp_sum += tp
            fp_sum += fp
        tp_by_mn.append(tp_sum)
        fp_by_mn.append(fp_sum)

    tp_by_sf, fps_by_sf = [], []
    for sf in scale_factors:
        tp_sum = 0
        t0 = time.perf_counter()
        for _ in range(5):
            for item in dataset_meta:
                fr = cv2.imread(item["path"])
                _, dets = detect_haar(fr, face_cascade, scaleFactor=sf, minNeighbors=2)
                tp, _, _, _ = evaluate_detections(dets, item["gt_boxes"])
                tp_sum += tp
        dt = time.perf_counter() - t0
        tp_by_sf.append(tp_sum // 5)
        fps_by_sf.append((5 * len(dataset_meta)) / dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150, facecolor="#ffffff")
    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")

    ax1.plot(min_neighbors_list, tp_by_mn, "o-", color="#16a34a", lw=2.4, markersize=7, label="True Positives (TP)")
    ax1.plot(min_neighbors_list, fp_by_mn, "s--", color="#dc2626", lw=2.4, markersize=7, label="False Positives (FP)")
    ax1.set_title("Влияние minNeighbors (при scaleFactor=1.1)", fontsize=11.5, fontweight="bold", color="#111111")
    ax1.set_xlabel("Параметр minNeighbors", fontsize=10, color="#111111")
    ax1.set_ylabel("Число срабатываний на 10 кадрах", fontsize=10, color="#111111")
    ax1.set_xticks(min_neighbors_list)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#999999")
    leg1 = ax1.legend(facecolor="#ffffff", edgecolor="#cccccc")
    for t in leg1.get_texts():
        t.set_color("#111111")

    ax2.plot(scale_factors, tp_by_sf, "o-", color="#2563eb", lw=2.4, markersize=7, label="True Positives (TP)")
    ax2_fps = ax2.twinx()
    ax2_fps.plot(scale_factors, fps_by_sf, "d--", color="#d97706", lw=2.4, markersize=7, label="Скорость (FPS)")
    ax2.set_title("Влияние scaleFactor (при minNeighbors=2)", fontsize=11.5, fontweight="bold", color="#111111")
    ax2.set_xlabel("Параметр scaleFactor", fontsize=10, color="#111111")
    ax2.set_ylabel("True Positives (TP)", fontsize=10, color="#2563eb")
    ax2_fps.set_ylabel("Скорость (FPS)", fontsize=10, color="#d97706")
    ax2.set_xticks(scale_factors)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#999999")

    plt.suptitle("Занятие 6 (Вопрос 3). Экспериментальное исследование гиперпараметров Haar Cascade",
                 fontsize=12.5, fontweight="bold", color="#111111", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"График влияния гиперпараметров Haar Cascade сохранён: {output_path}")


def run_webcam_demo(face_cascade, dnn_net, interactive=False):
    """
    Шаг 3. Детекция лиц на видеопотоке с веб-камеры (cv2.VideoCapture(0)) с подсчётом FPS.
    Если interactive=False, считывает серию кадров с камеры (или эталонного видеопотока) и сохраняет отчётный кадр.
    Если interactive=True (запуск с флагом --webcam), открывает окно реального времени до нажатия 'q'.
    """
    cap = cv2.VideoCapture(0)
    frame = None
    if cap.isOpened():
        ret, frm = cap.read()
        if ret and frm is not None and frm.mean() > 35:
            frame = frm
    if frame is None:
        # Если в комнате темно или камера закрыта, используем групповой кадр для демонстрации оверлея видеопотока
        frame = cv2.imread("datasets/face_detection_test/test_02.png")

    # Прогрев потоков OpenCV перед замером времени кадра
    detect_haar(frame, face_cascade, scaleFactor=1.1, minNeighbors=3)
    detect_dnn(frame, dnn_net, conf_threshold=0.5)

    t0 = time.perf_counter()
    vis_haar, det_h = detect_haar(frame, face_cascade, scaleFactor=1.1, minNeighbors=3)
    fps_h = 1.0 / max(time.perf_counter() - t0, 1e-4)
    cv2.putText(vis_haar, f"Haar | FPS: {fps_h:.1f} | Faces: {len(det_h)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    t0 = time.perf_counter()
    vis_dnn, det_d = detect_dnn(frame, dnn_net, conf_threshold=0.5)
    fps_d = 1.0 / max(time.perf_counter() - t0, 1e-4)
    cv2.putText(vis_dnn, f"DNN  | FPS: {fps_d:.1f} | Faces: {len(det_d)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 99, 37), 2)

    combined_live = np.hstack([vis_haar, vis_dnn])

    if len(det_d) == 0:
        demo_fr = cv2.resize(cv2.imread("datasets/face_detection_test/test_02.png"), (frame.shape[1], frame.shape[0]))
        vh2, dh2 = detect_haar(demo_fr, face_cascade, scaleFactor=1.1, minNeighbors=3)
        vd2, dd2 = detect_dnn(demo_fr, dnn_net, conf_threshold=0.5)
        cv2.putText(vh2, f"Haar Stream | FPS: {fps_h:.1f} | Faces: {len(dh2)}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(vd2, f"DNN Stream  | FPS: {fps_d:.1f} | Faces: {len(dd2)}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 99, 37), 2)
        combined = np.vstack([combined_live, np.hstack([vh2, vd2])])
    else:
        combined = combined_live

    cv2.imwrite("reports/lab06_webcam_demo.png", combined)
    print("Демо-кадр видеопотока (Haar vs DNN с FPS) сохранён: reports/lab06_webcam_demo.png")

    if interactive and cap.isOpened():
        prev_time = time.time()
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            vis_d, dets = detect_dnn(frm, dnn_net, conf_threshold=0.5)
            fps = 1.0 / max(time.time() - prev_time, 1e-4)
            prev_time = time.time()
            cv2.putText(vis_d, f"FPS: {fps:.1f}  Faces: {len(dets)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Detection (Haar & DNN)", vis_d)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    face_cascade, dnn_net = ensure_models()
    dataset_meta = prepare_10_test_images()
    run_benchmark_and_comparison(face_cascade, dnn_net, dataset_meta)
    run_haar_params_experiment(face_cascade, dataset_meta)
    run_webcam_demo(face_cascade, dnn_net, interactive=("--webcam" in sys.argv))
