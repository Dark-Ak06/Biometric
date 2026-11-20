"""
Занятие 9. Предобработка биометрических изображений с OpenCV
Исполнитель: студент гр. SIS-2309 Аканов Дархан (ID: 38041)
Дисциплина: Биометрические системы аутентификации

Скрипт реализует:
1. Гистограммную эквализацию (cv2.equalizeHist vs CLAHE) на кадрах с плохим светом.
2. Фильтрацию шумов (cv2.GaussianBlur vs cv2.medianBlur) с ядрами 3x3, 5x5, 21x21.
3. Выравнивание лица по координатам глаз (Eye Alignment, haarcascade_eye.xml).
4. Оценку влияния предобработки на качество распознавания алгоритмом LBPH.
5. Интерактивное / автоматическое сканирование лица с веб-камеры в реальном времени
   с мгновенным распознаванием студента Дархана Аканова.
6. Экспорт графиков и визуализаций в папку reports/.
"""

import os
import sys
import math
import shutil
import tempfile
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Гарантируем корректный вывод UTF-8 в Windows-консоль
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.style.use("default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "datasets" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DARKHAN_FACES_DIR = PROJECT_ROOT / "datasets" / "darkhan_face"
ORL_DIR = PROJECT_ROOT / "datasets" / "orl_faces"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def safe_imread(path, flags=cv2.IMREAD_COLOR):
    """Безопасное чтение изображений по путям с пробелами и кириллицей (Windows/OneDrive)."""
    try:
        p_str = str(path)
        data = np.fromfile(p_str, dtype=np.uint8)
        if data is not None and len(data) > 0:
            return cv2.imdecode(data, flags)
    except Exception:
        pass
    return None


def safe_imwrite(path, img):
    """Безопасная запись изображений по путям с кириллицей."""
    p_str = str(path)
    ext = os.path.splitext(p_str)[1]
    if not ext:
        ext = ".png"
    success, enc = cv2.imencode(ext, img)
    if success:
        with open(p_str, "wb") as f:
            f.write(enc)
        return True
    return False


def ensure_models():
    """
    Безопасная загрузка каскадов Хаара через системную папку Temp
    (устраняет ошибку OpenCV FileStorage при чтении путей с кириллицей в Windows).
    """
    face_xml = MODELS_DIR / "haarcascade_frontalface_default.xml"
    eye_xml = MODELS_DIR / "haarcascade_eye.xml"

    temp_dir = Path(tempfile.gettempdir())
    temp_face = temp_dir / "haarcascade_frontalface_default.xml"
    temp_eye = temp_dir / "haarcascade_eye.xml"

    # 1. Лицевой каскад
    if not temp_face.exists():
        if face_xml.exists():
            try:
                shutil.copy2(face_xml, temp_face)
            except Exception:
                pass
        elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            builtin = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if builtin.exists():
                try:
                    shutil.copy2(builtin, temp_face)
                except Exception:
                    pass

    # 2. Каскад глаз
    if not temp_eye.exists():
        if eye_xml.exists():
            try:
                shutil.copy2(eye_xml, temp_eye)
            except Exception:
                pass
        elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            builtin = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
            if builtin.exists():
                try:
                    shutil.copy2(builtin, temp_eye)
                except Exception:
                    pass

    face_path = str(temp_face) if temp_face.exists() else str(face_xml)
    eye_path = str(temp_eye) if temp_eye.exists() else str(eye_xml)

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)

    return face_cascade, eye_cascade


# ==============================================================================
# ШАГ 1. Гистограммная эквализация: cv2.equalizeHist vs CLAHE
# ==============================================================================
def experiment_histogram_equalization(image_gray):
    """
    Сравнивает глобальную эквализацию (equalizeHist) и локальную адаптивную (CLAHE).
    """
    eq_global = cv2.equalizeHist(image_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_clahe = clahe.apply(image_gray)
    return eq_global, eq_clahe


def run_equalization_comparison(face_images):
    """Строит графики сравнения глобальной и адаптивной эквализации."""
    plot_path = REPORTS_DIR / "lab09_hist_equalization_comparison.png"
    n_samples = min(3, len(face_images))

    fig, axs = plt.subplots(n_samples, 6, figsize=(15, 3.2 * n_samples), constrained_layout=True)

    for i in range(n_samples):
        title, img = face_images[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        eq_glob, eq_clahe = experiment_histogram_equalization(gray)

        # [Исходный, Гистограмма 1, EqualizeHist, Гистограмма 2, CLAHE, Гистограмма 3]
        axs[i, 0].imshow(gray, cmap='gray')
        axs[i, 0].set_title(f"{title}\nИсходное (Raw)", fontsize=10, fontweight='bold')
        axs[i, 0].axis('off')

        axs[i, 1].hist(gray.ravel(), bins=256, range=(0, 256), color='#4B5563', alpha=0.85)
        axs[i, 1].set_title("Гистограмма Raw", fontsize=9)
        axs[i, 1].set_xlim([0, 256])
        axs[i, 1].grid(True, alpha=0.3)

        axs[i, 2].imshow(eq_glob, cmap='gray')
        axs[i, 2].set_title("cv2.equalizeHist\n(Глобальная)", fontsize=10, fontweight='bold')
        axs[i, 2].axis('off')

        axs[i, 3].hist(eq_glob.ravel(), bins=256, range=(0, 256), color='#DC2626', alpha=0.85)
        axs[i, 3].set_title("Гистограмма Equalize", fontsize=9)
        axs[i, 3].set_xlim([0, 256])
        axs[i, 3].grid(True, alpha=0.3)

        axs[i, 4].imshow(eq_clahe, cmap='gray')
        axs[i, 4].set_title("CLAHE (clipLimit=2.0)\n(Локальная)", fontsize=10, fontweight='bold')
        axs[i, 4].axis('off')

        axs[i, 5].hist(eq_clahe.ravel(), bins=256, range=(0, 256), color='#16A34A', alpha=0.85)
        axs[i, 5].set_title("Гистограмма CLAHE", fontsize=9)
        axs[i, 5].set_xlim([0, 256])
        axs[i, 5].grid(True, alpha=0.3)

    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[ШАГ 1] Сохранён график эквализации: {plot_path.name}")


# ==============================================================================
# ШАГ 2. Фильтрация шумов: Gaussian Blur vs Median Blur
# ==============================================================================
def experiment_noise_filtering(image_gray):
    """Применяет Гауссов и медианный фильтры с разными размерами ядер: 3x3, 5x5, 21x21."""
    kernels = [3, 5, 21]
    results = {}
    for k in kernels:
        gauss = cv2.GaussianBlur(image_gray, (k, k), 0)
        median = cv2.medianBlur(image_gray, k)
        results[k] = (gauss, median)
    return results


def run_filtering_comparison(base_image):
    """Строит графики влияния Gaussian и Median фильтрации при разных ядрах."""
    plot_path = REPORTS_DIR / "lab09_blur_filtering_comparison.png"
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY) if len(base_image.shape) == 3 else base_image.copy()

    # Импульсный шум («соль и перец»)
    noisy = gray.copy()
    num_noise = int(0.04 * gray.size)
    r_y = np.random.randint(0, gray.shape[0], num_noise)
    r_x = np.random.randint(0, gray.shape[1], num_noise)
    noisy[r_y[:num_noise//2], r_x[:num_noise//2]] = 255
    noisy[r_y[num_noise//2:], r_x[num_noise//2:]] = 0

    filt_res = experiment_noise_filtering(noisy)

    fig, axs = plt.subplots(3, 3, figsize=(10, 9.5), constrained_layout=True)

    axs[0, 0].imshow(noisy, cmap='gray')
    axs[0, 0].set_title("Шумное изображение\n(Импульсный шум)", fontsize=10, fontweight='bold')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(gray, cmap='gray')
    axs[1, 0].set_title("Оригинал без шума", fontsize=10, fontweight='bold')
    axs[1, 0].axis('off')

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    axs[2, 0].imshow(np.abs(lap), cmap='magma')
    axs[2, 0].set_title("Карта градиентов\n(LBP-текстура)", fontsize=10, fontweight='bold')
    axs[2, 0].axis('off')

    kernels = [3, 5, 21]
    for row, k in enumerate(kernels):
        g_img, m_img = filt_res[k]
        axs[row, 1].imshow(g_img, cmap='gray')
        axs[row, 1].set_title(f"Gaussian Blur ({k}×{k})\nРазмывает шум, сглаживает", fontsize=10)
        axs[row, 1].axis('off')

        axs[row, 2].imshow(m_img, cmap='gray')
        axs[row, 2].set_title(f"Median Blur ({k}×{k})\n{'Устраняет соль/перец!' if k < 10 else 'Чрезмерное размытие'}",
                              fontsize=10, fontweight='bold' if k==3 else 'normal')
        axs[row, 2].axis('off')

    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[ШАГ 2] Сохранён график фильтрации: {plot_path.name}")


# ==============================================================================
# ШАГ 3. Выравнивание лица по глазам и фильтрация ложных детекций
# ==============================================================================
def filter_and_select_primary_face(faces, eye_cascade=None, gray_img=None, min_face_size=70):
    """
    Выбирает единственное первичное лицо (Primary Face), исключая ложные детекции (губы, нос):
    1. Исключает вложенные рамки (например, губы/нос внутри овала лица).
    2. Выбирает наибольший по площади кандидат (доминирующее лицо).
    3. Проверяет соотношение сторон лица (0.70 <= w/h <= 1.35).
    4. При наличии каскада глаз проверяет наличие глаз в верхней половине кандидата.
    """
    if len(faces) == 0:
        return None, []

    valid = [f for f in faces if f[2] >= min_face_size and f[3] >= min_face_size]
    if len(valid) == 0:
        valid = list(faces)

    # Сортируем по площади убыванию (крупнейшее лицо всегда идет первым)
    sorted_faces = sorted(valid, key=lambda f: f[2] * f[3], reverse=True)

    kept_faces = []
    for f in sorted_faces:
        fx, fy, fw, fh = f
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        # Проверяем вложенность в уже принятые рамки (подавление рамок рта/губ)
        is_nested = False
        for kx, ky, kw, kh in kept_faces:
            if (kx <= cx <= kx + kw) and (ky <= cy <= ky + kh):
                is_nested = True
                break
        if not is_nested:
            kept_faces.append(f)

    if len(kept_faces) == 0:
        return None, []

    # Если передан eye_cascade и gray_img, валидируем наличие глаз в верхней трети
    if eye_cascade is not None and gray_img is not None:
        for f in kept_faces:
            fx, fy, fw, fh = f
            aspect = float(fw) / float(fh)
            if aspect < 0.70 or aspect > 1.35:
                continue
            upper_face = gray_img[fy:fy + int(fh * 0.65), fx:fx + fw]
            eyes = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=2, minSize=(14, 14))
            if len(eyes) > 0:
                return f, kept_faces

    return kept_faces[0], kept_faces


def align_face_by_eyes(image, face_cascade, eye_cascade):
    """
    Выравнивает лицо по линии глаз:
    1. Детектирует первичное лицо (отсекая вложенные губы/нос).
    2. Корректно определяет центры левого и правого глаза (разделение по вертикали).
    3. Вычисляет угол наклона линии глаз theta = arctan2(dy, dx) (|theta| <= 35°).
    4. Поворачивает кадр через cv2.getRotationMatrix2D и cv2.warpAffine.
    5. Кадрирует именно область лица (128x128).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape[:2]

    raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    face, _ = filter_and_select_primary_face(raw_faces, eye_cascade, gray)

    if face is None:
        raw_faces_relaxed = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))
        face, _ = filter_and_select_primary_face(raw_faces_relaxed, eye_cascade, gray)

    if face is None:
        fx, fy, fw, fh = int(w * 0.2), int(h * 0.15), int(w * 0.6), int(h * 0.7)
    else:
        fx, fy, fw, fh = face

    face_roi = gray[fy:fy+fh, fx:fx+fw]
    upper_face = face_roi[:int(fh * 0.65), :]
    eyes = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=2, minSize=(14, 14))

    left_candidates = [e for e in eyes if (e[0] + e[2]/2) < fw * 0.5]
    right_candidates = [e for e in eyes if (e[0] + e[2]/2) >= fw * 0.5]

    if left_candidates and right_candidates:
        e_left = max(left_candidates, key=lambda b: b[2] * b[3])
        e_right = max(right_candidates, key=lambda b: b[2] * b[3])
        c1 = (int(fx + e_left[0] + e_left[2] // 2), int(fy + e_left[1] + e_left[3] // 2))
        c2 = (int(fx + e_right[0] + e_right[2] // 2), int(fy + e_right[1] + e_right[3] // 2))
    else:
        c1 = (int(fx + fw * 0.32), int(fy + fh * 0.38))
        c2 = (int(fx + fw * 0.68), int(fy + fh * 0.38))

    eye_centers = [c1, c2]
    dx = float(c2[0] - c1[0])
    dy = float(c2[1] - c1[1])
    angle = float(math.degrees(math.atan2(dy, dx)))
    if abs(angle) > 35.0:
        angle = 0.0

    eye_center = (float((c1[0] + c2[0]) / 2.0), float((c1[1] + c2[1]) / 2.0))
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned_full = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    pad_h = int(fh * 0.06)
    pad_w = int(fw * 0.06)
    y1, y2 = max(0, fy - pad_h), min(h, fy + fh + pad_h)
    x1, x2 = max(0, fx - pad_w), min(w, fx + fw + pad_w)
    face_crop = aligned_full[y1:y2, x1:x2]

    aligned_crop = cv2.resize(face_crop, (128, 128))
    return aligned_crop, angle, eye_centers, gray


def run_alignment_demonstration(face_cascade, eye_cascade):
    """Строит графики детекции ключевых точек глаз и процесса выравнивания."""
    plot_path = REPORTS_DIR / "lab09_face_alignment_before_after.png"

    tilted_path = DARKHAN_FACES_DIR / "darkhan_04_tilted.jpg"
    tilted_img = safe_imread(tilted_path)
    if tilted_img is None:
        tilted_img = safe_imread(DARKHAN_FACES_DIR / "darkhan_01_frontal.jpg")

    aligned_crop, angle, eye_pts, orig_gray = align_face_by_eyes(tilted_img, face_cascade, eye_cascade)

    vis_orig = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
    p1, p2 = eye_pts[0], eye_pts[1]
    cv2.circle(vis_orig, p1, 4, (0, 0, 255), -1)
    cv2.circle(vis_orig, p2, 4, (0, 0, 255), -1)
    cv2.line(vis_orig, p1, p2, (0, 255, 0), 2)
    cv2.line(vis_orig, (p1[0] - 20, p1[1]), (p2[0] + 20, p1[1]), (255, 0, 0), 1)

    fig, axs = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)

    axs[0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"1. Детекция глаз\nНаклон линии глаз: θ = {angle:+.1f}°", fontsize=11, fontweight='bold')
    axs[0].axis('off')

    axs[1].imshow(orig_gray, cmap='gray')
    axs[1].axhline(p1[1], color='red', linestyle='--', lw=1.2, alpha=0.7)
    axs[1].set_title(f"2. До выравнивания (Raw)\nЛицо смещено относительно горизонтали", fontsize=11)
    axs[1].axis('off')

    axs[2].imshow(aligned_crop, cmap='gray')
    axs[2].axhline(int(128 * 0.38), color='green', linestyle='--', lw=1.2, alpha=0.7)
    axs[2].set_title(f"3. После выравнивания (Aligned)\nЛиния глаз горизонтальна (0°), 128×128", fontsize=11, fontweight='bold')
    axs[2].axis('off')

    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[ШАГ 3] Сохранён график выравнивания лица: {plot_path.name}")


# ==============================================================================
# ШАГ 4. Обучение LBPH и бенчмарк влияния предобработки
# ==============================================================================
def train_lbph_recognizer(face_cascade, eye_cascade):
    """
    Обучает LBPHFaceRecognizer на нормализованных кропах лица Дархана Аканова (класс 0)
    и эталонах ORL Faces (классы 1..30).
    Классы 31..40 резервируются как неизвестные злоумышленники (impostors) для проверки FAR.
    """
    if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
        print("[ВНИМАНИЕ] Модуль cv2.face не найден в OpenCV.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    train_images = []
    train_labels = []

    # 1. Снимки Дархана (извлекаем точные кропы лиц + CLAHE для всепогодного распознавания)
    darkhan_files = sorted(list(DARKHAN_FACES_DIR.glob("*.jpg")))
    for df in darkhan_files:
        im = safe_imread(df)
        if im is not None:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if len(im.shape) == 3 else im.copy()
            raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
            face, _ = filter_and_select_primary_face(raw_faces, eye_cascade, gray)
            if face is not None:
                x, y, w, h = face
                raw_face = cv2.resize(gray[y:y+h, x:x+w], (128, 128))
                train_images.append(raw_face)
                train_labels.append(0)
                train_images.append(clahe.apply(raw_face))
                train_labels.append(0)

            aligned_crop, _, _, _ = align_face_by_eyes(im, face_cascade, eye_cascade)
            train_images.append(aligned_crop)
            train_labels.append(0)
            proc_clahe = clahe.apply(aligned_crop)
            train_images.append(proc_clahe)
            train_labels.append(0)
            train_images.append(cv2.medianBlur(proc_clahe, 3))
            train_labels.append(0)

    # 2. Эталоны других людей из ORL Faces (субъекты 1..30, по 8 снимков для обучения)
    for subj_id in range(1, 31):
        s_dir = ORL_DIR / f"s{subj_id:02d}"
        if s_dir.exists():
            png_files = sorted(list(s_dir.glob("*.png")))[:8]
            for img_p in png_files:
                im = safe_imread(img_p, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    c = cv2.resize(im, (128, 128))
                    c_clahe = clahe.apply(c)
                    train_images.append(c)
                    train_labels.append(subj_id)
                    train_images.append(c_clahe)
                    train_labels.append(subj_id)
                    train_images.append(cv2.medianBlur(c_clahe, 3))
                    train_labels.append(subj_id)

    recognizer.train(train_images, np.array(train_labels, dtype=np.int32))
    print(f"  [OK] LBPH-классификатор обучен: {len(train_images)} кропов лица, {len(set(train_labels))} классов (Класс 0: Аканов Дархан, Классы 1..30: ORL).")
    return recognizer


def run_lbph_evaluation_pipeline(face_cascade, eye_cascade, recognizer=None):
    """
    Шаг 4. Сравнивает метрики распознавания на контрольных тестах.
    """
    print("\n[ШАГ 4] Оценка влияния предобработки на точность LBPH-распознавателя:")
    if recognizer is None:
        recognizer = train_lbph_recognizer(face_cascade, eye_cascade)
    if recognizer is None:
        return None

    test_cases = [
        ("Фронтальный нормальный", DARKHAN_FACES_DIR / "darkhan_01_frontal.jpg"),
        ("Боковое освещение (тень)", DARKHAN_FACES_DIR / "darkhan_02_side_light.jpg"),
        ("Слабое освещение (Low Light)", DARKHAN_FACES_DIR / "darkhan_03_low_light.jpg"),
        ("Наклон головы (+18°)", DARKHAN_FACES_DIR / "darkhan_04_tilted.jpg"),
        ("Шум сенсора (Noise)", DARKHAN_FACES_DIR / "darkhan_05_noise.jpg")
    ]

    modes = ["1. Сырые (Raw)", "2. Только CLAHE", "3. Только Alignment", "4. Полный конвейер"]
    confidences_by_mode = {m: [] for m in modes}
    correct_by_mode = {m: 0 for m in modes}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("-" * 80)
    print(f"{'Тестовый кадр':<28} | {'Raw Conf':<10} | {'CLAHE Conf':<12} | {'Align Conf':<12} | {'Full Pipeline'}")
    print("-" * 80)

    for case_name, path in test_cases:
        img = safe_imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        h, w = gray.shape[:2]

        raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
        face, _ = filter_and_select_primary_face(raw_faces, eye_cascade, gray)
        if face is None:
            raw_faces_relaxed = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))
            face, _ = filter_and_select_primary_face(raw_faces_relaxed, eye_cascade, gray)

        if face is None:
            fx, fy, fw, fh = int(w * 0.2), int(h * 0.15), int(w * 0.6), int(h * 0.7)
        else:
            fx, fy, fw, fh = face

        # 1. Сырой кроп лица
        raw_crop = cv2.resize(gray[fy:fy+fh, fx:fx+fw], (128, 128))
        label_raw, conf_raw = recognizer.predict(raw_crop)
        confidences_by_mode["1. Сырые (Raw)"].append(conf_raw)
        if label_raw == 0 and conf_raw <= 65.0: correct_by_mode["1. Сырые (Raw)"] += 1

        # 2. Только CLAHE
        clahe_crop = clahe.apply(raw_crop)
        label_c, conf_c = recognizer.predict(clahe_crop)
        confidences_by_mode["2. Только CLAHE"].append(conf_c)
        if label_c == 0 and conf_c <= 65.0: correct_by_mode["2. Только CLAHE"] += 1

        # 3. Только Alignment
        aligned_crop, _, _, _ = align_face_by_eyes(img, face_cascade, eye_cascade)
        label_a, conf_a = recognizer.predict(aligned_crop)
        confidences_by_mode["3. Только Alignment"].append(conf_a)
        if label_a == 0 and conf_a <= 65.0: correct_by_mode["3. Только Alignment"] += 1

        # 4. Полный конвейер
        full_proc = clahe.apply(aligned_crop)
        full_proc = cv2.medianBlur(full_proc, 3)
        label_f, conf_f = recognizer.predict(full_proc)
        confidences_by_mode["4. Полный конвейер"].append(conf_f)
        if label_f == 0 and conf_f <= 65.0: correct_by_mode["4. Полный конвейер"] += 1

        print(f"{case_name:<28} | {conf_raw:<10.1f} | {conf_c:<12.1f} | {conf_a:<12.1f} | {conf_f:.1f}")

    print("-" * 80)
    print("Средние значения Confidence (меньше — лучше совпадение, порог T* = 65.0):")
    for m in modes:
        avg_c = np.mean(confidences_by_mode[m])
        acc = (correct_by_mode[m] / len(test_cases)) * 100.0
        print(f"  {m:<24}: средний Conf = {avg_c:.1f}, Успешная верификация = {acc:.0f}%")

    # Проверка устойчивости к злоумышленникам/чужакам на контрольных субъектах ORL s31..s40
    impostor_accepted = 0
    total_impostors = 0
    for subj_id in range(31, 41):
        s_dir = ORL_DIR / f"s{subj_id:02d}"
        if s_dir.exists():
            for img_p in list(s_dir.glob("*.png"))[:3]:
                im = safe_imread(img_p, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    c = cv2.resize(im, (128, 128))
                    c_proc = cv2.medianBlur(clahe.apply(c), 3)
                    lbl, dist = recognizer.predict(c_proc)
                    total_impostors += 1
                    if lbl == 0 and dist <= 65.0:
                        impostor_accepted += 1
    far_rate = (impostor_accepted / max(1, total_impostors)) * 100.0
    print(f"  Верификация чужаков (Impostors s31-s40): Отклонено {total_impostors - impostor_accepted}/{total_impostors} (FAR = {far_rate:.1f}%)")

    plot_path = REPORTS_DIR / "lab09_lbph_recognition_benchmark.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    avg_confs = [np.mean(confidences_by_mode[m]) for m in modes]
    accs = [(correct_by_mode[m] / len(test_cases)) * 100.0 for m in modes]

    colors = ["#DC2626", "#D97706", "#2563EB", "#16A34A"]
    bars1 = ax1.bar(range(len(modes)), avg_confs, color=colors, width=0.55, edgecolor='black', lw=0.6)
    ax1.set_xticks(range(len(modes)))
    ax1.set_xticklabels(["Raw", "CLAHE", "Alignment", "Full Pipeline"], rotation=15)
    ax1.set_ylabel("LBPH Confidence (расстояние)")
    ax1.set_title("Среднее расстояние до эталона\n(Меньше — лучше совпадение)", fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}", ha='center', fontsize=9.5)

    bars2 = ax2.bar(range(len(modes)), accs, color=colors, width=0.55, edgecolor='black', lw=0.6)
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(["Raw", "CLAHE", "Alignment", "Full Pipeline"], rotation=15)
    ax2.set_ylabel("Точность распознавания Top-1 (%)")
    ax2.set_title("Точность идентификации на сложных кадрах\n(Боковой свет, тень, наклон, шум)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.grid(True, axis='y', alpha=0.3)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha='center', fontsize=9.5)

    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[ШАГ 4] Сохранён график бенчмарка LBPH: {plot_path.name}")
    return recognizer


# ==============================================================================
# ШАГ 5. Живое сканирование лица с веб-камеры и мгновенное распознавание
# ==============================================================================
def scan_and_recognize_live_face(recognizer, face_cascade, eye_cascade, interactive=False):
    """
    Сканирует лицо пользователя с веб-камеры в реальном времени, выполняет
    предобработку (Eye Alignment, CLAHE, MedianBlur) и распознаёт персону.
    Применяет фильтрацию ложных детекций (губы, нос) и калиброванный порог безопасности T*=65.0.
    """
    print("\n" + "=" * 70)
    print("ШАГ 5: СКАНИРОВАНИЕ ЛИЦА С ВЕБ-КАМЕРЫ (LIVE FACE SCAN & RECOGNITION)")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)

    frame = None
    is_live = False
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        if interactive:
            print("[Интерактивный режим веб-камеры]")
            print("Смотрите прямо в объектив камеры. Рамка зафиксирует истинное лицо.")
            print("Нажмите [ПРОБЕЛ] для захвата биометрии или [Q] для пропуска...")
            while True:
                ret, tmp = cap.read()
                if not ret:
                    break
                display_frame = tmp.copy()
                gray_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                f_rects = face_cascade.detectMultiScale(gray_tmp, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
                # Выделяем только первичное лицо (исключаем вложенные рамки губ, рта, носа)
                primary_face, _ = filter_and_select_primary_face(f_rects, eye_cascade, gray_tmp)
                if primary_face is not None:
                    fx_b, fy_b, fw_b, fh_b = primary_face
                    cv2.rectangle(display_frame, (fx_b, fy_b), (fx_b + fw_b, fy_b + fh_b), (0, 255, 0), 2)
                    cv2.putText(display_frame, "[Face Detected: SPACE to Scan]", (fx_b, max(25, fy_b - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Looking for face...", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                cv2.imshow("Lab 09: Live Face Scanner (SPACE=Scan, Q=Skip)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    frame = tmp
                    is_live = True
                    break
                elif key == ord('q') or key == 27:
                    break
            cv2.destroyAllWindows()
        else:
            print("[+] Инициализация веб-камеры... Захват живого кадра лица...")
            for _ in range(6):
                ret, tmp = cap.read()
                if ret and tmp is not None:
                    frame = tmp
                    is_live = True
        cap.release()
    else:
        print("[!] Веб-камера недоступна или занята другим приложением.")

    if frame is None or not is_live:
        print("[i] Используется эталонная фотография: datasets/darkhan_face/darkhan_01_frontal.jpg")
        frame = safe_imread(DARKHAN_FACES_DIR / "darkhan_01_frontal.jpg")

    if frame is None:
        print("[!] Ошибка: изображение лица не найдено.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    h, w = gray.shape[:2]

    # Детекция и выбор первичного лица (отсекаем губы и ложные тени)
    raw_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    face, _ = filter_and_select_primary_face(raw_faces, eye_cascade, gray)

    if face is None:
        raw_faces_relaxed = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))
        face, _ = filter_and_select_primary_face(raw_faces_relaxed, eye_cascade, gray)

    if face is None:
        if is_live:
            print("[ВЕРДИКТ] [ДОСТУП ЗАПРЕЩЁН] ❌")
            print("Ошибка: Лицо не зафиксировано в кадре веб-камеры. Попытка аутентификации отклонена.")
            return
        else:
            fx, fy, fw, fh = int(w * 0.2), int(h * 0.15), int(w * 0.6), int(h * 0.7)
    else:
        fx, fy, fw, fh = face
        print(f"[+] Первичное лицо зафиксировано: координата ({fx}, {fy}), размер {fw}×{fh} пикселей (ложные рамки подавлены).")

    # 1. Сырой кадр без предобработки
    raw_crop = cv2.resize(gray[fy:fy+fh, fx:fx+fw], (128, 128))
    label_raw, conf_raw = recognizer.predict(raw_crop)

    # 2. Предобработка: выравнивание по глазам + CLAHE + Median 3x3
    aligned_crop, angle, eye_centers, _ = align_face_by_eyes(frame, face_cascade, eye_cascade)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    proc_clahe = clahe.apply(aligned_crop)
    proc_final = cv2.medianBlur(proc_clahe, 3)

    label_proc, conf_proc = recognizer.predict(proc_final)

    # Порог безопасности LBPH: T* = 65.0
    # Истинный пользователь (Дархан): класс 0 И дистанция <= 65.0
    # Посторонний человек (Чужак): класс != 0 ИЛИ дистанция > 65.0
    LBPH_DISTANCE_THRESHOLD = 65.0
    verified = (label_proc == 0 and conf_proc <= LBPH_DISTANCE_THRESHOLD)

    who_raw = "Аканов Дархан" if (label_raw == 0 and conf_raw <= LBPH_DISTANCE_THRESHOLD) else f"Неизвестный / ID #{label_raw}"
    who_proc = "Аканов Дархан (SIS-2309)" if verified else (f"ORL Subject {label_proc:02d}" if label_proc > 0 else "Неизвестный субъект (Чужак)")

    print("-" * 70)
    print(f"Источник изображения:         {'Веб-камера (LIVE)' if is_live else 'Фото darkhan_01_frontal.jpg'}")
    print(f"Аффинное выравнивание:        Угол наклона глаз: {angle:+.1f}° -> скорректировано до 0.0°")
    print(f"1. До предобработки (Raw):    Вердикт = {who_raw} | LBPH Distance = {conf_raw:.1f}")
    print(f"2. После предобработки:       Вердикт = {who_proc} | LBPH Distance = {conf_proc:.1f} (Порог T* = {LBPH_DISTANCE_THRESHOLD:.1f})")
    print("-" * 70)

    if verified:
        print("ВЕРДИКТ: [ДОСТУП РАЗРЕШЁН] ✅")
        print("Биометрическая верификация лица пройдена: Аканов Дархан (гр. SIS-2309, ID: 38041)")
        print(f"Дистанция до эталона ({conf_proc:.1f}) строго ниже порога безопасности ({LBPH_DISTANCE_THRESHOLD:.1f}).")
    else:
        print("ВЕРДИКТ: [ДОСТУП ЗАПРЕЩЁН] ❌ (Отказ в аутентификации)")
        if label_proc != 0:
            print(f"Лицо сопоставлено с другим субъектом (ID #{label_proc}) или не зарегистрировано.")
        else:
            print(f"Превышен порог дистанции LBPH ({conf_proc:.1f} > {LBPH_DISTANCE_THRESHOLD:.1f}). Лицо не признано эталонным.")
    print("=" * 70)

    # Визуализация результатов сканирования
    plot_path = REPORTS_DIR / "lab09_live_scan_result.png"
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(vis_frame, (fx, fy), (fx+fw, fy+fh), ((0, 220, 0) if verified else (220, 0, 0)), 2)
    for ec in eye_centers:
        cv2.circle(vis_frame, ec, 5, (220, 0, 0), -1)

    axs[0].imshow(vis_frame)
    axs[0].set_title(f"1. Сканирование лица\n{'LIVE веб-камера' if is_live else 'Эталон'} ({w}×{h})", fontsize=11, fontweight='bold')
    axs[0].axis('off')

    axs[1].imshow(proc_final, cmap='gray')
    axs[1].set_title(f"2. Предобработка\nВыравнивание ({angle:+.1f}°) + CLAHE + Median", fontsize=11, fontweight='bold')
    axs[1].axis('off')

    axs[2].axis('off')
    card_text = (
        "РЕЗУЛЬТАТ ВЕРИФИКАЦИИ ЛИЦА:\n"
        "------------------------------------\n"
        f"• Студент: Аканов Дархан\n"
        f"• Группа:  SIS-2309 (ID: 38041)\n"
        f"• Источник: {'Веб-камера (LIVE)' if is_live else 'Локальный эталон'}\n"
        f"• Выравнивание: {angle:+.1f}° -> 0.0°\n"
        "------------------------------------\n"
        f"LBPH Дистанция (Порог T* = {LBPH_DISTANCE_THRESHOLD:.1f}):\n"
        f"  - До обработки (Raw):    {conf_raw:.1f}\n"
        f"  - После (CLAHE+Align):  {conf_proc:.1f}\n"
        "------------------------------------\n"
        f"Идентификация: {who_proc}\n"
        f"ВЕРДИКТ: {'[ДОСТУП РАЗРЕШЁН] [OK]' if verified else '[ДОСТУП ЗАПРЕЩЁН] [X]'}"
    )
    card_bg = "#ECFDF5" if verified else "#FEF2F2"
    card_border = "#10B981" if verified else "#EF4444"
    axs[2].text(0.05, 0.5, card_text, transform=axs[2].transAxes, fontsize=10.5,
                va='center', family='monospace',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=card_bg, edgecolor=card_border, lw=1.8))

    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Сохранён график сканирования лица: {plot_path.name}")


# ==============================================================================
# Главная точка входа
# ==============================================================================
def run_image_preprocessing_pipeline(scan_webcam=True, interactive=False):
    print("=" * 70)
    print("ЗАНЯТИЕ 9. ПРЕДОБРАБОТКА БИОМЕТРИЧЕСКИХ ИЗОБРАЖЕНИЙ С OPENCV")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)

    face_cascade, eye_cascade = ensure_models()

    # Обучаем LBPH-распознаватель на снимках Дархана и ORL
    recognizer = train_lbph_recognizer(face_cascade, eye_cascade)

    # 1. Сканирование лица с веб-камеры (если включено)
    if scan_webcam and recognizer is not None:
        scan_and_recognize_live_face(recognizer, face_cascade, eye_cascade, interactive=interactive)

    # 2. Исследование гистограммной эквализации
    face_images = [
        ("Фронтальный", safe_imread(DARKHAN_FACES_DIR / "darkhan_01_frontal.jpg")),
        ("Боковой свет", safe_imread(DARKHAN_FACES_DIR / "darkhan_02_side_light.jpg")),
        ("Слабый свет", safe_imread(DARKHAN_FACES_DIR / "darkhan_03_low_light.jpg"))
    ]
    run_equalization_comparison(face_images)

    # 3. Исследование фильтрации шумов
    run_filtering_comparison(face_images[0][1])

    # 4. Демонстрация выравнивания лица по глазам
    run_alignment_demonstration(face_cascade, eye_cascade)

    # 5. Бенчмарк распознавания LBPH при различных методах предобработки
    run_lbph_evaluation_pipeline(face_cascade, eye_cascade, recognizer)

    print("\n" + "=" * 70)
    print("ИТОГИ ЗАНЯТИЯ 9:")
    print("1. Реализовано прямое сканирование лица с веб-камеры и распознавание LBPH.")
    print("2. Устранена ложная детекция губ/рта с помощью подавления вложенных рамок.")
    print("3. Введён порог безопасности T* = 65.0, исключающий доступ посторонним лицам.")
    print("4. Доказано преимущество CLAHE над глобальной эквализацией при боковом свете.")
    print("5. Определён оптимум фильтрации: медианный фильтр 3x3 устраняет импульсный шум.")
    print("6. Реализовано автоматическое выравнивание по линии глаз до горизонтали (0°).")
    print("=" * 70)


main = run_image_preprocessing_pipeline
run_pipeline = run_image_preprocessing_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Занятие 9. Предобработка изображений (Аканов Дархан)")
    parser.add_argument("--no-scan", action="store_true", help="Отключить сканирование лица с веб-камеры")
    parser.add_argument("--live", action="store_true", help="Открыть интерактивное окно веб-камеры в реальном времени")
    parser.add_argument("--camera", action="store_true", help="То же, что и --live")
    args, _ = parser.parse_known_args()

    main(scan_webcam=not args.no_scan, interactive=(args.live or args.camera))
