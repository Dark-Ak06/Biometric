# load_dataset.py — загрузка, визуализация и анализ биометрических датасетов (Занятие 4)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

plt.style.use("default")
os.makedirs("datasets", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def prepare_orl_faces_dataset(root_dir="datasets/orl_faces"):
    """
    Шаг 1. Подготовка реального датасета лиц ORL / AT&T Faces (40 субъектов, по 10 образцов = 400 изображений).
    Сохраняет изображения в иерархию папок datasets/orl_faces/s01 ... s40.
    """
    if os.path.exists(root_dir) and len(os.listdir(root_dir)) >= 40:
        return root_dir

    print("Загрузка датасета ORL (Olivetti/AT&T) Faces...")
    olivetti = fetch_olivetti_faces()
    images = olivetti.images  # (400, 64, 64), float32 [0..1]
    targets = olivetti.target  # 0..39

    os.makedirs(root_dir, exist_ok=True)
    counts = {}
    for img_float, subj_id in zip(images, targets):
        subj_folder = os.path.join(root_dir, f"s{subj_id + 1:02d}")
        os.makedirs(subj_folder, exist_ok=True)
        counts[subj_id] = counts.get(subj_id, 0) + 1
        # Приводим к классическому пропорциональному разрешению ORL 92x112 (ширина 92, высота 112), uint8
        img_uint8 = (img_float * 255).astype(np.uint8)
        img_resized = cv2.resize(img_uint8, (92, 112), interpolation=cv2.INTER_CUBIC)
        fname = os.path.join(subj_folder, f"{counts[subj_id]:02d}.png")
        cv2.imwrite(fname, img_resized)

    print(f"Датасет ORL Faces сохранён в {root_dir} (40 субъектов, 400 изображений 112x92).")
    return root_dir


def _synthesize_fingerprint_impression(rng_subj, rng_imp, size=(128, 128)):
    """
    Генерирует реалистичное изображение отпечатка пальца (FVC-стиль):
    индивидуальная топология папиллярных линий субъекта (центр завитка/петли, частота гребней, минуции)
    + искажения конкретного прокатывания/прижима (сдвиг, поворот, влажность/сухость кожи, шум сенсора).
    """
    h, w = size
    y, x = np.mgrid[-1.0:1.0:complex(0, h), -1.0:1.0:complex(0, w)]

    # Индивидуальные параметры папиллярного узора субъекта
    core_x = rng_subj.uniform(-0.18, 0.18)
    core_y = rng_subj.uniform(-0.18, 0.18)
    delta_x = rng_subj.uniform(-0.45, 0.45)
    delta_y = rng_subj.uniform(0.25, 0.60)
    freq = rng_subj.uniform(58.0, 74.0)
    pattern_type = rng_subj.choice(["whorl", "loop_left", "loop_right", "arch"])
    phase_offset = rng_subj.uniform(0, 2 * np.pi)

    # Искажения конкретного образца (impression): поворот и сдвиг пальца на сканере
    angle = rng_imp.uniform(-0.22, 0.22)  # до +-12.5 градусов
    tx = rng_imp.uniform(-0.08, 0.08)
    ty = rng_imp.uniform(-0.08, 0.08)

    xr = (x - tx) * np.cos(angle) - (y - ty) * np.sin(angle)
    yr = (x - tx) * np.sin(angle) + (y - ty) * np.cos(angle)

    dx = xr - core_x
    dy = yr - core_y
    r = np.sqrt(dx**2 + dy**2 + 1e-4)
    theta = np.arctan2(dy, dx)

    dx_d = xr - delta_x
    dy_d = yr - delta_y
    theta_d = np.arctan2(dy_d, dx_d)

    if pattern_type == "whorl":
        field = freq * r + 2.0 * theta - 1.0 * theta_d + phase_offset
    elif pattern_type == "loop_left":
        field = freq * (r + 0.35 * yr) + 1.0 * theta - 1.0 * theta_d + phase_offset
    elif pattern_type == "loop_right":
        field = freq * (r + 0.35 * yr) - 1.0 * theta + 1.0 * theta_d + phase_offset
    else:  # arch
        field = freq * (yr + 0.38 * np.exp(-3.2 * xr**2)) + phase_offset

    # Добавляем индивидуальные локальные разрывы/бифуркации (минуции)
    for _ in range(10):
        mx = rng_subj.uniform(-0.48, 0.48)
        my = rng_subj.uniform(-0.55, 0.55)
        field += 1.4 * np.exp(-((xr - mx) ** 2 + (yr - my) ** 2) / 0.004)

    ridges = np.cos(field)

    # Моделируем сухой / нормальный / влажный прижим пальца (порог бинаризации и контраст)
    pressure_bias = rng_imp.uniform(-0.18, 0.18)
    ridges = np.tanh(3.2 * (ridges + pressure_bias))

    # Эллиптическая маска подушечки пальца при контакте с оптической призмой
    ellipse_r = ((xr / rng_imp.uniform(0.68, 0.80)) ** 2 + (yr / rng_imp.uniform(0.84, 0.95)) ** 2)
    mask = np.clip(1.28 - ellipse_r, 0.0, 1.0) ** 1.6

    # Фон сканера светлый (245), гребни тёмные (25..70)
    img = 244.0 - mask * ((ridges + 1.0) * 0.5 * 195.0 + 18.0)

    # Высокочастотный шум пор и оптического сенсора
    noise = rng_imp.normal(0, rng_imp.uniform(5.0, 12.0), size=size)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def prepare_fingerprint_dataset(root_dir="datasets/fingerprints_fvc", n_subjects=25):
    """
    Подготавливает второй биометрический датасет — отпечатки пальцев (FVC-style)
    с реалистичным дисбалансом классов (от 6 до 14 отпечатков на субъекта),
    чтобы продемонстрировать разницу между сбалансированным и несбалансированным датасетами.
    """
    if os.path.exists(root_dir) and len(os.listdir(root_dir)) >= n_subjects:
        return root_dir

    print("Подготовка датасета отпечатков пальцев (FVC-style)...")
    os.makedirs(root_dir, exist_ok=True)
    rng_global = np.random.default_rng(2026)

    total = 0
    for subj_idx in range(1, n_subjects + 1):
        subj_folder = os.path.join(root_dir, f"fp_{subj_idx:02d}")
        os.makedirs(subj_folder, exist_ok=True)
        rng_subj = np.random.default_rng(1000 + subj_idx)

        # Различное число образцов на субъекта (от 6 до 14) для анализа баланса классов
        n_samples = int(rng_global.integers(6, 15))
        for imp_idx in range(1, n_samples + 1):
            rng_imp = np.random.default_rng(10000 * subj_idx + imp_idx)
            fp_img = _synthesize_fingerprint_impression(rng_subj, rng_imp, size=(128, 128))
            # Переинициализируем генератор субъекта, чтобы базовый узор оставался тем же для всех отпечатков субъекта!
            rng_subj = np.random.default_rng(1000 + subj_idx)
            cv2.imwrite(os.path.join(subj_folder, f"imp_{imp_idx:02d}.png"), fp_img)
            total += 1

    print(f"Датасет отпечатков сохранён в {root_dir} ({n_subjects} субъектов, {total} изображений 128x128).")
    return root_dir


# ==================== ШАГ 2. ЗАГРУЗЧИК ДАТАСЕТА ====================
def load_face_dataset(root_dir):
    """
    Читает все изображения из подпапок субъектов в массив NumPy и формирует метки классов (subject_id).
    """
    images, labels = [], []
    for subject_id, subject_dir in enumerate(sorted(os.listdir(root_dir))):
        subject_path = os.path.join(root_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        for fname in sorted(os.listdir(subject_path)):
            img = cv2.imread(os.path.join(subject_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(subject_id)
    try:
        return np.array(images, dtype=np.uint8), np.array(labels)
    except ValueError:
        arr = np.empty(len(images), dtype=object)
        arr[:] = images
        return arr, np.array(labels)


# ==================== ШАГ 3. ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ 4x4 ====================
def show_samples(images, labels, n=16, output_path="reports/dataset_samples.png", title="Случайные образцы датасета"):
    """
    Выводит сеткой 4x4 случайные изображения с подписями subject_id через Matplotlib.
    """
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.5), dpi=150)
    idxs = rng.choice(len(images), n, replace=False)
    for ax, i in zip(axes.flat, idxs):
        ax.imshow(np.asarray(images[i], dtype=np.uint8), cmap="gray")
        ax.set_title(f"ID {labels[i]}", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Визуализация 4x4 сохранена: {output_path}")


# ==================== ШАГ 4. СТАТИСТИКА ДАТАСЕТА И ГИСТОГРАММЫ ====================
def compute_dataset_statistics(images, labels, name="Dataset"):
    """
    Вычисляет и выводит статистику биометрического датасета:
    число субъектов, число образцов на субъекта (мин/макс/среднее), разрешение изображений.
    """
    unique_subjects, counts = np.unique(labels, return_counts=True)
    shapes = set(img.shape for img in images)
    pixel_values = np.concatenate([img.ravel() for img in images])

    stats = {
        "name": name,
        "total_images": len(images),
        "n_subjects": len(unique_subjects),
        "min_per_subject": int(counts.min()),
        "max_per_subject": int(counts.max()),
        "mean_per_subject": float(counts.mean()),
        "std_per_subject": float(counts.std()),
        "resolutions": [f"{w}x{h} (H={h}, W={w})" for (h, w) in shapes],
        "mean_intensity": float(pixel_values.mean()),
        "std_intensity": float(pixel_values.std()),
        "unique_subjects": unique_subjects,
        "counts": counts,
    }

    print("=" * 65)
    print(f"СТАТИСТИКА ДАТАСЕТА: {name}")
    print("=" * 65)
    print(f"  Всего изображений:            {stats['total_images']}")
    print(f"  Число уникальных субъектов:   {stats['n_subjects']}")
    print(f"  Образцов на субъекта:         min={stats['min_per_subject']}, "
          f"max={stats['max_per_subject']}, mean={stats['mean_per_subject']:.2f} ± {stats['std_per_subject']:.2f}")
    print(f"  Разрешение изображений:       {', '.join(stats['resolutions'])}")
    print(f"  Средняя яркость пикселей:     {stats['mean_intensity']:.1f} ± {stats['std_intensity']:.1f}")
    print("=" * 65)
    return stats


def plot_class_distributions(stats_faces, stats_fp, output_path="reports/dataset_class_distribution.png"):
    """
    Строит гистограммы распределения образцов по классам (subject_id) для обоих датасетов:
    1) ORL Faces (строго сбалансированный датасет: 40 классов по 10 изображений)
    2) Fingerprints FVC (несбалансированный датасет: 25 классов от 6 до 14 изображений)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=150)

    # 1. ORL Faces
    ax1.bar(
        stats_faces["unique_subjects"], stats_faces["counts"],
        color="#1976d2", edgecolor="#0d47a1", alpha=0.85, width=0.75
    )
    ax1.axhline(stats_faces["mean_per_subject"], color="#d32f2f", linestyle="--", lw=2,
                label=f"Среднее = {stats_faces['mean_per_subject']:.1f} обр./субъект (Идеальный баланс)")
    ax1.set_title("Распределение изображений по классам — Датасет лиц ORL Faces (40 субъектов, 400 изображений)",
                  fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("ID субъекта (subject_id)", fontsize=10)
    ax1.set_ylabel("Число образцов", fontsize=10)
    ax1.set_xticks(stats_faces["unique_subjects"])
    ax1.set_ylim(0, 14)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2. Fingerprints FVC
    ax2.bar(
        stats_fp["unique_subjects"], stats_fp["counts"],
        color="#2e7d32", edgecolor="#1b5e20", alpha=0.85, width=0.7
    )
    ax2.axhline(stats_fp["mean_per_subject"], color="#d32f2f", linestyle="--", lw=2,
                label=f"Среднее = {stats_fp['mean_per_subject']:.1f} ± {stats_fp['std_per_subject']:.1f} обр./субъект (Дисбаланс: min={stats_fp['min_per_subject']}, max={stats_fp['max_per_subject']})")
    ax2.set_title("Распределение изображений по классам — Датасет отпечатков Fingerprints FVC (25 субъектов, дисбаланс классов)",
                  fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("ID субъекта (subject_id)", fontsize=10)
    ax2.set_ylabel("Число образцов", fontsize=10)
    ax2.set_xticks(stats_fp["unique_subjects"])
    ax2.set_ylim(0, 17)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Гистограмма распределения по классам сохранена: {output_path}")


# ==================== ШАГ 5. TRAIN/TEST SPLIT ПО СУБЪЕКТАМ ====================
def split_by_subjects(X, y, test_size=0.2, random_state=42):
    """
    Выполняет корректное разбиение биометрического датасета ПО СУБЪЕКТАМ (Subject-Disjoint Split / Open-Set),
    гарантируя, что изображения одного и того же человека не попадут одновременно в train и test.
    """
    unique_subjects = np.unique(y)
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state, shuffle=True
    )

    train_mask = np.isin(y, train_subjects)
    test_mask = np.isin(y, test_subjects)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Строгая проверка отсутствия пересечения субъектов (Data Leakage Check)
    overlap = set(y_train).intersection(set(y_test))
    assert len(overlap) == 0, f"Ошибка! Пересечение субъектов между train и test: {overlap}"

    print("\nРЕЗУЛЬТАТ РАЗБИЕНИЯ ПО СУБЪЕКТАМ (80/20 Subject-Disjoint Split):")
    print(f"  Всего субъектов:   {len(unique_subjects)} -> Train: {len(train_subjects)} суб., Test: {len(test_subjects)} суб.")
    print(f"  Всего изображений: {len(X)} -> X_train: {len(X_train)} изобр., X_test: {len(X_test)} изобр.")
    print(f"  Subject IDs в Test: {sorted(test_subjects.tolist())}")
    print(f"  Пересечение субъектов (Train ∩ Test): {len(overlap)} (утечка данных отсутствует!)")

    return X_train, X_test, y_train, y_test, train_subjects, test_subjects


def plot_split_comparison(X, y, train_subjects, test_subjects, output_path="reports/subject_split_comparison.png"):
    """
    Визуализирует разницу между:
    - Некорректным случайным разбиением ПО ИЗОБРАЖЕНИЯМ (Image-level Split -> Data Leakage)
    - Корректным разбиением ПО СУБЪЕКТАМ (Subject-level Split -> Open-Set Verification)
    """
    n_subjects = len(np.unique(y))
    n_per_subj = 10

    # 1. Случайный split по изображениям (80/20)
    idx_all = np.arange(len(y))
    idx_train_img, idx_test_img = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y)
    grid_img_split = np.zeros((n_subjects, n_per_subj))  # 0 = Train, 1 = Test
    for idx in idx_test_img:
        s = y[idx]
        sample_num = idx % n_per_subj
        grid_img_split[s, sample_num] = 1

    # 2. Корректный split по субъектам (80/20)
    grid_subj_split = np.zeros((n_subjects, n_per_subj))
    for s in test_subjects:
        grid_subj_split[s, :] = 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)
    cmap = plt.cm.colors.ListedColormap(["#42a5f5", "#ef5350"])

    # Левый график: Split по изображениям
    ax1.imshow(grid_img_split, aspect="auto", cmap=cmap, interpolation="nearest")
    ax1.set_title("[X] НЕКОРРЕКТНО: Split по изображениям (Closed-Set / Утечка)\n"
                  "Фото каждого субъекта есть и в Train (синий), и в Test (красный)!\n"
                  "Пересечение субъектов: 40 из 40 (100% Data Leakage)",
                  fontsize=10.5, fontweight="bold", color="#b71c1c")
    ax1.set_xlabel("Номер образца субъекта (1..10)", fontsize=10)
    ax1.set_ylabel("ID субъекта (0..39)", fontsize=10)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(range(1, 11))

    # Правый график: Split по субъектам
    ax2.imshow(grid_subj_split, aspect="auto", cmap=cmap, interpolation="nearest")
    ax2.set_title("[OK] КОРРЕКТНО: Split по субъектам 80/20 (Open-Set Verification)\n"
                  "32 субъекта целиком в Train (синий), 8 новых субъектов целиком в Test (красный)\n"
                  "Пересечение субъектов: 0 из 40 (Честная оценка обобщения)",
                  fontsize=10.5, fontweight="bold", color="#1b5e20")
    ax2.set_xlabel("Номер образца субъекта (1..10)", fontsize=10)
    ax2.set_ylabel("ID субъекта (0..39)", fontsize=10)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(range(1, 11))

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#42a5f5", edgecolor="#0d47a1", label="Train (Обучающая выборка — 80%)"),
        Patch(facecolor="#ef5350", edgecolor="#b71c1c", label="Test (Тестовая выборка — 20%)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Диаграмма сравнения стратегий разбиения сохранена: {output_path}")


if __name__ == "__main__":
    # 1. Подготовка датасетов (Лица ORL + Отпечатки пальцев FVC)
    orl_dir = prepare_orl_faces_dataset("datasets/orl_faces")
    fp_dir = prepare_fingerprint_dataset("datasets/fingerprints_fvc")

    # 2. Загрузка обоих датасетов
    X_faces, y_faces = load_face_dataset(orl_dir)
    X_fp, y_fp = load_face_dataset(fp_dir)

    print(f"\n[ORL Faces] Всего изображений: {len(X_faces)}, субъектов: {len(set(y_faces))}")
    print(f"[Fingerprints FVC] Всего изображений: {len(X_fp)}, субъектов: {len(set(y_fp))}\n")

    # 3. Визуализация 4x4 образцов каждого датасета
    show_samples(X_faces, y_faces, n=16, output_path="reports/dataset_samples.png",
                 title="Датасет лиц ORL Faces — 16 случайных образцов (4x4)")
    show_samples(X_fp, y_fp, n=16, output_path="reports/fingerprint_samples.png",
                 title="Датасет отпечатков Fingerprints FVC — 16 случайных образцов (4x4)")

    # 4. Статистика и гистограммы распределения по классам
    stats_faces = compute_dataset_statistics(X_faces, y_faces, name="ORL / AT&T Faces (Лица)")
    stats_fp = compute_dataset_statistics(X_fp, y_fp, name="Fingerprints FVC (Отпечатки пальцев)")
    plot_class_distributions(stats_faces, stats_fp, output_path="reports/dataset_class_distribution.png")

    # 5. Корректный train/test split по субъектам (80/20)
    X_tr, X_te, y_tr, y_te, tr_subj, te_subj = split_by_subjects(X_faces, y_faces, test_size=0.2, random_state=42)
    X_fp_tr, X_fp_te, y_fp_tr, y_fp_te, _, _ = split_by_subjects(X_fp, y_fp, test_size=0.2, random_state=42)

    plot_split_comparison(X_faces, y_faces, tr_subj, te_subj, output_path="reports/subject_split_comparison.png")
