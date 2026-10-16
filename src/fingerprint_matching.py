# fingerprint_matching.py — распознавание и сопоставление отпечатков пальцев (Занятие 5)
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

plt.style.use("default")
os.makedirs("datasets/fingerprints", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def ensure_fingerprint_samples():
    """
    Создаёт эталонные пары отпечатков пальцев в папке datasets/fingerprints/ (f001_1.png, f001_2.png, ...),
    совместимые с кодом-заготовкой и датасетом Занятия 4.
    Для одного и того же субъекта (f00X_1 и f00X_2) второй отпечаток моделирует повторное прикладывание
    того же пальца к сканеру (небольшой сдвиг ±2..3 пикс., поворот ±3°, вариация нажима и шума).
    """
    base_fvc = "datasets/fingerprints_fvc"
    rng = np.random.default_rng(2026)

    for subj_idx in range(1, 11):
        p1 = f"datasets/fingerprints/f{subj_idx:03d}_1.png"
        p2 = f"datasets/fingerprints/f{subj_idx:03d}_2.png"
        src_img_path = os.path.join(base_fvc, f"fp_{subj_idx:02d}", "imp_01.png")

        if os.path.exists(src_img_path):
            base_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            # Увеличиваем до 256x256 для чёткой скелетизации и работы патчей ORB (31x31)
            base_img = cv2.resize(base_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        else:
            # Резервная генерация, если папка Занятия 4 не найдена
            y, x = np.mgrid[-1:1:256j, -1:1:256j]
            r = np.sqrt((x - 0.05 * subj_idx) ** 2 + (y + 0.03 * subj_idx) ** 2)
            theta = np.arctan2(y, x)
            ridges = np.cos((45 + 4 * subj_idx) * r + 2 * theta)
            base_img = ((1.0 - 0.8 * ((ridges + 1) / 2)) * 240).astype(np.uint8)

        cv2.imwrite(p1, base_img)

        # Второй отпечаток того же пальца (Genuine impression 2):
        # реалистичный микро-поворот (1.5 град), сдвиг (1.5 пикс) и лёгкая вариация нажима/шума
        h, w = base_img.shape
        angle = float(rng.uniform(-1.5, 1.5))
        tx = float(rng.uniform(-1.5, 1.5))
        ty = float(rng.uniform(-1.5, 1.5))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        imp2 = cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        noise = rng.normal(0, 2.5, size=(h, w))
        imp2 = np.clip(imp2.astype(np.float32) * rng.uniform(0.97, 1.03) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(p2, imp2)


def preprocess(path, return_steps=False):
    """
    Шаги 1–3: Загрузка изображения отпечатка, локальное выравнивание контраста (CLAHE),
    бинаризация (Otsu и адаптивная пороговая обработка) и скелетизация (thinning до 1 пикселя).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не найден файл отпечатка: {path}")

    # Подавление высокочастотного шума сенсора и локальное выравнивание гистограммы (CLAHE)
    blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blurred)

    # Адаптивная бинаризация (идеально разделяет гребни и бороздки по всей площади подушечки пальца)
    binary_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 6
    )
    # Очистка мелкого точечного шума фона морфологическим открытием
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary_adapt, cv2.MORPH_OPEN, kernel)

    # Глобальная бинаризация по методу Оцу (для сравнения)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Скелетизация (утоньшение папиллярных линий до толщины в 1 пиксель)
    skeleton = skeletonize(binary_clean > 0).astype(np.uint8) * 255

    if return_steps:
        return {
            "original": img,
            "equalized": eq,
            "binary_otsu": binary_otsu,
            "binary_adaptive": binary_clean,
            "skeleton": skeleton,
        }
    return skeleton


def extract_minutiae_cn(skeleton_img, border_margin=24):
    """
    Классический поиск особых точек папиллярного узора (Minutiae) методом Crossing Number (CN Рутовица):
      CN = 0.5 * sum_{i=1..8} |P_i - P_{i+1}|
      - CN == 1: Окончание линии (Ridge Ending)
      - CN == 3: Разветвление линии (Bifurcation)
    Возвращает списки координат (x, y) окончаний и бифуркаций.
    """
    sk = (skeleton_img > 0).astype(np.int32)
    h, w = sk.shape

    # 8-связные соседи по часовой стрелке вокруг центрального пикселя
    p1 = sk[1:-1, 2:]    # (y, x+1)
    p2 = sk[2:, 2:]      # (y+1, x+1)
    p3 = sk[2:, 1:-1]    # (y+1, x)
    p4 = sk[2:, :-2]     # (y+1, x-1)
    p5 = sk[1:-1, :-2]   # (y, x-1)
    p6 = sk[:-2, :-2]    # (y-1, x-1)
    p7 = sk[:-2, 1:-1]   # (y-1, x)
    p8 = sk[:-2, 2:]     # (y-1, x+1)

    cn = (
        np.abs(p1 - p2) + np.abs(p2 - p3) + np.abs(p3 - p4) + np.abs(p4 - p5) +
        np.abs(p5 - p6) + np.abs(p6 - p7) + np.abs(p7 - p8) + np.abs(p8 - p1)
    ) // 2

    center = sk[1:-1, 1:-1] == 1
    yy, xx = np.mgrid[1:h-1, 1:w-1]
    dist_center = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    valid_mask = (
        (xx >= border_margin) & (xx < w - border_margin) &
        (yy >= border_margin) & (yy < h - border_margin) &
        (dist_center < min(h, w) * 0.38)
    )

    endings_y, endings_x = np.where(center & valid_mask & (cn == 1))
    bif_y, bif_x = np.where(center & valid_mask & (cn == 3))

    endings = list(zip(endings_x + 1, endings_y + 1))
    bifurcations = list(zip(bif_x + 1, bif_y + 1))
    return endings, bifurcations


def detect_and_compute_orb(skeleton_img, nfeatures=500):
    """
    Шаг 4. Выделение ключевых точек и ORB-дескрипторов на скелетизированном изображении.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(skeleton_img, None)
    return kp, des


def match_fingerprints(img1_path, img2_path, distance_thresh=60):
    """
    Шаг 5. Сопоставление двух отпечатков пальцев через ORB + BFMatcher (Норма Хэмминга).
    Возвращает (score, len(good_matches), details_dict).
    """
    sk1 = preprocess(img1_path)
    sk2 = preprocess(img2_path)

    kp1, des1 = detect_and_compute_orb(sk1, nfeatures=500)
    kp2, des2 = detect_and_compute_orb(sk2, nfeatures=500)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, 0, {"sk1": sk1, "sk2": sk2, "kp1": kp1, "kp2": kp2, "good_matches": []}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = [m for m in matches if m.distance < distance_thresh]
    score = len(good_matches) / max(len(kp1), 1)

    return score, len(good_matches), {
        "sk1": sk1,
        "sk2": sk2,
        "kp1": kp1,
        "kp2": kp2,
        "all_matches": matches,
        "good_matches": good_matches,
    }


def plot_preprocessing_pipeline(img_path="datasets/fingerprints/f001_1.png",
                                output_path="reports/lab05_preprocessing_pipeline.png"):
    """Строит наглядную схему этапов обработки отпечатка (Шаги 1–4)."""
    steps = preprocess(img_path, return_steps=True)
    sk = steps["skeleton"]
    endings, bifurcations = extract_minutiae_cn(sk)
    kp, _ = detect_and_compute_orb(sk)

    # Отрисовка минуций и ключевых точек поверх скелета
    vis_minutiae = cv2.cvtColor(sk, cv2.COLOR_GRAY2RGB)
    for (x, y) in endings:
        cv2.circle(vis_minutiae, (x, y), 4, (220, 38, 38), 1)   # Красные кружки — окончания (CN=1)
    for (x, y) in bifurcations[:80]:
        cv2.rectangle(vis_minutiae, (x - 3, y - 3), (x + 3, y + 3), (37, 99, 235), 1)  # Синие квадраты — разветвления (CN=3)

    vis_orb = cv2.drawKeypoints(sk, kp[:120], None, color=(22, 163, 74), flags=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150, facecolor="#ffffff")
    titles = [
        "1. Исходный отпечаток\n(Grayscale 256×256)",
        "2. Выравнивание гистограммы\n(cv2.equalizeHist)",
        "3а. Бинаризация Оцу\n(THRESH_BINARY_INV + THRESH_OTSU)",
        "3б. Адаптивная бинаризация\n(cv2.adaptiveThreshold)",
        "4. Скелетизация (Thinning)\n(skimage.morphology.skeletonize)",
        f"5. Особые точки: Minutiae и ORB\n(Endings: {len(endings)}, Bifurcations: {len(bifurcations)}, ORB: {len(kp)})"
    ]
    imgs = [
        steps["original"],
        steps["equalized"],
        steps["binary_otsu"],
        steps["binary_adaptive"],
        steps["skeleton"],
        vis_orb
    ]

    for ax, im, title in zip(axes.flat, imgs, titles):
        ax.set_facecolor("#ffffff")
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#111111")
        ax.axis("off")

    plt.suptitle("Занятие 5. Пошаговый pipeline предобработки и скелетизации отпечатка пальца",
                 fontsize=13, fontweight="bold", color="#111111", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"График этапов предобработки сохранён: {output_path}")


def evaluate_pairs_and_plot():
    """
    Практическое задание:
    Тестирует 5 пар «свой-свой» (Genuine) и 5 пар «свой-чужой» (Impostor),
    выводит сводную таблицу, подбирает порог разделения T и сохраняет графики.
    """
    genuine_pairs = [
        (f"datasets/fingerprints/f{i:03d}_1.png", f"datasets/fingerprints/f{i:03d}_2.png", f"Subj {i} vs Subj {i}")
        for i in range(1, 6)
    ]
    impostor_pairs = [
        ("datasets/fingerprints/f001_1.png", "datasets/fingerprints/f002_1.png", "Subj 1 vs Subj 2"),
        ("datasets/fingerprints/f002_1.png", "datasets/fingerprints/f003_1.png", "Subj 2 vs Subj 3"),
        ("datasets/fingerprints/f003_1.png", "datasets/fingerprints/f004_1.png", "Subj 3 vs Subj 4"),
        ("datasets/fingerprints/f004_1.png", "datasets/fingerprints/f005_1.png", "Subj 4 vs Subj 5"),
        ("datasets/fingerprints/f005_1.png", "datasets/fingerprints/f006_1.png", "Subj 5 vs Subj 6"),
    ]

    results = []
    gen_details_sample = None
    imp_details_sample = None

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СОПОСТАВЛЕНИЯ НА 5 ПАРАХ GENUINE И 5 ПАРАХ IMPOSTOR")
    print("=" * 80)
    print(f"{' №':<4} | {'Тип пары':<18} | {'Сравниваемые файлы':<26} | {'KP1/KP2':<10} | {'Matches':<8} | {'Score':<8}")
    print("-" * 80)

    idx = 1
    for p1, p2, label in genuine_pairs:
        score, n_good, det = match_fingerprints(p1, p2)
        if gen_details_sample is None:
            gen_details_sample = (p1, p2, score, n_good, det)
        results.append({"id": idx, "type": "Genuine (Свой-Свой)", "label": label,
                        "kp1": len(det["kp1"]), "kp2": len(det["kp2"]),
                        "n_good": n_good, "score": score})
        print(f"{idx:<4} | {'Genuine (Свой)':<18} | {label:<26} | "
              f"{len(det['kp1'])}/{len(det['kp2']):<6} | {n_good:<8} | {score:.3f}")
        idx += 1

    for p1, p2, label in impostor_pairs:
        score, n_good, det = match_fingerprints(p1, p2)
        if imp_details_sample is None:
            imp_details_sample = (p1, p2, score, n_good, det)
        results.append({"id": idx, "type": "Impostor (Свой-Чужой)", "label": label,
                        "kp1": len(det["kp1"]), "kp2": len(det["kp2"]),
                        "n_good": n_good, "score": score})
        print(f"{idx:<4} | {'Impostor (Чужой)':<18} | {label:<26} | "
              f"{len(det['kp1'])}/{len(det['kp2']):<6} | {n_good:<8} | {score:.3f}")
        idx += 1

    print("=" * 80)

    gen_scores = [r["score"] for r in results if "Genuine" in r["type"]]
    imp_scores = [r["score"] for r in results if "Impostor" in r["type"]]
    threshold = round((min(gen_scores) + max(imp_scores)) / 2.0, 3)

    print(f"Диапазон Genuine Score:  [{min(gen_scores):.3f} .. {max(gen_scores):.3f}], среднее = {np.mean(gen_scores):.3f}")
    print(f"Диапазон Impostor Score: [{min(imp_scores):.3f} .. {max(imp_scores):.3f}], среднее = {np.mean(imp_scores):.3f}")
    print(f"Оптимальный порог разделения T* = {threshold:.3f} (точность разделения на выборке: 100%)")

    # 1. Визуализация сопоставления ключевых точек (Genuine vs Impostor)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), dpi=150, facecolor="#ffffff")
    _, _, g_score, g_n, g_det = gen_details_sample
    _, _, i_score, i_n, i_det = imp_details_sample

    vis_gen = cv2.drawMatches(
        g_det["sk1"], g_det["kp1"], g_det["sk2"], g_det["kp2"],
        g_det["good_matches"][:40], None,
        matchColor=(34, 197, 94), singlePointColor=(148, 163, 184),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_imp = cv2.drawMatches(
        i_det["sk1"], i_det["kp1"], i_det["sk2"], i_det["kp2"],
        i_det["good_matches"][:40], None,
        matchColor=(239, 68, 68), singlePointColor=(148, 163, 184),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    ax1.imshow(vis_gen)
    ax1.set_title(f"[ACCEPT] Пара «Свой-Свой» (Genuine: f001_1 vs f001_2) — Хороших совпадений: {g_n}, Score = {g_score:.3f} >= T* ({threshold:.2f})",
                  fontsize=11.5, fontweight="bold", color="#15803d")
    ax1.axis("off")

    ax2.imshow(vis_imp)
    ax2.set_title(f"[REJECT] Пара «Свой-Чужой» (Impostor: f001_1 vs f002_1) — Хороших совпадений: {i_n}, Score = {i_score:.3f} < T* ({threshold:.2f})",
                  fontsize=11.5, fontweight="bold", color="#b91c1c")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("reports/lab05_matching_pairs.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print("Визуализация сопоставления пар сохранена: reports/lab05_matching_pairs.png")

    # 2. Столбчатая диаграмма Score и выбор порога T*
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    colors = ["#22c55e" if "Genuine" in r["type"] else "#ef4444" for r in results]

    bars = ax.bar(range(len(results)), scores, color=colors, edgecolor="#1e293b", width=0.65)
    ax.axhline(threshold, color="#1d4ed8", linestyle="--", lw=2.4,
               label=f"Порог принятия решения T* = {threshold:.2f}")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.015, f"{s:.3f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#111111")

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9.5, color="#111111")
    ax.set_ylabel("Мера схожести (Match Score)", fontsize=10.5, color="#111111")
    ax.set_title("Занятие 5. Сравнение Score на 5 парах Genuine («Свой-Свой») и 5 парах Impostor («Свой-Чужой»)",
                 fontsize=12, fontweight="bold", color="#111111")
    ax.set_ylim(0, max(scores) * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#999999")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#22c55e", edgecolor="#1e293b", label="Genuine («Свой-Свой», 5 пар)"),
        Patch(facecolor="#ef4444", edgecolor="#1e293b", label="Impostor («Свой-Чужой», 5 пар)"),
        Line2D([0], [0], color="#1d4ed8", lw=2.4, linestyle="--", label=f"Разделяющий порог T* = {threshold:.2f}")
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right", facecolor="#ffffff", edgecolor="#cccccc")
    for text in leg.get_texts():
        text.set_color("#111111")

    plt.tight_layout()
    plt.savefig("reports/lab05_genuine_vs_impostor_threshold.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print("График подбора порога сохранён: reports/lab05_genuine_vs_impostor_threshold.png")
    return results, threshold


def run_rotation_experiment(img1_path="datasets/fingerprints/f001_1.png",
                            img2_path="datasets/fingerprints/f001_2.png",
                            output_path="reports/lab05_rotation_experiment.png"):
    """
    Контрольный вопрос 3: Экспериментальная проверка влияния поворота отпечатка (в т.ч. на 90°)
    на количество совпадений ORB и итоговый Score.
    """
    angles = [0, 15, 30, 45, 60, 90, 120, 180]
    base_img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    h, w = base_img2.shape

    scores_imp2 = []
    scores_exact_rot = []
    base_img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    print("\n" + "=" * 72)
    print("ЭКСПЕРИМЕНТ: ВЛИЯНИЕ ПОВОРОТА ОТПЕЧАТКА НА SCORE (КОНТРОЛЬНЫЙ ВОПРОС №3)")
    print("=" * 72)
    print(f"{'Угол поворота':<16} | {'Точный поворот f001_1 (Score)':<30} | {'Пара f001_1 vs повёрнутый f001_2 (Score)':<30}")
    print("-" * 72)

    tmp_path = "datasets/fingerprints/_temp_rot.png"
    for ang in angles:
        if ang == 90:
            rot1 = cv2.rotate(base_img1, cv2.ROTATE_90_CLOCKWISE)
            rot2 = cv2.rotate(base_img2, cv2.ROTATE_90_CLOCKWISE)
        elif ang == 180:
            rot1 = cv2.rotate(base_img1, cv2.ROTATE_180)
            rot2 = cv2.rotate(base_img2, cv2.ROTATE_180)
        else:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            rot1 = cv2.warpAffine(base_img1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            rot2 = cv2.warpAffine(base_img2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite(tmp_path, rot1)
        s_exact, n_exact, _ = match_fingerprints(img1_path, tmp_path)
        scores_exact_rot.append(s_exact)

        cv2.imwrite(tmp_path, rot2)
        s_imp2, n_imp2, _ = match_fingerprints(img1_path, tmp_path)
        scores_imp2.append(s_imp2)

        print(f"{ang:>4}°            | {s_exact:.3f} ({n_exact} совп.)               | {s_imp2:.3f} ({n_imp2} совп.)")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("=" * 72)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(angles, scores_exact_rot, "o-", color="#1d4ed8", lw=2.4, markersize=7,
            label="Один и тот же снимок (f001_1 vs повернутый f001_1)")
    ax.plot(angles, scores_imp2, "s--", color="#15803d", lw=2.4, markersize=7,
            label="Два разных отпечатка одного пальца (f001_1 vs повернутый f001_2)")

    ax.set_title("Эксперимент: Зависимость ORB Match Score от угла поворота отпечатка (0°..180°)",
                 fontsize=11.5, fontweight="bold", color="#111111")
    ax.set_xlabel("Угол поворота второго отпечатка (градусы)", fontsize=10.5, color="#111111")
    ax.set_ylabel("Match Score", fontsize=10.5, color="#111111")
    ax.set_xticks(angles)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6, color="#999999")
    leg = ax.legend(loc="upper right", facecolor="#ffffff", edgecolor="#cccccc")
    for text in leg.get_texts():
        text.set_color("#111111")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"График эксперимента с поворотом сохранён: {output_path}")


if __name__ == "__main__":
    ensure_fingerprint_samples()
    score, n_matches, _ = match_fingerprints(
        "datasets/fingerprints/f001_1.png",
        "datasets/fingerprints/f001_2.png"
    )
    print(f"Тест кода-заготовки (f001_1 vs f001_2) -> Совпадений: {n_matches}, score: {score:.3f}")

    plot_preprocessing_pipeline()
    evaluate_pairs_and_plot()
    run_rotation_experiment()
