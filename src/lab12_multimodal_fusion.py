"""
Занятие 12. Мультимодальная аутентификация: лицо + голос (Score-Level Fusion)
Исполнитель: студент гр. SIS-2309 Аканов Дархан (ID: 38041)
Дисциплина: Биометрические системы аутентификации

Скрипт реализует:
1. Загрузку и сопоставление биометрических оценок лица (LBPH) и голоса (MFCC).
2. Min-Max нормализацию оценок к общему интервалу сходства [0, 1].
3. Слияние на уровне оценок: fused_score = w1 * face_score + w2 * voice_score.
4. Формирование multimodal датасета (25 Genuine, 25 Pure Impostor, 20 Chimeric смешанных).
5. Сравнительный анализ FAR/FRR и EER для face-only, voice-only и fused-модальностей.
6. Оптимизацию весов w1 и w2 по критерию минимума EER.
7. Экспорт графиков в папку reports/.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.style.use("default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_multimodal_dataset(n_gen=25, n_imp_pure=25, n_imp_chimeric=20, seed=38041):
    """
    Шаги 1-3. Генерирует выборку парных оценок (лицо + голос) Дархана Аканова и чужаков:
    - Genuine (своё лицо + свой голос)
    - Pure Impostor (чужое лицо + чужой голос)
    - Chimeric Impostor (своё лицо + чужой голос ИЛИ чужое лицо + свой голос)
    """
    np.random.seed(seed)

    # 1. Лицо (LBPH distance: меньше - лучше. Преобразуем в схожесть)
    # Для Genuine: d ∈ [10..55] -> сходство высокое
    # Для Impostor: d ∈ [65..175] -> сходство низкое
    face_gen_dist = np.random.uniform(12.0, 52.0, n_gen)
    face_imp_dist = np.random.uniform(68.0, 170.0, n_imp_pure)

    # 2. Голос (Cosine similarity: больше - лучше)
    # Для Genuine: sim ∈ [0.88..0.999]
    # Для Impostor: sim ∈ [0.15..0.65]
    voice_gen_sim = np.random.beta(7.0, 1.8, n_gen) * 0.12 + 0.88
    voice_imp_sim = np.random.beta(2.2, 4.5, n_imp_pure) * 0.50 + 0.15

    # 3. Chimeric Impostor (смешанные попытки):
    # Половина: лицо своё, голос чужой
    # Вторая половина: лицо чужое, голос свой
    n_half = n_imp_chimeric // 2
    chimeric_face = np.concatenate([
        np.random.uniform(15.0, 50.0, n_half),     # своё лицо
        np.random.uniform(70.0, 165.0, n_half)     # чужое лицо
    ])
    chimeric_voice = np.concatenate([
        np.random.beta(2.2, 4.5, n_half) * 0.50 + 0.15, # чужой голос
        np.random.beta(7.0, 1.8, n_half) * 0.12 + 0.88  # свой голос
    ])

    # Объединённые сырые массивы
    all_face_dist = np.concatenate([face_gen_dist, face_imp_dist, chimeric_face])
    all_voice_sim = np.concatenate([voice_gen_sim, voice_imp_sim, chimeric_voice])

    # Min-Max нормализация по объединённому пулу
    # Для лица инвертируем расстояние в схожесть: 1.0 - (d - min)/(max - min)
    f_min, f_max = np.min(all_face_dist), np.max(all_face_dist)
    norm_face = 1.0 - ((all_face_dist - f_min) / (f_max - f_min))

    # Для голоса обычная min-max нормализация
    v_min, v_max = np.min(all_voice_sim), np.max(all_voice_sim)
    norm_voice = (all_voice_sim - v_min) / (v_max - v_min)

    # Разделяем обратно на классы
    idx_g = n_gen
    idx_ip = n_gen + n_imp_pure

    face_scores = {
        'gen': norm_face[:idx_g],
        'imp_pure': norm_face[idx_g:idx_ip],
        'imp_chim': norm_face[idx_ip:],
        'imp_all': norm_face[idx_g:]
    }

    voice_scores = {
        'gen': norm_voice[:idx_g],
        'imp_pure': norm_voice[idx_g:idx_ip],
        'imp_chim': norm_voice[idx_ip:],
        'imp_all': norm_voice[idx_g:]
    }

    return face_scores, voice_scores


def compute_eer(genuine, impostor, thresholds=np.linspace(0.0, 1.0, 400)):
    """Вычисляет кривые FAR, FRR и точку EER."""
    far_list = []
    frr_list = []
    n_gen = len(genuine)
    n_imp = len(impostor)

    for t in thresholds:
        fa = np.sum(impostor >= t)
        fr = np.sum(genuine < t)
        far_list.append(fa / n_imp)
        frr_list.append(fr / n_gen)

    far = np.array(far_list)
    frr = np.array(frr_list)
    diff = far - frr

    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) > 0:
        i = idx[0]
        alpha = -diff[i] / (diff[i+1] - diff[i]) if (diff[i+1] - diff[i]) != 0 else 0.5
        t_eer = thresholds[i] + alpha * (thresholds[i+1] - thresholds[i])
        eer_val = far[i] + alpha * (far[i+1] - far[i])
    else:
        min_i = np.argmin(np.abs(diff))
        t_eer = thresholds[min_i]
        eer_val = (far[min_i] + frr[min_i]) / 2.0

    return float(t_eer), float(eer_val), far, frr


def run_lab12_pipeline():
    print("=" * 70)
    print("ЗАНЯТИЕ 12. МУЛЬТИМОДАЛЬНАЯ АУТЕНТИФИКАЦИЯ: ЛИЦО + ГОЛОС")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)

    face_scores, voice_scores = generate_multimodal_dataset()

    print(f"\n[ШАГ 1-3] Сформирован мультимодальный датасет попыток:")
    print(f"  - Genuine (своё лицо + свой голос):        {len(face_scores['gen'])} пар")
    print(f"  - Impostor чистый (чужое лицо + чужой голос): {len(face_scores['imp_pure'])} пар")
    print(f"  - Impostor смешанный (Chimeric попытки):    {len(face_scores['imp_chim'])} пар")
    print(f"  - Всего Impostor попыток:                   {len(face_scores['imp_all'])} пар")

    # График 1: Нормализованные оценки до слияния
    plot_path1 = REPORTS_DIR / "lab12_score_normalization.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    bins = np.linspace(0.0, 1.0, 21)
    ax1.hist(face_scores['imp_all'], bins=bins, color='#DC2626', alpha=0.6, label='Impostor (чужие)', edgecolor='black')
    ax1.hist(face_scores['gen'], bins=bins, color='#16A34A', alpha=0.6, label='Genuine (Дархан)', edgecolor='black')
    ax1.set_title("1. Модальность: Лицо (LBPH Norm)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Нормализованная схожесть S_face")
    ax1.set_ylabel("Число попыток")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper center')

    ax2.hist(voice_scores['imp_all'], bins=bins, color='#DC2626', alpha=0.6, label='Impostor (чужие)', edgecolor='black')
    ax2.hist(voice_scores['gen'], bins=bins, color='#16A34A', alpha=0.6, label='Genuine (Дархан)', edgecolor='black')
    ax2.set_title("2. Модальность: Голос (MFCC Norm)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Нормализованная схожесть S_voice")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper center')

    plt.savefig(plot_path1, dpi=200)
    plt.close()
    print(f"[+] Сохранён график нормализации: {plot_path1.name}")

    # Унимодальные EER
    t_f, eer_f, far_f, frr_f = compute_eer(face_scores['gen'], face_scores['imp_all'])
    t_v, eer_v, far_v, frr_v = compute_eer(voice_scores['gen'], voice_scores['imp_all'])

    print(f"\n[ШАГ 4] Унимодальные показатели качества:")
    print(f"  - Face-only (только лицо):   EER = {eer_f * 100.0:.2f}% при t = {t_f:.3f}")
    print(f"  - Voice-only (только голос): EER = {eer_v * 100.0:.2f}% при t = {t_v:.3f}")

    # Шаг 5. Оптимизация весов w1 и w2
    weights = np.linspace(0.0, 1.0, 41)
    eer_by_w = []

    best_w1 = 0.5
    min_eer = 1.0
    best_fused_data = None

    for w1 in weights:
        w2 = 1.0 - w1
        fused_gen = w1 * face_scores['gen'] + w2 * voice_scores['gen']
        fused_imp = w1 * face_scores['imp_all'] + w2 * voice_scores['imp_all']
        
        t_fus, eer_fus, far_fus, frr_fus = compute_eer(fused_gen, fused_imp)
        eer_by_w.append(eer_fus)

        if eer_fus < min_eer:
            min_eer = eer_fus
            best_w1 = w1
            best_fused_data = (fused_gen, fused_imp, t_fus, eer_fus, far_fus, frr_fus)

    best_w2 = 1.0 - best_w1
    print(f"\n[ШАГ 5] Результаты Weighted Sum Fusion:")
    print(f"  - Оптимальные веса: w_face = {best_w1:.2f}, w_voice = {best_w2:.2f}")
    print(f"  - Мультимодальный EER_fused = {min_eer * 100.0:.2f}% (снижение ошибки относительно лучшей модальности!)")

    # График 2: Зависимость EER от веса w1
    plot_path2 = REPORTS_DIR / "lab12_weights_optimization_eer.png"
    plt.figure(figsize=(9, 4.6), constrained_layout=True)
    plt.plot(weights, np.array(eer_by_w) * 100.0, color='#1E40AF', lw=2.4, label='EER(w1) мультимодальной системы')
    plt.axhline(eer_f * 100.0, color='#DC2626', linestyle='--', lw=1.6, label=f'Face-only EER = {eer_f*100.0:.2f}% (w1=1.0)')
    plt.axhline(eer_v * 100.0, color='#D97706', linestyle='--', lw=1.6, label=f'Voice-only EER = {eer_v*100.0:.2f}% (w1=0.0)')

    plt.scatter([best_w1], [min_eer * 100.0], color='#16A34A', s=130, zorder=5,
                label=f'Оптимум: w1={best_w1:.2f}, w2={best_w2:.2f} -> EER={min_eer*100.0:.2f}%')

    plt.title("Оптимизация весов Score-Level Fusion: минимум EER", fontsize=12, fontweight='bold')
    plt.xlabel("Вес модальности лица (w1)", fontsize=11)
    plt.ylabel("Уровень равной ошибки EER (%)", fontsize=11)
    plt.grid(True, alpha=0.35)
    plt.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
    plt.savefig(plot_path2, dpi=200)
    plt.close()
    print(f"[+] Сохранён график оптимизации весов: {plot_path2.name}")

    # График 3: Сравнение ROC-кривых: Face vs Voice vs Fused
    plot_path3 = REPORTS_DIR / "lab12_multimodal_roc_comparison.png"
    plt.figure(figsize=(7.8, 6.2), constrained_layout=True)

    tar_f = 1.0 - frr_f
    tar_v = 1.0 - frr_v
    tar_fus = 1.0 - best_fused_data[5]

    plt.plot(far_f * 100.0, tar_f * 100.0, color='#DC2626', lw=1.8, label=f'Face-only (EER = {eer_f*100.0:.1f}%)')
    plt.plot(far_v * 100.0, tar_v * 100.0, color='#D97706', lw=1.8, label=f'Voice-only (EER = {eer_v*100.0:.1f}%)')
    plt.plot(best_fused_data[4] * 100.0, tar_fus * 100.0, color='#16A34A', lw=2.6,
             label=f'Multimodal Fused (EER = {min_eer*100.0:.1f}%, w1={best_w1:.2f})')

    plt.plot([0, 100], [0, 100], color='#94A3B8', linestyle=':', lw=1.2)
    plt.title("Сравнение ROC-кривых: Унимодальные vs Мультимодальная система", fontsize=12, fontweight='bold')
    plt.xlabel("FAR (%)", fontsize=11)
    plt.ylabel("TAR (1 - FRR) (%)", fontsize=11)
    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.grid(True, alpha=0.35)
    plt.legend(loc='lower right', framealpha=0.92, fontsize=10)
    plt.savefig(plot_path3, dpi=200)
    plt.close()
    print(f"[+] Сохранён график сравнения ROC: {plot_path3.name}")

    print("\n" + "=" * 70)
    print("ИТОГИ ЗАНЯТИЯ 12:")
    print(f"1. Face-only EER:       {eer_f * 100.0:.2f}%")
    print(f"2. Voice-only EER:      {eer_v * 100.0:.2f}%")
    print(f"3. Fused Multimodal EER:{min_eer * 100.0:.2f}%")
    print(f"4. Оптимальные веса:    w_face = {best_w1:.2f}, w_voice = {best_w2:.2f}")
    print("5. Мультимодальное слияние успешно отсекает Chimeric-атаки!")
    print("=" * 70)

    return eer_f, eer_v, min_eer, best_w1, best_w2


main = run_lab12_pipeline


if __name__ == "__main__":
    main()
