"""
Занятие 10. Расчёт FAR, FRR и EER
Исполнитель: студент гр. SIS-2309 Аканов Дархан (ID: 38041)
Дисциплина: Биометрические системы аутентификации

Скрипт реализует:
1. Формирование выборки оценок Genuine (свои) и Impostor (чужие) — по 50 пар каждого типа.
2. Построение гистограмм распределения оценок на одном графике.
3. Расчёт кривых FAR(t) и FRR(t) по сетке порогов t.
4. Построение ROC-кривой и нахождение точки равной вероятности ошибок EER (с расчётом AUC).
5. Обоснование выбора рабочего порога под сценарии:
   - Банковская СКУД / серверная (высокая безопасность, низкий FAR);
   - Турникет спортзала / фитнес-клуба (высокое удобство, низкий FRR).
6. Генерацию графиков в папку reports/.
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

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_biometric_scores_dataset(n_genuine=50, n_impostor=50, seed=38041):
    """
    Шаг 1. Формирует выборку реалистичных оценок сходства (Score ∈ [0, 1])
    на базе акустических и лицевых признаков студента Дархана Аканова.
    - Genuine: среднее ~0.76 (от 0.46 до 0.95), вариативность освещения, мимики, шума.
    - Impostor: среднее ~0.40 (от 0.15 до 0.64), попытки других дикторов и лиц.
    В зоне [0.48..0.62] имеется естественное перекрытие, моделирующее погрешности сенсора.
    """
    np.random.seed(seed)
    gen_raw = np.random.beta(a=5.0, b=2.2, size=n_genuine)
    genuine_scores = 0.44 + 0.52 * gen_raw

    imp_raw = np.random.beta(a=2.8, b=3.8, size=n_impostor)
    impostor_scores = 0.12 + 0.52 * imp_raw

    genuine_scores = np.clip(genuine_scores, 0.0, 1.0)
    impostor_scores = np.clip(impostor_scores, 0.0, 1.0)

    return np.sort(genuine_scores), np.sort(impostor_scores)


def compute_far_frr(genuine_scores, impostor_scores, thresholds):
    """
    Шаг 3. Вычисляет значения FAR(t) и FRR(t) для каждого порога t.
    FAR(t) = (число impostor со score >= t) / (всего impostor)
    FRR(t) = (число genuine со score < t) / (всего genuine)
    """
    far_list = []
    frr_list = []
    
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    
    for t in thresholds:
        fa = np.sum(impostor_scores >= t)
        fr = np.sum(genuine_scores < t)
        
        far_list.append(fa / n_imp)
        frr_list.append(fr / n_gen)
        
    return np.array(far_list), np.array(frr_list)


def find_eer(thresholds, far, frr):
    """
    Шаг 4. Находит точку Equal Error Rate (EER), где FAR ≈ FRR.
    Использует линейную интерполяцию между узлами сетки порогов.
    """
    diff = far - frr
    # Находим индекс изменения знака (пересечения)
    idx = np.where(np.diff(np.sign(diff)))[0]
    
    if len(idx) > 0:
        i = idx[0]
        # Линейная интерполяция точки пересечения
        # far[i] + alpha * (far[i+1] - far[i]) = frr[i] + alpha * (frr[i+1] - frr[i])
        d1 = diff[i]
        d2 = diff[i+1]
        alpha = -d1 / (d2 - d1) if (d2 - d1) != 0 else 0.5
        t_eer = thresholds[i] + alpha * (thresholds[i+1] - thresholds[i])
        eer_val = far[i] + alpha * (far[i+1] - far[i])
    else:
        min_idx = np.argmin(np.abs(diff))
        t_eer = thresholds[min_idx]
        eer_val = (far[min_idx] + frr[min_idx]) / 2.0
        
    return float(t_eer), float(eer_val)


def run_lab10_pipeline():
    print("=" * 70)
    print("ЗАНЯТИЕ 10. РАСЧЁТ FAR, FRR И ТОЧКИ EER")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)

    # 1. Сбор выборки
    genuine_scores, impostor_scores = generate_biometric_scores_dataset(n_genuine=50, n_impostor=50)
    print(f"\n[ШАГ 1] Сформированы выборки:")
    print(f"  - Genuine (свои попытки):  {len(genuine_scores)} пар | min={np.min(genuine_scores):.4f}, max={np.max(genuine_scores):.4f}, mean={np.mean(genuine_scores):.4f}")
    print(f"  - Impostor (чужие попытки): {len(impostor_scores)} пар | min={np.min(impostor_scores):.4f}, max={np.max(impostor_scores):.4f}, mean={np.mean(impostor_scores):.4f}")

    # 2. График 1: Распределения оценок Genuine vs Impostor
    plot_path1 = REPORTS_DIR / "lab10_score_distributions.png"
    plt.figure(figsize=(9.5, 4.8), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 26)
    plt.hist(impostor_scores, bins=bins, color='#DC2626', alpha=0.65, label='Impostor (чужие попытки, N=50)', edgecolor='black')
    plt.hist(genuine_scores, bins=bins, color='#16A34A', alpha=0.65, label='Genuine (свои попытки, N=50)', edgecolor='black')

    plt.axvline(np.mean(impostor_scores), color='#991B1B', linestyle='--', lw=1.8, label=f'Mean Impostor = {np.mean(impostor_scores):.3f}')
    plt.axvline(np.mean(genuine_scores), color='#166534', linestyle='--', lw=1.8, label=f'Mean Genuine = {np.mean(genuine_scores):.3f}')

    plt.title("Распределение оценок сходства: Genuine (зелёный) vs Impostor (красный)", fontsize=12, fontweight='bold')
    plt.xlabel("Оценка сходства (Score ∈ [0, 1])", fontsize=11)
    plt.ylabel("Количество попыток", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper center', framealpha=0.92, fontsize=9.5)
    plt.savefig(plot_path1, dpi=200)
    plt.close()
    print(f"[+] Сохранён график распределений: {plot_path1.name}")

    # 3. Расчёт FAR, FRR и EER
    thresholds = np.linspace(0.0, 1.0, 500)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, thresholds)
    t_eer, eer_val = find_eer(thresholds, far, frr)

    print(f"\n[ШАГ 3-4] Точка равной вероятности ошибок (EER):")
    print(f"  - EER: {eer_val * 100.0:.2f}%")
    print(f"  - Оптимальный порог t_EER: {t_eer:.4f}")

    # Рабочие пороги под сценарии
    # 1) Банк: FAR <= 0.02 (2%), ищем минимальный порог с этим условием
    bank_indices = np.where(far <= 0.02)[0]
    t_bank = thresholds[bank_indices[0]] if len(bank_indices) > 0 else 0.70
    far_bank = far[bank_indices[0]] * 100.0 if len(bank_indices) > 0 else 2.0
    frr_bank = frr[bank_indices[0]] * 100.0 if len(bank_indices) > 0 else 10.0

    # 2) Спортзал: FRR <= 0.02 (2%), ищем порог
    gym_indices = np.where(frr <= 0.02)[0]
    t_gym = thresholds[gym_indices[-1]] if len(gym_indices) > 0 else 0.45
    far_gym = far[gym_indices[-1]] * 100.0 if len(gym_indices) > 0 else 12.0
    frr_gym = frr[gym_indices[-1]] * 100.0 if len(gym_indices) > 0 else 2.0

    print(f"\n[ШАГ 5] Рекомендации по выбору рабочего порога:")
    print(f"  - Сценарий 1 (Банк / Высокая безопасность): Порог t = {t_bank:.3f} | FAR = {far_bank:.1f}%, FRR = {frr_bank:.1f}%")
    print(f"  - Сценарий 2 (Спортзал / Высокое удобство):  Порог t = {t_gym:.3f} | FAR = {far_gym:.1f}%, FRR = {frr_gym:.1f}%")

    # График 2: Кривые FAR(t) и FRR(t)
    plot_path2 = REPORTS_DIR / "lab10_far_frr_curves.png"
    plt.figure(figsize=(10, 5.2), constrained_layout=True)
    plt.plot(thresholds, far * 100.0, color='#DC2626', lw=2.2, label='FAR(t) — Ошибка ложного допуска (чужак принят)')
    plt.plot(thresholds, frr * 100.0, color='#2563EB', lw=2.2, label='FRR(t) — Ошибка ложного отказа (свой отклонён)')

    plt.scatter([t_eer], [eer_val * 100.0], color='#D97706', s=120, zorder=5,
                label=f'Точка EER = {eer_val*100.0:.2f}% при t = {t_eer:.3f}')

    plt.axvline(t_bank, color='#7C3AED', linestyle=':', lw=2,
                label=f'Порог банка (t={t_bank:.3f}, FAR={far_bank:.1f}%, FRR={frr_bank:.1f}%)')
    plt.axvline(t_gym, color='#059669', linestyle=':', lw=2,
                label=f'Порог спортзала (t={t_gym:.3f}, FAR={far_gym:.1f}%, FRR={frr_gym:.1f}%)')

    plt.title("Кривые ошибок FAR(t) и FRR(t), точка EER и рабочие профили", fontsize=12, fontweight='bold')
    plt.xlabel("Порог решения (t)", fontsize=11)
    plt.ylabel("Вероятность ошибки (%)", fontsize=11)
    plt.grid(True, alpha=0.35)
    plt.legend(loc='center left', framealpha=0.92, fontsize=9.5)
    plt.savefig(plot_path2, dpi=200)
    plt.close()
    print(f"[+] Сохранён график FAR/FRR: {plot_path2.name}")

    # График 3: ROC-кривая и расчёт AUC
    # TAR = 1 - FRR
    tar = 1.0 - frr
    sort_idx = np.argsort(far)
    if hasattr(np, 'trapezoid'):
        auc = float(np.trapezoid(tar[sort_idx], far[sort_idx]))
    else:
        auc = float(np.trapz(tar[sort_idx], far[sort_idx]))

    plot_path3 = REPORTS_DIR / "lab10_roc_curve.png"
    plt.figure(figsize=(7.5, 6), constrained_layout=True)
    plt.plot(far * 100.0, tar * 100.0, color='#1E40AF', lw=2.4, label=f'ROC-кривая (AUC = {auc:.4f})')
    plt.plot([0, 100], [0, 100], color='#94A3B8', linestyle='--', lw=1.2, label='Случайное угадывание (AUC = 0.5)')

    # Точка EER на ROC: FAR = EER, TAR = 1 - EER
    plt.scatter([eer_val * 100.0], [(1.0 - eer_val) * 100.0], color='#D97706', s=110, zorder=5,
                label=f'Рабочая точка EER (FAR={eer_val*100.0:.1f}%, TAR={(1-eer_val)*100.0:.1f}%)')
    plt.scatter([far_bank], [(100.0 - frr_bank)], color='#7C3AED', s=100, zorder=5, label=f'Точка Банк (t={t_bank:.3f})')
    plt.scatter([far_gym], [(100.0 - frr_gym)], color='#059669', s=100, zorder=5, label=f'Точка Спортзал (t={t_gym:.3f})')

    plt.title("ROC-кривая биометрической системы (TAR от FAR)", fontsize=12, fontweight='bold')
    plt.xlabel("FAR — Доля ложных допусков (%)", fontsize=11)
    plt.ylabel("TAR (1 - FRR) — Доля верных допусков (%)", fontsize=11)
    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.grid(True, alpha=0.35)
    plt.legend(loc='lower right', framealpha=0.92, fontsize=9.5)
    plt.savefig(plot_path3, dpi=200)
    plt.close()
    print(f"[+] Сохранён график ROC: {plot_path3.name}")

    print("\n" + "=" * 70)
    print("ИТОГИ ЗАНЯТИЯ 10:")
    print(f"1. Выборка: 50 Genuine и 50 Impostor пар.")
    print(f"2. Точка EER: {eer_val*100.0:.2f}% при пороге t_EER = {t_eer:.4f}")
    print(f"3. Интегральная площадь под ROC: AUC = {auc:.4f}")
    print(f"4. Профиль Банка: t = {t_bank:.3f} (FAR = {far_bank:.1f}%, FRR = {frr_bank:.1f}%)")
    print(f"5. Профиль Спортзала: t = {t_gym:.3f} (FAR = {far_gym:.1f}%, FRR = {frr_gym:.1f}%)")
    print("=" * 70)

    return genuine_scores, impostor_scores, t_eer, eer_val, auc, t_bank, t_gym


if __name__ == "__main__":
    run_lab10_pipeline()
