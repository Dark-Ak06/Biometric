# lab03_access_control.py — моделирование архитектуры СКУД и графики Decision Module (Занятие 3)
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm

os.makedirs("reports", exist_ok=True)


def draw_architecture_diagram(output_path="reports/lab03_architecture_diagram.png"):
    """
    Рисует высококонтрастную блок-схему контуров Enrollment и Verification с точками атак (Threat Modeling).
    Все цвета текста и фона заданы явно (не зависят от тёмной темы IDE / Jupyter).
    """
    fig, ax = plt.subplots(figsize=(18, 11), dpi=160, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-0.3, 10.4)
    ax.axis("off")

    # Заголовок диаграммы
    ax.text(
        8.0, 9.85,
        "Архитектура биометрической СКУД научно-исследовательской лаборатории\n"
        "(Контуры Enrollment и Verification + Threat Modeling)",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#0d254c",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#dbeafe", edgecolor="#1d4ed8", lw=2.2)
    )

    def add_box(x, y, w, h, text, fc="#ffffff", ec="#1e293b", tc="#0f172a", fontsize=10.5):
        # Тень блока для объёма и читаемости
        shadow = patches.FancyBboxPatch(
            (x + 0.04, y - 0.04), w, h,
            boxstyle="round,pad=0.12",
            facecolor="#cbd5e1", edgecolor="none", zorder=1
        )
        ax.add_patch(shadow)
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=2
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=tc, zorder=3, linespacing=1.25
        )

    def add_arrow(x1, y1, x2, y2, label="", color="#1e293b", rad=0.0, label_offset=(0.0, 0.24)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.4,head_length=0.55",
                lw=2.2, color=color,
                connectionstyle=f"arc3,rad={rad}"
            ),
            zorder=4
        )
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax.text(
                mx, my, label,
                ha="center", va="center", fontsize=9.0, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=color, lw=1.0, alpha=0.95),
                zorder=5
            )

    def add_threat_badge(x, y, text):
        ax.text(
            x, y, text,
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee2e2", edgecolor="#dc2626", lw=1.6),
            zorder=6, linespacing=1.2
        )

    # ==================== КОНТУР ENROLLMENT (ВЕРХНИЙ) ====================
    ax.text(
        0.1, 8.95,
        "1. КОНТУР РЕГИСТРАЦИИ (ENROLLMENT PIPELINE)",
        fontsize=12.5, fontweight="bold", color="#1e3a8a"
    )

    add_box(0.1, 7.2, 2.4, 1.3, "1. Сенсор захвата\n(RGB + NIR камера\nвысокого разрешения)", fc="#fef3c7", ec="#d97706")
    add_box(3.3, 7.2, 2.6, 1.3, "2. Предобработка и QA\n(Детекция лица,\nвыравнивание 112x112)", fc="#dcfce7", ec="#15803d")
    add_box(6.7, 7.2, 2.7, 1.3, "3. Извлечение признаков\n(Feature Extractor)\nCNN ArcFace → 512 float", fc="#dcfce7", ec="#15803d")
    add_box(10.2, 7.2, 2.6, 1.3, "4. Генератор шаблона\n(Cancelable Biometrics +\nШифрование AES-256)", fc="#f3e8ff", ec="#7e22ce")
    add_box(13.5, 5.1, 2.5, 2.2, "5. Хранилище шаблонов\n(Template DB + HSM)\n[Subject_ID,\nEncrypted_Template,\nAccess_Role]", fc="#ffe4e6", ec="#be123c")

    add_arrow(2.5, 7.85, 3.3, 7.85, "Кадр", label_offset=(0.0, 0.28))
    add_arrow(5.9, 7.85, 6.7, 7.85, "ROI 112x112", label_offset=(0.0, 0.28))
    add_arrow(9.4, 7.85, 10.2, 7.85, "Вектор v", label_offset=(0.0, 0.28))
    add_arrow(12.8, 7.85, 14.75, 7.3, "Запись в БД", label_offset=(0.15, 0.28))

    # ==================== КОНТУР VERIFICATION (НИЖНИЙ) ====================
    ax.text(
        0.1, 5.25,
        "2. КОНТУР ВЕРИФИКАЦИИ 1:1 (VERIFICATION PIPELINE)",
        fontsize=12.5, fontweight="bold", color="#1e3a8a"
    )

    add_box(0.1, 3.65, 2.4, 1.15, "RFID / Smart-карта\n(Ввод заявленного\nSubject_ID)", fc="#e0f2fe", ec="#0284c7")
    add_box(0.1, 1.75, 2.4, 1.35, "1. Сенсор захвата\n(RGB + NIR камера +\nИК-подсветка)", fc="#fef3c7", ec="#d97706")
    add_box(3.3, 1.75, 2.6, 1.35, "2. Предобработка + PAD\n(Liveness Detection:\nпроверка «живости»)", fc="#dcfce7", ec="#15803d")
    add_box(6.7, 1.75, 2.7, 1.35, "3. Извлечение признаков\n(Feature Extractor)\nCNN ArcFace → v_live", fc="#dcfce7", ec="#15803d")
    add_box(10.2, 1.75, 2.6, 1.35, "6. Модуль сравнения\n(Matcher)\nS = cos(v_live, v_db)", fc="#ffedd5", ec="#ea580c")
    add_box(10.2, 0.05, 2.6, 1.15, "7. Принятие решения\n(Decision Module)\nПроверка: S >= T ?", fc="#fce7f3", ec="#be185d")
    add_box(13.5, 1.75, 2.5, 1.3, "8a. ACCEPT (Доступ открыт)\nРазблокировка замка\nтурникета + Аудит-лог", fc="#dcfce7", ec="#15803d")
    add_box(13.5, 0.05, 2.5, 1.15, "8b. REJECT (Отказ)\nБлокировка двери +\nСигнал тревоги охране", fc="#ffe4e6", ec="#be123c")

    # Стрелки контура верификации
    add_arrow(2.5, 2.42, 3.3, 2.42, "Сырой кадр", label_offset=(0.0, 0.28))
    add_arrow(5.9, 2.42, 6.7, 2.42, "Живой ROI", label_offset=(0.0, 0.28))
    add_arrow(9.4, 2.42, 10.2, 2.42, "v_live", label_offset=(0.0, 0.28))
    add_arrow(2.5, 4.22, 13.5, 5.75, "Запрос шаблона по Subject_ID", color="#0284c7", rad=0.05, label_offset=(-0.4, 0.32))
    add_arrow(14.1, 5.1, 11.5, 3.1, "Шаблон v_db (mTLS)", color="#7e22ce", rad=0.14, label_offset=(0.35, 0.2))
    add_arrow(11.5, 1.75, 11.5, 1.2, "Score S", label_offset=(-0.45, 0.0))
    add_arrow(12.8, 0.85, 13.5, 2.25, "S >= T (Да)", color="#15803d", label_offset=(-0.1, 0.25))
    add_arrow(12.8, 0.5, 13.5, 0.5, "S < T (Нет)", color="#be123c", label_offset=(0.0, -0.25))

    # ==================== ТОЧКИ УГРОЗ (THREAT MODELING) ====================
    add_threat_badge(1.3, 1.05, "[!] Угроза 1: Spoofing\n(фото/видео/3D-маска)\nЗащита: PAD + ИК-камера")
    add_threat_badge(4.6, 3.55, "[!] Угроза 2: Replay в канале\nЗащита: mTLS + Timestamp")
    add_threat_badge(8.05, 1.05, "[!] Угроза 3: Подмена вектора\nЗащита: TEE / SGX анклав")
    add_threat_badge(14.75, 8.35, "[!] Угроза 4: Утечка БД шаблонов\nЗащита: AES-256 + Cancelable Bio")
    add_threat_badge(11.2, 3.65, "[!] Угроза 5: Атака на Matcher\n(Hill-Climbing по score S)\nЗащита: Квантование score, лимит попыток")
    add_threat_badge(14.75, 3.55, "[!] Угроза 6: Обход замка / реле\nЗащита: RS-485 OSDP внутри чистой зоны")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Диаграмма архитектуры сохранена: {output_path}")


def plot_decision_module_graphs(output_path="reports/lab03_far_frr_decision_graph.png"):
    """
    Строит графики работы модуля принятия решения (Decision Module):
    1) Распределение оценок сходства (Score S) для «Чужих» (Impostor) и «Своих» (Genuine) с порогами T.
    2) Кривые FAR(T), FRR(T), точка EER и зоны FAR-ориентированной vs FRR-ориентированной систем.
    """
    scores = np.linspace(0, 1, 1000)

    mu_imp, sigma_imp = 0.36, 0.11   # Impostors («Чужие»)
    mu_gen, sigma_gen = 0.74, 0.10   # Genuine («Свои»)

    pdf_imp = norm.pdf(scores, mu_imp, sigma_imp)
    pdf_gen = norm.pdf(scores, mu_gen, sigma_gen)

    far = 1.0 - norm.cdf(scores, mu_imp, sigma_imp)
    frr = norm.cdf(scores, mu_gen, sigma_gen)

    eer_idx = np.argmin(np.abs(far - frr))
    t_eer = scores[eer_idx]
    eer_val = (far[eer_idx] + frr[eer_idx]) / 2.0

    t_frr_oriented = 0.45
    t_far_oriented = 0.68

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150, facecolor="#ffffff")
    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#111111")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # --- График 1 ---
    ax1.plot(scores, pdf_imp, color="#d32f2f", lw=2.5, label="Распределение «Чужие» (Impostor)")
    ax1.plot(scores, pdf_gen, color="#2e7d32", lw=2.5, label="Распределение «Свои» (Genuine)")

    mask_frr = scores <= t_eer
    mask_far = scores >= t_eer
    ax1.fill_between(scores[mask_frr], pdf_gen[mask_frr], color="#ffb74d", alpha=0.45, label="Зона FRR (Ложный отказ «своему»)")
    ax1.fill_between(scores[mask_far], pdf_imp[mask_far], color="#ef5350", alpha=0.45, label="Зона FAR (Ложный допуск «чужого»)")

    ax1.axvline(t_frr_oriented, color="#1976d2", linestyle="--", lw=2.0,
                label=f"T_удобство = {t_frr_oriented:.2f} (FRR-ориент.)")
    ax1.axvline(t_eer, color="#111111", linestyle="-", lw=2.2,
                label=f"T_EER = {t_eer:.2f} (Баланс, EER={eer_val*100:.1f}%)")
    ax1.axvline(t_far_oriented, color="#6a1b9a", linestyle="-.", lw=2.0,
                label=f"T_безопасность = {t_far_oriented:.2f} (FAR-ориент.)")

    ax1.set_title("Распределение Score S и выбор порога T в Decision Module", fontsize=12, fontweight="bold", color="#111111")
    ax1.set_xlabel("Оценка биометрического сходства S (Match Score)", fontsize=10.5, color="#111111")
    ax1.set_ylabel("Плотность вероятности p(S)", fontsize=10.5, color="#111111")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(pdf_gen.max(), pdf_imp.max()) * 1.15)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#999999")
    leg1 = ax1.legend(loc="upper left", fontsize=8.5, facecolor="#ffffff", edgecolor="#cccccc")
    for text in leg1.get_texts():
        text.set_color("#111111")

    # --- График 2 ---
    ax2.plot(scores, far * 100, color="#d32f2f", lw=2.5, label="FAR(T) — Ложный допуск чужого (%)")
    ax2.plot(scores, frr * 100, color="#1976d2", lw=2.5, label="FRR(T) — Ложный отказ своему (%)")

    ax2.scatter([t_eer], [eer_val * 100], color="#111111", s=70, zorder=5)
    ax2.annotate(
        f"Точка EER = {eer_val*100:.2f}%\nпри T = {t_eer:.2f}",
        xy=(t_eer, eer_val * 100), xytext=(t_eer - 0.22, eer_val * 100 + 22),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#111111"),
        fontsize=9.5, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", edgecolor="#fbc02d")
    )

    far_at_sec = (1.0 - norm.cdf(t_far_oriented, mu_imp, sigma_imp)) * 100
    frr_at_sec = norm.cdf(t_far_oriented, mu_gen, sigma_gen) * 100
    far_at_conv = (1.0 - norm.cdf(t_frr_oriented, mu_imp, sigma_imp)) * 100
    frr_at_conv = norm.cdf(t_frr_oriented, mu_gen, sigma_gen) * 100

    ax2.axvline(t_frr_oriented, color="#1976d2", linestyle="--", lw=1.8)
    ax2.axvline(t_far_oriented, color="#6a1b9a", linestyle="-.", lw=1.8)

    ax2.axvspan(0.30, 0.50, color="#e3f2fd", alpha=0.5, label=f"FRR-ориент. (Удобство: FRR={frr_at_conv:.2f}%, FAR={far_at_conv:.1f}%)")
    ax2.axvspan(0.63, 0.85, color="#f3e5f5", alpha=0.5, label=f"FAR-ориент. (Безопасность: FAR={far_at_sec:.2f}%, FRR={frr_at_sec:.1f}%)")

    ax2.set_title("Кривые ошибок FAR(T) и FRR(T) от порога Decision Module", fontsize=12, fontweight="bold", color="#111111")
    ax2.set_xlabel("Порог принятия решения T (Decision Threshold)", fontsize=10.5, color="#111111")
    ax2.set_ylabel("Вероятность ошибки (%)", fontsize=10.5, color="#111111")
    ax2.set_xlim(0.15, 0.95)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#999999")
    leg2 = ax2.legend(loc="center left", fontsize=8.5, facecolor="#ffffff", edgecolor="#cccccc")
    for text in leg2.get_texts():
        text.set_color("#111111")

    plt.suptitle("Занятие 3. Математическая модель модуля принятия решения (Decision Module СКУД)",
                 fontsize=13, fontweight="bold", color="#111111", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"График FAR/FRR Decision Module сохранён: {output_path}")


if __name__ == "__main__":
    draw_architecture_diagram()
    plot_decision_module_graphs()
