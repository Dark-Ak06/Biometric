"""
Генерация интерактивных блокнотов Jupyter для всех лабораторных работ курса (Лаб 8, 9, 10, 11, 12, 13, 14).
Исполнитель: Студент гр. SIS-2309 Аканов Дархан (ID: 38041)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(title, subtitle, lab_num, description, code_cells):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Занятие {lab_num}. {title}\n",
                f"## {subtitle}\n",
                "\n",
                "**Дисциплина:** Биометрические системы аутентификации  \n",
                "**Выполнил:** студент гр. SIS-2309 Аканов Дархан (ID: 38041)\n",
                "\n",
                f"**Цель работы:** {description}"
            ]
        }
    ]

    for section_title, code_str in code_cells:
        if section_title:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {section_title}"]
            })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code_str]
        })

    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def main():
    notebooks = {}

    # Общий надёжный преамбул для нахождения корня и перезагрузки модулей
    def make_preamble(mod_name):
        return (
            "import os, sys, importlib\n"
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n\n"
            "# 1. Гарантированный поиск корня проекта без риска потери рабочей директории\n"
            "curr = Path.cwd()\n"
            "PROJECT_ROOT = curr if (curr / 'src').is_dir() else (curr.parent if (curr.parent / 'src').is_dir() else Path.cwd())\n"
            "src_dir = PROJECT_ROOT / 'src'\n"
            "if str(src_dir) not in sys.path:\n"
            "    sys.path.insert(0, str(src_dir))\n\n"
            f"# 2. Импорт и принудительная перезагрузка {mod_name} из диска (без необходимости рестарта Kernel)\n"
            f"import {mod_name}\n"
            f"importlib.reload({mod_name})\n"
        )

    # Lab 8
    notebooks["lab08_voice_authentication.ipynb"] = create_notebook(
        "Голосовая биометрическая аутентификация",
        "Извлечение MFCC признаков и косинусное сопоставление",
        8,
        "Реализовать базовую верификацию диктора по голосу с использованием MFCC-признаков и косинусного сходства.",
        [
            ("1. Извлечение 13 мел-кепстральных коэффициентов (MFCC)",
             make_preamble("voice_authentication") + "\n"
             "sample_path = PROJECT_ROOT / 'datasets' / 'voice_samples' / 'darkhan_akanov' / 'phrase_01.wav'\n"
             "y, sr, mfcc, stft, mel, emb = voice_authentication.extract_embedding(sample_path)\n"
             "print(f'Длина: {len(y)} отсчётов ({len(y)/sr:.2f} сек), MFCC shape: {mfcc.shape}, Embedding: {emb.shape}')"),
            ("2. Запуск конвейера верификации голоса и расчёт EER",
             "test_records, opt_th, eer = voice_authentication.run_voice_pipeline()\n"
             "print(f'Готово! Оптимальный порог T* = {opt_th:.3f}, EER = {eer:.1f}%')\n\n"
             "plot_path = PROJECT_ROOT / 'reports' / 'lab08_voice_features_analysis.png'\n"
             "if plot_path.exists():\n"
             "    display(Image(filename=str(plot_path)))")
        ]
    )

    # Lab 9
    notebooks["lab09_image_preprocessing.ipynb"] = create_notebook(
        "Предобработка биометрических изображений с OpenCV",
        "Эквализация гистограмм, фильтрация шума, аффинное выравнивание и сканирование лица",
        9,
        "Освоить методы предобработки биометрических изображений (эквализация CLAHE, фильтрация шума, выравнивание по глазам), провести мгновенное сканирование лица с веб-камеры и оценить распознавание LBPH.",
        [
            ("1. Сканирование лица с веб-камеры и запуск конвейера предобработки",
             make_preamble("image_preprocessing") + "\n"
             "# Запуск сканирования лица с веб-камеры в реальном времени и полного бенчмарка\n"
             "# Для симуляции чужака ('НЕ Я'): передайте force_impostor=True\n"
             "print('Запуск Занятия 9: сканирование лица и оценка предобработки...')\n"
             "image_preprocessing.main(scan_webcam=True, force_impostor=False)"),
            ("2. Просмотр результатов сканирования и влияния предобработки на LBPH",
             "scan_img = PROJECT_ROOT / 'reports' / 'lab09_live_scan_result.png'\n"
             "if scan_img.exists():\n"
             "    print('Результат сканирования лица с веб-камеры:')\n"
             "    display(Image(filename=str(scan_img)))\n\n"
             "bench_img = PROJECT_ROOT / 'reports' / 'lab09_lbph_recognition_benchmark.png'\n"
             "if bench_img.exists():\n"
             "    print('Бенчмарк влияния методов предобработки на LBPH:')\n"
             "    display(Image(filename=str(bench_img)))")
        ]
    )

    # Lab 10
    notebooks["lab10_far_frr_eer.ipynb"] = create_notebook(
        "Расчёт метрик биометрической точности (FAR, FRR, EER)",
        "Оценка плотностей распределений, построение кривых ошибок и ROC-анализ",
        10,
        "Провести расчёт ключевых вероятностных метрик биометрической точности (FAR, FRR, EER, AUC), визуализировать распределения Genuine и Impostor оценок и обосновать выбор порогов верификации.",
        [
            ("Запуск моделирования метрик точности и построение ROC-кривой",
             make_preamble("lab10_metrics_eer") + "\n"
             "lab10_metrics_eer.main()\n\n"
             "roc_plot = PROJECT_ROOT / 'reports' / 'lab10_roc_curve.png'\n"
             "if roc_plot.exists():\n"
             "    display(Image(filename=str(roc_plot)))")
        ]
    )

    # Lab 11
    notebooks["lab11_spoofing_liveness.ipynb"] = create_notebook(
        "Анализ спуфинг-атак и детекция живого лица (PAD / Liveness Detection)",
        "Пассивный анализ текстуры (Лапласиан) и активный Challenge-Response (детекция моргания)",
        11,
        "Исследовать спуфинг-атаки (предъявление фото/экрана) на биометрическую систему лица, реализовать алгоритмы Presentation Attack Detection (текстурный анализ лапласиана и отслеживание моргания) и оценить метрики APCER и BPCER.",
        [
            ("Запуск тестирования Liveness Detection и расчёт APCER/BPCER",
             make_preamble("lab11_spoofing_liveness") + "\n"
             "lab11_spoofing_liveness.main()\n\n"
             "pad_plot = PROJECT_ROOT / 'reports' / 'lab11_liveness_detection.png'\n"
             "if pad_plot.exists():\n"
             "    display(Image(filename=str(pad_plot)))")
        ]
    )

    # Lab 12
    notebooks["lab12_multimodal_fusion.ipynb"] = create_notebook(
        "Мультимодальная аутентификация: слияние лица и голоса",
        "Нормализация Min-Max, Score-level Fusion и оптимизация весов по критерию минимума EER",
        12,
        "Разработать и оценить мультимодальную систему аутентификации, объединяющую биометрию лица (LBPH) и голоса (MFCC) на уровне решений (Score Fusion) с оптимизацией весовых коэффициентов.",
        [
            ("Запуск мультимодального слияния (Score Fusion) и сравнение EER",
             make_preamble("lab12_multimodal_fusion") + "\n"
             "lab12_multimodal_fusion.main()\n\n"
             "fuse_plot = PROJECT_ROOT / 'reports' / 'lab12_multimodal_roc_comparison.png'\n"
             "if fuse_plot.exists():\n"
             "    display(Image(filename=str(fuse_plot)))")
        ]
    )

    # Lab 13
    notebooks["lab13_biometric_mfa.ipynb"] = create_notebook(
        "Биометрическая аутентификация + MFA (Multi-Factor Authentication)",
        "Интеграция TOTP (RFC 6238), генерация QR-кода, защита от повторов и Fallback-сценарий",
        13,
        "Реализовать многофакторную схему аутентификации (MFA), объединяющую биометрический фактор (лицо/голос) с генератором одноразовых паролей по времени (TOTP), сформировать QR-код для Google Authenticator и разработать резервный Fallback-контур.",
        [
            ("Запуск сценариев биометрической MFA и проверка перехода в Fallback",
             make_preamble("lab13_biometric_mfa") + "\n"
             "lab13_biometric_mfa.main()\n\n"
             "mfa_plot = PROJECT_ROOT / 'reports' / 'lab13_mfa_scenarios.png'\n"
             "if mfa_plot.exists():\n"
             "    display(Image(filename=str(mfa_plot)))")
        ]
    )

    # Lab 14
    notebooks["lab14_privacy_biohashing.ipynb"] = create_notebook(
        "Защита биометрических данных и приватность (Privacy & BioHashing)",
        "Ортогональные проекции BioCode, отзываемость Cancelable Biometrics и аудит по GDPR / закону РК",
        14,
        "Изучить требования GDPR ст. 9 и закона РК о персональных данных, реализовать схему отменяемой биометрии BioHashing, доказать статистическую независимость при смене ключа и провести аудит приватности системы.",
        [
            ("Запуск экспериментов BioHashing и демонстрация отзываемости ключа",
             make_preamble("lab14_privacy_biohashing") + "\n"
             "lab14_privacy_biohashing.main()\n\n"
             "bio_plot = PROJECT_ROOT / 'reports' / 'lab14_biohashing_scheme.png'\n"
             "if bio_plot.exists():\n"
             "    display(Image(filename=str(bio_plot)))")
        ]
    )

    print("Генерация Jupyter Notebooks в папку notebooks/...")
    for filename, nb_content in notebooks.items():
        out_path = NOTEBOOKS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_content, f, ensure_ascii=False, indent=2)
        print(f"  -> {filename}")

    print("[OK] Все блокноты успешно сгенерированы.")


if __name__ == "__main__":
    main()
