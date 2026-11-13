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

    # Lab 8
    notebooks["lab08_voice_authentication.ipynb"] = create_notebook(
        "Голосовая биометрическая аутентификация",
        "Извлечение MFCC признаков и косинусное сопоставление",
        8,
        "Реализовать базовую верификацию диктора по голосу с использованием MFCC-признаков и косинусного сходства.",
        [
            ("1. Извлечение 13 мел-кепстральных коэффициентов (MFCC)",
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nfrom voice_authentication import extract_embedding, run_voice_pipeline\n\nsample_path = '../datasets/voice_samples/darkhan_akanov/phrase_01.wav'\ny, sr, mfcc, stft, mel, emb = extract_embedding(sample_path)\nprint(f'Длина: {len(y)} отсчётов ({len(y)/sr:.2f} сек), MFCC shape: {mfcc.shape}, Embedding: {emb.shape}')"),
            ("2. Запуск конвейера верификации и расчёт EER",
             "os.chdir('..')\ntest_records, opt_th, eer = run_voice_pipeline()\nos.chdir('notebooks')\nprint(f'Готово! Оптимальный порог T* = {opt_th:.3f}, EER = {eer:.1f}%')")
        ]
    )

    # Lab 9
    notebooks["lab09_image_preprocessing.ipynb"] = create_notebook(
        "Предобработка биометрических изображений с OpenCV",
        "Эквализация гистограмм, фильтрация шума и аффинное выравнивание лиц",
        9,
        "Освоить типовые методы предобработки биометрических изображений (эквализация, фильтрация шума, выравнивание лица по глазам) и оценить их влияние на распознавание LBPH.",
        [
            ("Запуск конвейера предобработки и бенчмарка LBPH",
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nimport image_preprocessing\n\nos.chdir('..')\nimage_preprocessing.main()\nos.chdir('notebooks')\nprint('Бенчмарк предобработки и распознавания успешно завершён!')")
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
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nimport lab10_metrics_eer\n\nos.chdir('..')\nlab10_metrics_eer.main()\nos.chdir('notebooks')\nprint('Расчёт FAR/FRR/EER завершён!')")
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
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nimport lab11_spoofing_liveness\n\nos.chdir('..')\nlab11_spoofing_liveness.main()\nos.chdir('notebooks')\nprint('Тестирование Presentation Attack Detection успешно выполнено!')")
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
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nimport lab12_multimodal_fusion\n\nos.chdir('..')\nlab12_multimodal_fusion.main()\nos.chdir('notebooks')\nprint('Оптимизация мультимодального весового коэффициента завершена!')")
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
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nimport lab13_biometric_mfa\n\nos.chdir('..')\nlab13_biometric_mfa.main()\nos.chdir('notebooks')\nprint('MFA конвейер успешно протестирован!')")
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
             "import os, sys\nsys.path.append(os.path.abspath('../src'))\nimport lab14_privacy_biohashing\n\nos.chdir('..')\nlab14_privacy_biohashing.main()\nos.chdir('notebooks')\nprint('BioHashing эксперименты и аудит приватности завершены!')")
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
