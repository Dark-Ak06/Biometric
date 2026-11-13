"""
Универсальный мастер-раннер для всех лабораторных работ по курсу Биометрии (Лаб 1..14)
Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041
Репозиторий: https://github.com/Dark-Ak06/Biometric

Запуск:
  python src/run_labs.py --all          # Запустить все 12 работ подряд и пересобрать отчёты
  python src/run_labs.py --lab 14       # Запустить конкретную лабу (1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
  python src/run_labs.py --export       # Экспортировать все отчёты в Word (.docx)
  python src/run_labs.py --notebooks    # Сгенерировать интерактивные Jupyter Notebooks
  python src/run_labs.py --check        # Проверить зависимости и библиотеки
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Гарантируем корректный вывод UTF-8 в Windows-консоль
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

LAB_SCRIPTS = {
    "1": ("Занятие 1: Тест окружения и захват камеры", "test_environment.py"),
    "3": ("Занятие 3: СКУД и модель принятия решений", "lab03_access_control.py"),
    "4": ("Занятие 4: Анализ датасетов и split по субъектам", "load_dataset.py"),
    "5": ("Занятие 5: Распознавание отпечатков пальцев", "fingerprint_matching.py"),
    "6": ("Занятие 6: Детекция лиц (Haar Cascade vs DNN)", "face_detection.py"),
    "8": ("Занятие 8: Голосовая аутентификация (MFCC)", "voice_authentication.py"),
    "9": ("Занятие 9: Предобработка изображений и LBPH", "image_preprocessing.py"),
    "10": ("Занятие 10: Расчёт метрик точности (FAR, FRR, EER)", "lab10_metrics_eer.py"),
    "11": ("Занятие 11: Анализ спуфинг-атак и Liveness (PAD)", "lab11_spoofing_liveness.py"),
    "12": ("Занятие 12: Мультимодальная аутентификация (Score Fusion)", "lab12_multimodal_fusion.py"),
    "13": ("Занятие 13: Биометрическая аутентификация + MFA (TOTP)", "lab13_biometric_mfa.py"),
    "14": ("Занятие 14: Защита приватности и BioHashing", "lab14_privacy_biohashing.py"),
}


def check_dependencies():
    """Проверяет наличие необходимых библиотек и выдаёт понятную подсказку."""
    print("=" * 70)
    print("ПРОВЕРКА БИБЛИОТЕК И ОКРУЖЕНИЯ")
    print("=" * 70)
    
    packages = [
        ("OpenCV", "cv2"),
        ("OpenCV contrib (cv2.face)", "cv2.face"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Matplotlib", "matplotlib"),
        ("Scikit-Learn", "sklearn"),
        ("Scikit-Image", "skimage"),
        ("SoundFile", "soundfile"),
        ("Python-Docx", "docx"),
        ("PyOTP (MFA TOTP)", "pyotp"),
        ("QRCode (MFA QR)", "qrcode"),
        ("PyPDF", "pypdf"),
    ]
    
    all_ok = True
    for name, mod in packages:
        try:
            if "." in mod:
                parent, child = mod.split(".", 1)
                p = __import__(parent)
                has_sub = hasattr(p, child)
                if not has_sub:
                    raise ImportError(f"Отсутствует подмодуль {mod}")
            else:
                __import__(mod)
            print(f"  [OK] {name:<30} — установлен")
        except Exception as e:
            print(f"  [!] {name:<30} — НЕ НАЙДЕН ({e})")
            all_ok = False
            
    if all_ok:
        print("\n[УСПЕХ] Все ключевые библиотеки установлены и готовы к работе!")
    else:
        print("\n[РЕКОМЕНДАЦИЯ] Для установки всех зависимостей выполните:")
        print("  pip install -r requirements.txt")
    print("=" * 70)
    return all_ok


def run_single_lab(lab_key):
    """Запускает скрипт конкретной лабораторной работы."""
    if lab_key not in LAB_SCRIPTS:
        print(f"[ОШИБКА] Неизвестный номер работы: {lab_key}. Доступные: {list(LAB_SCRIPTS.keys())}")
        return False
        
    title, script_name = LAB_SCRIPTS[lab_key]
    script_path = SRC_DIR / script_name
    
    print("\n" + "#" * 70)
    print(f"ЗАПУСК: {title}")
    print(f"Скрипт: {script_path.relative_to(PROJECT_ROOT)}")
    print("#" * 70 + "\n")
    
    t0 = time.time()
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    
    if res.returncode == 0:
        print(f"\n[OK] {title} успешно завершена за {elapsed:.1f} сек.")
        return True
    else:
        print(f"\n[ОШИБКА] Код возврата {res.returncode} при выполнении {script_name}.")
        return False


def run_all_labs():
    """Последовательно запускает все лабораторные работы и генерирует отчёты."""
    print("=" * 70)
    print("ПОЛНЫЙ ПАКЕТНЫЙ ЗАПУСК ВСЕХ 12 ЛАБОРАТОРНЫХ РАБОТ")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)
    
    check_dependencies()
    
    results = {}
    for key in LAB_SCRIPTS.keys():
        success = run_single_lab(key)
        results[key] = success
        
    print("\n" + "=" * 70)
    print("ГЕНЕРАЦИЯ БЛОКНОТОВ И ОТЧЁТОВ В ФОРМАТЕ WORD (.DOCX)")
    print("=" * 70)
    subprocess.run([sys.executable, str(SRC_DIR / "create_notebooks.py")], cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable, str(SRC_DIR / "export_reports_docx.py")], cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable, str(SRC_DIR / "export_defense_guide_docx.py")], cwd=str(PROJECT_ROOT))
    
    print("\n" + "=" * 70)
    print("СВОДНЫЙ СТАТУС ВЫПОЛНЕНИЯ РАБОТ:")
    print("=" * 70)
    for key, ok in results.items():
        title = LAB_SCRIPTS[key][0]
        status_str = "[УСПЕХ]" if ok else "[СБОЙ]"
        print(f"  {status_str} {title}")
    print("=" * 70)


def export_word():
    """Компилирует отчеты и шпаргалку к защите в Word."""
    print("Сборка отчётов Word...")
    subprocess.run([sys.executable, str(SRC_DIR / "export_reports_docx.py")], cwd=str(PROJECT_ROOT))
    subprocess.run([sys.executable, str(SRC_DIR / "export_defense_guide_docx.py")], cwd=str(PROJECT_ROOT))
    print("[УСПЕХ] Все документы Word обновлены в папке reports/!")


def export_notebooks():
    """Генерирует Jupyter Notebooks."""
    print("Генерация интерактивных блокнотов Jupyter...")
    subprocess.run([sys.executable, str(SRC_DIR / "create_notebooks.py")], cwd=str(PROJECT_ROOT))
    print("[УСПЕХ] Все блокноты обновлены в папке notebooks/!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный запуск лабораторных работ по биометрии")
    parser.add_argument("--lab", type=str, help="Номер лабораторной работы (1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)")
    parser.add_argument("--all", action="store_true", help="Запустить все работы подряд")
    parser.add_argument("--check", action="store_true", help="Проверить библиотеки и зависимости")
    parser.add_argument("--export", action="store_true", help="Сгенерировать Word-отчёты (.docx)")
    parser.add_argument("--notebooks", action="store_true", help="Сгенерировать блокноты Jupyter (.ipynb)")
    args = parser.parse_args()

    if args.check:
        check_dependencies()
    elif args.lab:
        run_single_lab(args.lab)
    elif args.export:
        export_word()
    elif args.notebooks:
        export_notebooks()
    elif args.all:
        run_all_labs()
    else:
        # Интерактивное консольное меню, если запущено без параметров
        while True:
            print("\n" + "=" * 70)
            print("ЛАБОРАТОРНЫЕ РАБОТЫ ПО БИОМЕТРИИ (Аканов Дархан, SIS-2309)")
            print("=" * 70)
            for k, (title, _) in LAB_SCRIPTS.items():
                print(f"{k:>2}. {title}")
            print("-" * 70)
            print(" E. Собрать все отчёты в Word (.docx)")
            print(" N. Сгенерировать интерактивные блокноты Jupyter (.ipynb)")
            print(" A. Запустить ВСЕ 12 работ подряд (пакетный режим)")
            print(" C. Проверить зависимости библиотек")
            print(" Q. Выход")
            print("=" * 70)
            choice = input("Выберите пункт меню и нажмите Enter: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'a':
                run_all_labs()
            elif choice == 'e':
                export_word()
            elif choice == 'n':
                export_notebooks()
            elif choice == 'c':
                check_dependencies()
            elif choice in LAB_SCRIPTS:
                run_single_lab(choice)
            else:
                print("Неверный ввод, повторите попытку.")
