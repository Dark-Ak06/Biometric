"""
Утилита для сбора персональных биометрических данных студента (Аканов Дархан, гр. SIS-2309):
1. Запись голосовых образцов через микрофон (для Занятия 8).
2. Захват фотографий лица через веб-камеру (для Занятия 9).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Гарантируем корректный вывод UTF-8 в Windows-консоль
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = PROJECT_ROOT / "datasets" / "voice_samples" / "darkhan_akanov"
FACE_DIR = PROJECT_ROOT / "datasets" / "darkhan_face"


def print_status():
    """Выводит статус наличия персональных данных."""
    print("=" * 60)
    print("СТАТУС ПЕРСОНАЛЬНЫХ ДАННЫХ: Аканов Дархан (SIS-2309)")
    print("=" * 60)
    
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    
    wav_files = sorted(list(VOICE_DIR.glob("*.wav")))
    img_files = sorted(list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png")))
    
    print(f"1. Голосовые образцы (Занятие 8): {len(wav_files)} шт. в {VOICE_DIR.relative_to(PROJECT_ROOT)}")
    for wf in wav_files:
        print(f"   - {wf.name} ({wf.stat().st_size / 1024:.1f} КБ)")
    if len(wav_files) >= 5:
        print("   [OK] Достаточно образцов для enrollment и verification!")
    else:
        print("   [!] Рекомендуется записать 5 фраз: python src/capture_my_data.py --voice")
        
    print(f"\n2. Фотографии лица (Занятие 9): {len(img_files)} шт. в {FACE_DIR.relative_to(PROJECT_ROOT)}")
    for im in img_files:
        print(f"   - {im.name} ({im.stat().st_size / 1024:.1f} КБ)")
    if len(img_files) >= 5:
        print("   [OK] Фотографии лица присутствуют!")
    else:
        print("   [!] Можно сделать фото с веб-камеры: python src/capture_my_data.py --face")
    print("=" * 60)


def record_voice(num_phrases=5, duration=3.5, sr=16000):
    """Записывает голосовые фразы с микрофона через sounddevice."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("[ОШИБКА] Библиотеки sounddevice или soundfile не установлены.")
        print("Выполните: pip install sounddevice soundfile")
        return

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"ЗАПИСЬ ГОЛОСОВЫХ ОБРАЗЦОВ ДЛЯ АУТЕНТИФИКАЦИИ ({num_phrases} фраз)")
    print(f"Папка сохранения: {VOICE_DIR}")
    print("=" * 60)
    
    suggested_phrases = [
        "Мой голос — мой персональный биометрический ключ.",
        "Биометрическая система аутентификации диктора активна.",
        "Аканов Дархан подтверждает доступ к защищённому контуру.",
        "Проверка частотных мел-кепстральных признаков речи.",
        "Успешное завершение калибровки голосового профиля."
    ]

    for i in range(1, num_phrases + 1):
        phrase_text = suggested_phrases[i - 1] if i - 1 < len(suggested_phrases) else f"Тестовая фраза #{i}"
        print(f"\n[Образец {i}/{num_phrases}]")
        print(f"Фраза для чтения: «{phrase_text}»")
        input("Нажмите [Enter], чтобы начать запись...")
        
        for cd in [3, 2, 1]:
            print(f"Запись начнётся через {cd}...", end="\r", flush=True)
            time.sleep(1)
        print(">>> ИДЁТ ЗАПИСЬ... ГОВОРИТЕ! <<<          ")
        
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        print("[Запись завершена]")
        
        # Удаляем DC-смещение и нормализуем
        audio = audio.flatten()
        audio -= np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 1e-4:
            audio = 0.9 * (audio / max_val)
            
        save_path = VOICE_DIR / f"phrase_{i:02d}.wav"
        sf.write(str(save_path), audio, sr)
        print(f"Сохранено: {save_path.name}")
        
    print("\n[УСПЕХ] Все голосовые образцы успешно записаны!")
    print_status()


def capture_face(num_photos=5):
    """Захватывает серию фото лица с веб-камеры с детекцией лица."""
    try:
        import cv2
    except ImportError:
        print("[ОШИБКА] OpenCV не установлен. Выполните: pip install opencv-python")
        return

    FACE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"СЪЁМКА ФОТОГРАФИЙ ЛИЦА ({num_photos} кадров)")
    print(f"Папка сохранения: {FACE_DIR}")
    print("Подсказка: смотрите в камеру, слегка меняйте ракурс и освещение.")
    print("Нажмите [Пробел] для снимка кадра, [Q] для досрочного выхода.")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ОШИБКА] Не удалось открыть веб-камеру. Проверьте подключение камеры.")
        return

    cascade_path = PROJECT_ROOT / "datasets" / "models" / "haarcascade_frontalface_default.xml"
    if cascade_path.exists():
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    instructions = [
        "Кадр 1: Фронтальное лицо (нормальный свет)",
        "Кадр 2: Небольшой наклон головы вправо",
        "Кадр 3: Небольшой наклон головы влево",
        "Кадр 4: Лицо с улыбкой / эмоцией",
        "Кадр 5: Изменение угла освещения"
    ]

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(100, 100))

        inst = instructions[count] if count < len(instructions) else f"Кадр {count + 1}"
        cv2.putText(vis, f"[{count + 1}/{num_photos}] {inst}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(vis, "Нажмите [ПРОБЕЛ] - снимок, [Q] - выход", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Darkhan Face (Space: Snap, Q: Quit)", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            save_path = FACE_DIR / f"darkhan_{count + 1:02d}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"[+] Сохранён {save_path.name}")
            count += 1
            time.sleep(0.3)
        elif key == ord('q') or key == 27:
            print("Прервано пользователем.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[УСПЕХ] Захвачено {count} фото!")
    print_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сбор биометрических данных Аканова Дархана (SIS-2309)")
    parser.add_argument("--status", action="store_true", help="Проверить наличие данных")
    parser.add_argument("--voice", action="store_true", help="Записать голос через микрофон")
    parser.add_argument("--face", action="store_true", help="Снять лицо через веб-камеру")
    args = parser.parse_args()

    if args.voice:
        record_voice()
    elif args.face:
        capture_face()
    else:
        print_status()
        print("\nИспользование:")
        print("  python src/capture_my_data.py --status   (проверить текущие файлы)")
        print("  python src/capture_my_data.py --voice    (записать фразы в микрофон)")
        print("  python src/capture_my_data.py --face     (снять лицо на веб-камеру)")
