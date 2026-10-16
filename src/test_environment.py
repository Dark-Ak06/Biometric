# test_environment.py — проверка окружения (Занятие 1)
import os
import sys
import cv2
import numpy as np

print("=" * 50)
print("Проверка установленных библиотек")
print("=" * 50)
print(f"Python version:       {sys.version.split()[0]}")
print(f"OpenCV version:       {cv2.__version__}")
print(f"NumPy version:        {np.__version__}")

try:
    import matplotlib
    print(f"Matplotlib version:   {matplotlib.__version__}")
except ImportError:
    print("Matplotlib: не установлен")

try:
    import skimage
    print(f"scikit-image version: {skimage.__version__}")
except ImportError:
    print("scikit-image: не установлен")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn: не установлен")

print("=" * 50)

# Создаём папку reports если её нет
os.makedirs("reports", exist_ok=True)

# Проверка веб-камеры
print("\nПроверка веб-камеры...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Камера не найдена")
else:
    ret, frame = cap.read()
    if ret:
        output_path = os.path.join("reports", "camera_test.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Кадр успешно сохранён в {output_path}")
        print(f"Размер кадра: {frame.shape[1]}x{frame.shape[0]} пикселей")
    else:
        print("Не удалось получить кадр с камеры")
    cap.release()

print("\nПроверка окружения завершена!")
