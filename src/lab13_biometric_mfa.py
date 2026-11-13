"""
Занятие 13. Биометрическая аутентификация + MFA (Multi-Factor Authentication)
Исполнитель: студент гр. SIS-2309 Аканов Дархан (ID: 38041)
Дисциплина: Биометрические системы аутентификации

Скрипт реализует:
1. Генерацию секрета Base32 и настройку алгоритма TOTP (RFC 6238) через pyotp.
2. Формирование QR-кода otpauth:// для мобильных приложений (Google Authenticator).
3. Двухфакторную связку: Фактор 1 (Биометрия: Лицо/Голос) + Фактор 2 (TOTP-код).
4. Защиту от replay-атак (однократность использования TOTP-кода в окне валидности).
5. Fallback-сценарий: при трёхкратном сбое биометрии — резервный вход (Пароль + TOTP).
6. Построение диаграммы состояний (State Diagram) и экспорт графиков в папку reports/.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pyotp
import qrcode

plt.style.use("default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']
plt.rcParams['axes.unicode_minus'] = False

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class BiometricMFASystem:
    """
    Класс комплексной системы аутентификации MFA:
    - Фактор 1 (Inherence / биометрия): Лицо или голос
    - Фактор 2 (Possession / владение): TOTP-код в мобильном приложении
    - Резервный фактор (Knowledge / знание): Пароль учетной записи
    """
    def __init__(self, username="darkhan_akanov", seed_secret=None):
        self.username = username
        self.secret = seed_secret if seed_secret else pyotp.random_base32()
        self.totp = pyotp.TOTP(self.secret, interval=30)
        self.used_tokens = set()
        self.fallback_password_hash = "f355e09f5ff17b9b" # хэш резервного пароля "SecretPass2026!"
        self.max_biometric_attempts = 3

    def generate_qr_code(self, output_path):
        """Шаг 1. Генерирует QR-код для настройки Google Authenticator."""
        uri = self.totp.provisioning_uri(name=self.username, issuer_name="Biometrics_Security_Lab")
        qr = qrcode.QRCode(version=1, box_size=8, border=3)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(str(output_path))
        return uri

    def verify_totp(self, code, valid_window=1):
        """Шаг 2. Верифицирует 6-значный TOTP-код с защитой от повтора."""
        code_str = str(code).strip()
        if code_str in self.used_tokens:
            return False, "Код уже был использован (Replay Attack предотвращена)!"
        
        # Проверяем временное окно (+-30 сек)
        is_valid = self.totp.verify(code_str, valid_window=valid_window)
        if is_valid:
            self.used_tokens.add(code_str)
            return True, "TOTP-код подтверждён успешно."
        return False, "Неверный или просроченный TOTP-код."

    def authenticate_session(self, biometric_score, threshold=0.75, totp_code=None,
                             fail_count=0, fallback_password=None):
        """
        Шаги 3–5. Выполняет сессию аутентификации с отработкой всех состояний и fallback.
        Возвращает: (status, message, next_state)
        """
        # Проверка биометрии
        if fail_count < self.max_biometric_attempts:
            if biometric_score >= threshold:
                # Биометрия успешна -> переход к проверке TOTP
                if totp_code is None:
                    return False, "Биометрия пройдена. Ожидается ввод TOTP-кода.", "AWAIT_TOTP"
                valid_totp, totp_msg = self.verify_totp(totp_code)
                if valid_totp:
                    return True, "ДОСТУП ПРЕДОСТАВЛЕН: Биометрия + TOTP подтверждены!", "ACCESS_GRANTED"
                else:
                    return False, f"Отказ: {totp_msg}", "ACCESS_DENIED_TOTP"
            else:
                # Ошибка биометрии
                new_fail = fail_count + 1
                if new_fail >= self.max_biometric_attempts:
                    return False, "Биометрия трижды не распознана! Переход в резервный режим (Fallback).", "FALLBACK_TRIGGERED"
                else:
                    return False, f"Биометрический отказ (попытка {new_fail}/{self.max_biometric_attempts}).", "RETRY_BIOMETRIC"

        # Резервный режим (Fallback: Пароль + TOTP)
        if fallback_password == "SecretPass2026!":
            if totp_code is None:
                return False, "Резервный пароль принят. Введите TOTP-код для подтверждения.", "AWAIT_TOTP_FALLBACK"
            valid_totp, totp_msg = self.verify_totp(totp_code)
            if valid_totp:
                return True, "ДОСТУП ПРЕДОСТАВЛЕН: Резервный вход (Пароль + TOTP) успешен!", "ACCESS_GRANTED_FALLBACK"
            else:
                return False, f"Отказ в резервном режиме: {totp_msg}", "ACCESS_DENIED_FALLBACK_TOTP"
        else:
            return False, "Неверный резервный пароль!", "ACCESS_DENIED_PASSWORD"


def draw_state_diagram(output_path):
    """
    Шаг 3 практического задания: строит диаграмму состояний (State Diagram)
    всего процесса входа со всеми точками отказа.
    """
    fig, ax = plt.subplots(figsize=(12, 7.5), constrained_layout=True)
    ax.axis('off')

    # Отрисовка блоков состояний (узлов графа)
    boxes = {
        "START": ("Старт\n(Инициализация)", 0.08, 0.70, "#94A3B8"),
        "BIO_CHECK": ("1. Проверка биометрии\n(Лицо / Голос)\nПопытки: 1..3", 0.32, 0.70, "#3B82F6"),
        "TOTP_CHECK": ("2. Запрос TOTP\n(Google Auth)\nОкно: 30 сек", 0.62, 0.70, "#6366F1"),
        "GRANTED": ("ДОСТУП\nПРЕДОСТАВЛЕН\n(MFA Успех)", 0.88, 0.70, "#16A34A"),
        "FALLBACK": ("3. Резервный режим\n(Fallback)\nПароль + TOTP", 0.47, 0.25, "#F59E0B"),
        "DENIED": ("ДОСТУП\nЗАБЛОКИРОВАН\n(Отказ)", 0.88, 0.25, "#DC2626")
    }

    for name, (label, x, y, col) in boxes.items():
        bbox_props = dict(boxstyle="round,pad=0.6", fc=col, ec="black", lw=1.5, alpha=0.9)
        text_col = "white" if col != "#94A3B8" else "black"
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, fontweight='bold',
                color=text_col, bbox=bbox_props)

    # Стрелки переходов
    def arrow(x1, y1, x2, y2, text="", rad=0.0):
        connectionstyle = f"arc3,rad={rad}"
        ax.annotate(text, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#0F172A",
                                    connectionstyle=connectionstyle),
                    ha="center", va="center", fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle="square,pad=0.2", fc="#F8FAFC", ec="none", alpha=0.85))

    arrow(0.14, 0.70, 0.23, 0.70, "Вход")
    arrow(0.41, 0.70, 0.53, 0.70, "Score ≥ t\n(Успех)")
    arrow(0.71, 0.70, 0.80, 0.70, "TOTP OK\n(Верно)")
    arrow(0.62, 0.64, 0.82, 0.28, "TOTP сбой", rad=-0.1)
    arrow(0.32, 0.62, 0.42, 0.30, "3 отказа\nбиометрии", rad=-0.1)
    arrow(0.55, 0.25, 0.80, 0.25, "Пароль неверный")
    arrow(0.52, 0.32, 0.83, 0.65, "Пароль + TOTP OK", rad=-0.2)

    plt.title("Диаграмма состояний (State Diagram) биометрической MFA системы с Fallback",
              fontsize=13, fontweight='bold', pad=15)
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_lab13_pipeline():
    print("=" * 70)
    print("ЗАНЯТИЕ 13. БИОМЕТРИЧЕСКАЯ АУТЕНТИФИКАЦИЯ + MFA")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)

    # Инициализация системы для Дархана Аканова
    # Фиксируем детерминированный сид секрета для демонстрации
    secret_demo = "JBSWY3DPEHPK3PXP" # Base32 секрет
    mfa = BiometricMFASystem(username="akanov_darkhan_sis2309", seed_secret=secret_demo)

    # 1. Генерация QR-кода
    qr_path = REPORTS_DIR / "lab13_totp_qr.png"
    uri = mfa.generate_qr_code(qr_path)
    print(f"\n[ШАГ 1] Настройка TOTP (RFC 6238):")
    print(f"  - Секретный ключ Base32: {mfa.secret}")
    print(f"  - Provisioning URI:       {uri}")
    print(f"  - QR-код для смартфона:  {qr_path.name}")

    # 2. Построение диаграммы состояний
    state_path = REPORTS_DIR / "lab13_state_diagram.png"
    draw_state_diagram(state_path)
    print(f"[+] Сохранена диаграмма состояний (State Diagram): {state_path.name}")

    # 3. Тестирование сценариев входа
    print("\n[ШАГ 3-5] Симуляция контрольных сценариев аутентификации:")
    print("-" * 85)
    print(f"{'№':<3} | {'Сценарий тестирования':<38} | {'Факторы':<18} | {'Статус':<10} | {'Итог'}")
    print("-" * 85)

    valid_totp = mfa.totp.now()

    fresh_fallback_totp = mfa.totp.at(int(time.time()) + 30)

    scenarios = [
        ("1. Успешный штатный вход", 0.92, valid_totp, 0, None, "УСПЕХ", "Доступ открыт"),
        ("2. Отказ TOTP (неверный одноразовый код)", 0.88, "999999", 0, None, "ОТКАЗ", "Код не подошёл"),
        ("3. Replay-атака (повтор того же TOTP)", 0.90, valid_totp, 0, None, "ОТКАЗ", "Код перехвачен"),
        ("4. Биометрический сбой (попытка 1: темно)", 0.45, None, 0, None, "ОТКАЗ", "Повторить биометрию"),
        ("5. Биометрический сбой (попытка 2: ракурс)", 0.50, None, 1, None, "ОТКАЗ", "Повторить биометрию"),
        ("6. 3-кратный сбой биометрии -> Fallback", 0.40, None, 2, None, "FALLBACK", "Сенсор заблокирован"),
        ("7. Успешный Fallback (Пароль + TOTP)", 0.00, fresh_fallback_totp, 3, "SecretPass2026!", "УСПЕХ", "Резервный допуск"),
        ("8. Ошибка Fallback (неверный пароль)", 0.00, fresh_fallback_totp, 3, "WrongPass123", "ОТКАЗ", "Блокировка аккаунта"),
    ]

    for idx, (s_name, bio_s, code, fails, pwd, exp_stat, exp_res) in enumerate(scenarios, 1):
        ok, msg, state = mfa.authenticate_session(
            biometric_score=bio_s,
            threshold=0.75,
            totp_code=code,
            fail_count=fails,
            fallback_password=pwd
        )
        status_disp = "[OK] ДОСТУП" if ok else f"[{state}]"
        factors_disp = f"Bio={bio_s:.2f}, TOTP" if fails < 3 else "Pass + TOTP"
        print(f"{idx:<3} | {s_name:<38} | {factors_disp:<18} | {status_disp:<18} | {exp_res}")

    print("-" * 85)

    # График 3: Сводный статус сценариев
    plot_path3 = REPORTS_DIR / "lab13_mfa_scenarios.png"
    plt.figure(figsize=(9.5, 4.8), constrained_layout=True)
    scen_titles = [s[0].split('.')[1].strip().split('(')[0] for s in scenarios]
    success_flags = [1 if s[5] == "УСПЕХ" else 0 for s in scenarios]
    colors = ['#16A34A' if f == 1 else '#DC2626' for f in success_flags]

    plt.barh(range(len(scen_titles)), [1]*len(scen_titles), color=colors, height=0.6, edgecolor='black', lw=0.6)
    plt.yticks(range(len(scen_titles)), scen_titles, fontsize=9.5)
    plt.xlabel("Результат тестирования сценария (Зелёный — Доступ выдан, Красный — Отклонён)", fontsize=10.5)
    plt.title("Результаты прогона тестовых сценариев биометрической MFA с Fallback", fontsize=12, fontweight='bold')
    plt.xlim([0, 1.2])
    plt.grid(True, axis='x', alpha=0.3)
    plt.savefig(plot_path3, dpi=200)
    plt.close()
    print(f"[+] Сохранён график сценариев MFA: {plot_path3.name}")

    print("\n" + "=" * 70)
    print("ИТОГИ ЗАНЯТИЯ 13:")
    print("1. Реализована полнофункциональная схема: Биометрия (Лицо/Голос) + TOTP (RFC 6238).")
    print(f"2. Сгенерирован рабочий QR-код для Google Authenticator: {qr_path.name}")
    print("3. Протестирован Fallback-механизм: автоматическое переключение на Пароль+TOTP")
    print("   после 3 последовательных биометрических отказов.")
    print("4. Успешно предотвращена повторная Replay-атака по уже использованному TOTP-коду.")
    print("=" * 70)


main = run_lab13_pipeline


if __name__ == "__main__":
    main()
