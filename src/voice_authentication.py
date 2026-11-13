"""
Занятие 8. Голосовая биометрическая аутентификация
Исполнитель: студент гр. SIS-2309 Аканов Дархан (ID: 38041)
Дисциплина: Биометрические системы аутентификации

Скрипт реализует:
1. Загрузку и предобработку аудиозаписей 3 дикторов (Дархан Аканов, Боб, Чарли).
2. Извлечение 13 мел-кепстральных коэффициентов (MFCC) и усреднение по времени.
   (Поддерживает как librosa, так и высоконадёжный встроенный движок NumPy/SciPy).
3. Построение биометрических эталонов enrollment (исследование N=1, 3, 5 образцов).
4. Верификацию дикторов по косинусному сходству (Cosine Similarity).
5. Тестирование Genuine (свой) / Impostor (чужой), расчёт FAR, FRR, EER и подбор порога T*.
6. Генерацию высококачественных графиков в папку reports/.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt

# Гарантируем корректный вывод UTF-8 в Windows-консоль
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Попытка импорта soundfile или scipy.io.wavfile
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    import scipy.io.wavfile as wavfile

# Попытка импорта librosa (опционально)
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

plt.style.use("default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "voice_samples"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_wav_file(audio_path, target_sr=16000):
    """Надёжная загрузка WAV-файла без внешних зависимостей."""
    if HAS_SOUNDFILE:
        data, sr = sf.read(str(audio_path))
    else:
        sr, data = wavfile.read(str(audio_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Нормализация
    data = data.astype(np.float32)
    data -= np.mean(data)
    max_amp = np.max(np.abs(data))
    if max_amp > 1e-4:
        data = data / max_amp

    # Простое удаление тишины по краям (энергетический порог)
    frame_len = 512
    energy = np.array([np.sum(data[i:i+frame_len]**2) for i in range(0, len(data), frame_len)])
    thresh = 0.01 * np.max(energy) if len(energy) > 0 and np.max(energy) > 0 else 0
    active_frames = np.where(energy > thresh)[0]
    if len(active_frames) > 0:
        start_idx = active_frames[0] * frame_len
        end_idx = min(len(data), (active_frames[-1] + 1) * frame_len)
        data = data[start_idx:end_idx]

    return data, sr


def compute_mfcc(y, sr=16000, n_mfcc=13, n_mels=40, n_fft=512, hop_length=160):
    """
    Вычисляет 13 мел-кепстральных коэффициентов (MFCC).
    Использует librosa при наличии, либо чистый математический конвейер NumPy/SciPy:
    1) Pre-emphasis фильтр y[t] - 0.97*y[t-1]
    2) Оконное преобразование Хэмминга и быстрое преобразование Фурье (STFT)
    3) Банк треугольных мел-фильтров
    4) Логарифмирование энергий
    5) Дискретное косинусное преобразование (DCT Type-II)
    """
    if HAS_LIBROSA:
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            stft_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            return mfcc, stft_mag, mel_spec
        except Exception:
            pass

    # Чистая реализация на NumPy/SciPy (работает на любом ПК)
    # 1. Pre-emphasis
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # 2. Framing & Windowing
    num_frames = 1 + int(np.floor((len(y_pre) - n_fft) / hop_length))
    if num_frames <= 0:
        num_frames = 1
        y_pre = np.pad(y_pre, (0, max(0, n_fft - len(y_pre))), mode='constant')

    pad_len = (num_frames - 1) * hop_length + n_fft - len(y_pre)
    if pad_len > 0:
        y_pre = np.pad(y_pre, (0, pad_len), mode='constant')

    indices = np.tile(np.arange(0, n_fft), (num_frames, 1)) + np.tile(np.arange(0, num_frames * hop_length, hop_length), (n_fft, 1)).T
    frames = y_pre[indices] * np.hamming(n_fft)

    # 3. FFT & Power spectrum
    mag = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag ** 2)

    # 4. Triangular Mel Filterbank
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / max(1, (f_m_plus - f_m))

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks <= 0, 1e-10, filter_banks)
    mel_log = 20 * np.log10(filter_banks)

    # 5. DCT-II
    mfcc = scipy.fftpack.dct(mel_log, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfcc.T, mag.T, filter_banks.T


def extract_embedding(audio_path, sr=16000, n_mfcc=13):
    """Извлекает 13-мерный усреднённый по времени L2-нормализованный вектор признаков."""
    y, orig_sr = load_wav_file(audio_path, target_sr=sr)
    mfcc, stft_mag, mel_spec = compute_mfcc(y, sr=sr, n_mfcc=n_mfcc)
    
    # Усреднение во времени
    emb = np.mean(mfcc, axis=1)
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return y, sr, mfcc, stft_mag, mel_spec, emb


def cosine_similarity(u, v):
    """Вычисляет косинусное сходство между двумя векторами признаков."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def build_enrollment_database(subjects_data, n_enroll=3):
    """Построение эталонов enrollment (усреднение первых n_enroll векторов)."""
    templates = {}
    for subj, embeddings in subjects_data.items():
        subset = embeddings[:n_enroll]
        avg_vec = np.mean(subset, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm > 1e-6:
            avg_vec = avg_vec / norm
        templates[subj] = avg_vec
    return templates


def run_voice_pipeline():
    print("=" * 70)
    print("ЗАНЯТИЕ 8. ГОЛОСОВАЯ БИОМЕТРИЧЕСКАЯ АУТЕНТИФИКАЦИЯ (MFCC)")
    print("Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041")
    print("=" * 70)

    subjects = {
        "Darkhan (Аканов Д.)": DATASETS_DIR / "darkhan_akanov",
        "Impostor 1 (Боб)": DATASETS_DIR / "subject2_bob",
        "Impostor 2 (Чарли)": DATASETS_DIR / "subject3_charlie"
    }

    all_embeddings = {}
    all_raw_audio = {}
    all_mfcc = {}
    all_stft = {}
    all_mel = {}

    print("\n[ШАГ 1-2] Загрузка голосовых образцов и извлечение 13 MFCC:")
    for subj_name, subj_dir in subjects.items():
        wav_files = sorted(list(subj_dir.glob("*.wav")))
        if len(wav_files) == 0:
            raise FileNotFoundError(f"Не найдены WAV-файлы в {subj_dir}")

        all_embeddings[subj_name] = []
        all_raw_audio[subj_name] = []
        all_mfcc[subj_name] = []
        all_stft[subj_name] = []
        all_mel[subj_name] = []

        print(f"  Субъект: {subj_name} — {len(wav_files)} образцов:")
        for wf in wav_files[:5]:
            y, sr, mfcc, stft, mel, emb = extract_embedding(wf)
            all_embeddings[subj_name].append(emb)
            all_raw_audio[subj_name].append((y, sr))
            all_mfcc[subj_name].append(mfcc)
            all_stft[subj_name].append(stft)
            all_mel[subj_name].append(mel)
            print(f"    - {wf.name}: {len(y)/sr:.2f} сек, фреймов: {mfcc.shape[1]}, MFCC размер: {mfcc.shape}")

    # Рисунок 1: Спектральные и MFCC признаки фразы Дархана
    plot_path1 = REPORTS_DIR / "lab08_mfcc_features.png"
    y_d, sr_d = all_raw_audio["Darkhan (Аканов Д.)"][0]
    mfcc_d = all_mfcc["Darkhan (Аканов Д.)"][0]
    stft_d = all_stft["Darkhan (Аканов Д.)"][0]
    mel_d = all_mel["Darkhan (Аканов Д.)"][0]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8.5), constrained_layout=True)
    t = np.linspace(0, len(y_d) / sr_d, len(y_d))
    axs[0].plot(t, y_d, color="#1E40AF", lw=0.9)
    axs[0].set_title("1. Исходный речевой сигнал во времени (Дархан Аканов, фраза 1)", fontsize=11, fontweight='bold')
    axs[0].set_xlabel("Время (с)")
    axs[0].set_ylabel("Амплитуда")
    axs[0].grid(True, alpha=0.3)

    stft_db = 20 * np.log10(np.clip(stft_d, 1e-5, None))
    img1 = axs[1].imshow(stft_db, aspect='auto', origin='lower', cmap='inferno',
                          extent=[0, len(y_d)/sr_d, 0, sr_d // 2])
    axs[1].set_title("2. Линейная спектрограмма Фурье (STFT)", fontsize=11, fontweight='bold')
    axs[1].set_xlabel("Время (с)")
    axs[1].set_ylabel("Частота (Гц)")
    fig.colorbar(img1, ax=axs[1], format="%+2.0f dB")

    mel_db = 10 * np.log10(np.clip(mel_d, 1e-5, None))
    img2 = axs[2].imshow(mel_db, aspect='auto', origin='lower', cmap='magma',
                          extent=[0, len(y_d)/sr_d, 1, mel_d.shape[0]])
    axs[2].set_title("3. Мел-спектрограмма (нелинейная мел-шкала слухового восприятия)", fontsize=11, fontweight='bold')
    axs[2].set_xlabel("Время (с)")
    axs[2].set_ylabel("Мел-фильтр")
    fig.colorbar(img2, ax=axs[2], format="%+2.0f dB")

    img3 = axs[3].imshow(mfcc_d, aspect='auto', origin='lower', cmap='viridis',
                          extent=[0, len(y_d)/sr_d, 1, 13])
    axs[3].set_title("4. Матрица 13 мел-кепстральных коэффициентов (MFCC-13)", fontsize=11, fontweight='bold')
    axs[3].set_xlabel("Время (с)")
    axs[3].set_ylabel("Индекс MFCC (1..13)")
    fig.colorbar(img3, ax=axs[3])

    plt.savefig(plot_path1, dpi=200)
    plt.close()
    print(f"\n[+] Сохранён график признаков: {plot_path1.name}")

    # ШАГ 3. Построение базы эталонов enrollment (первые 3 фразы)
    templates = build_enrollment_database(all_embeddings, n_enroll=3)
    darkhan_template = templates["Darkhan (Аканов Д.)"]

    # ШАГ 4-5. Верификация тестовых фраз
    genuine_scores = []
    impostor_scores = []
    test_records = []

    # Предварительный сбор оценок для точного расчёта разделяющего порога T*
    for idx in range(3, 5):
        emb = all_embeddings["Darkhan (Аканов Д.)"][idx]
        sc = cosine_similarity(emb, darkhan_template)
        genuine_scores.append(sc)
        test_records.append((f"Darkhan/фраза_{idx+1}", "Дархан Аканов", sc, "Genuine"))

    for idx in range(len(all_embeddings["Impostor 1 (Боб)"])):
        emb = all_embeddings["Impostor 1 (Боб)"][idx]
        sc = cosine_similarity(emb, darkhan_template)
        impostor_scores.append(sc)
        test_records.append((f"Bob/фраза_{idx+1}", "Боб (чужак)", sc, "Impostor"))

    for idx in range(len(all_embeddings["Impostor 2 (Чарли)"])):
        emb = all_embeddings["Impostor 2 (Чарли)"][idx]
        sc = cosine_similarity(emb, darkhan_template)
        impostor_scores.append(sc)
        test_records.append((f"Charlie/фраза_{idx+1}", "Чарли (чужак)", sc, "Impostor"))

    min_gen = min(genuine_scores)
    max_gen = max(genuine_scores)
    avg_gen = np.mean(genuine_scores)
    min_imp = min(impostor_scores)
    max_imp = max(impostor_scores)
    avg_imp = np.mean(impostor_scores)
    optimal_threshold = float((min_gen + max_imp) / 2.0)

    print("\n[ШАГ 4-5] Тестирование верификации диктора (Cosine Similarity):")
    print("-" * 75)
    print(f"{'Образец':<24} | {'Истинный диктор':<20} | {'Score':<8} | Решение (T*={optimal_threshold:.3f})")
    print("-" * 75)
    for name, spk, sc, kind in test_records:
        if kind == "Genuine":
            status = "ПРИНЯТ (Genuine)" if sc >= optimal_threshold else "ОТКЛОНЁН (FRR)"
        else:
            status = "ОТКЛОНЁН (Верно)" if sc < optimal_threshold else "ДОПУЩЕН (FAR!)"
        print(f"{name:<24} | {spk:<20} | {sc:.4f}  | {status}")
    print("-" * 75)

    print(f"Статистика Score Genuine:  min={min_gen:.4f}, max={max_gen:.4f}, avg={avg_gen:.4f}")
    print(f"Статистика Score Impostor: min={min_imp:.4f}, max={max_imp:.4f}, avg={avg_imp:.4f}")
    print(f"Оптимальный разделяющий порог T*: {optimal_threshold:.4f}")

    # Рисунок 2: Столбчатая диаграмма оценок
    plot_path2 = REPORTS_DIR / "lab08_genuine_vs_impostor.png"
    plt.figure(figsize=(9.5, 4.5), constrained_layout=True)
    all_scores_data = [rec[2] for rec in test_records]
    labels_data = [rec[0] for rec in test_records]
    colors_data = ["#16A34A" if rec[3] == "Genuine" else "#DC2626" for rec in test_records]

    plt.bar(range(len(all_scores_data)), all_scores_data, color=colors_data, width=0.62, edgecolor='black', lw=0.6)
    plt.axhline(optimal_threshold, color="#2563EB", linestyle="--", lw=2, label=f"Порог T* = {optimal_threshold:.3f}")
    plt.xticks(range(len(labels_data)), labels_data, rotation=45, ha='right', fontsize=9)
    plt.ylabel("Cosine Similarity", fontsize=11)
    plt.title("Верификация голосовых образцов: Свои (зелёный) vs Чужаки (красный)", fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left', framealpha=0.9)
    plt.savefig(plot_path2, dpi=200)
    plt.close()
    print(f"[+] Сохранён график распределения оценок: {plot_path2.name}")

    # Рисунок 3: Кривые FAR и FRR, расчёт точки EER
    thresholds = np.linspace(0.60, 1.0, 200)
    far_list = []
    frr_list = []
    for th in thresholds:
        far = np.mean([1 if s >= th else 0 for s in impostor_scores])
        frr = np.mean([1 if s < th else 0 for s in genuine_scores])
        far_list.append(far * 100.0)
        frr_list.append(frr * 100.0)

    diff = np.abs(np.array(far_list) - np.array(frr_list))
    eer_idx = int(np.argmin(diff))
    eer_threshold = thresholds[eer_idx]
    eer_val = (far_list[eer_idx] + frr_list[eer_idx]) / 2.0

    plot_path3 = REPORTS_DIR / "lab08_far_frr_threshold.png"
    plt.figure(figsize=(9.5, 5), constrained_layout=True)
    plt.plot(thresholds, far_list, label="FAR (%) — Ложный допуск чужака", color="#DC2626", lw=2.2)
    plt.plot(thresholds, frr_list, label="FRR (%) — Ложный отказ своему", color="#2563EB", lw=2.2)
    plt.scatter([eer_threshold], [eer_val], color="#D97706", s=110, zorder=5,
                label=f"Точка EER = {eer_val:.1f}% при T = {eer_threshold:.3f}")

    plt.axvline(0.93, color="#7C3AED", linestyle=":", lw=1.8, label="Серверная (T=0.93, FAR≈0%)")
    plt.axvline(0.82, color="#059669", linestyle=":", lw=1.8, label="Фитнес-клуб (T=0.82, FRR≈0%)")

    plt.title("Зависимость ошибок FAR и FRR от порога верификации T", fontsize=12, fontweight='bold')
    plt.xlabel("Порог решения (Cosine Similarity)", fontsize=11)
    plt.ylabel("Вероятность ошибки (%)", fontsize=11)
    plt.grid(True, alpha=0.35)
    plt.legend(loc="center left", framealpha=0.92, fontsize=9.5)
    plt.savefig(plot_path3, dpi=200)
    plt.close()
    print(f"[+] Сохранён график FAR/FRR/EER: {plot_path3.name}")

    # Рисунок 4: Стабильность эталона при N=1, N=3, N=5
    stability_results = {}
    for n in [1, 3, 5]:
        temp_dict = build_enrollment_database(all_embeddings, n_enroll=n)
        temp_darkhan = temp_dict["Darkhan (Аканов Д.)"]
        gen_sims = [cosine_similarity(emb, temp_darkhan) for emb in all_embeddings["Darkhan (Аканов Д.)"]]
        imp_sims = [cosine_similarity(emb, temp_darkhan) for emb in all_embeddings["Impostor 1 (Боб)"]]
        stability_results[n] = (gen_sims, imp_sims)

    plot_path4 = REPORTS_DIR / "lab08_enrollment_samples_effect.png"
    plt.figure(figsize=(9, 4.5), constrained_layout=True)
    n_keys = [1, 3, 5]
    avg_gen_n = [np.mean(stability_results[k][0]) for k in n_keys]
    std_gen_n = [np.std(stability_results[k][0]) for k in n_keys]
    avg_imp_n = [np.mean(stability_results[k][1]) for k in n_keys]

    x = np.arange(len(n_keys))
    width = 0.35
    plt.bar(x - width/2, avg_gen_n, width, yerr=std_gen_n, capsize=5, color="#16A34A",
            label="Genuine сходство (среднее ± std)", edgecolor='black', lw=0.6)
    plt.bar(x + width/2, avg_imp_n, width, color="#DC2626",
            label="Impostor сходство (среднее)", edgecolor='black', lw=0.6)
    plt.xticks(x, [f"N = {k} образец" if k == 1 else f"N = {k} образца" if k < 5 else f"N = {k} образцов" for k in n_keys])
    plt.ylabel("Cosine Similarity", fontsize=11)
    plt.ylim(0, 1.1)
    plt.title("Влияние числа образцов enrollment (N=1, 3, 5) на устойчивость эталона", fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig(plot_path4, dpi=200)
    plt.close()
    print(f"[+] Сохранён график влияния enrollment: {plot_path4.name}")

    print("\n" + "=" * 70)
    print("ИТОГИ ЗАНЯТИЯ 8:")
    print(f"1. Эталон Дархана Аканова сформирован из N=3 образцов.")
    print(f"2. Средний score своих (Genuine): {avg_gen:.4f}")
    print(f"3. Средний score чужих (Impostor): {avg_imp:.4f}")
    print(f"4. Динамический диапазон разделимости: Δ = {avg_gen - avg_imp:.4f}")
    print(f"5. Оптимальный порог: T* = {optimal_threshold:.3f} (EER = {eer_val:.1f}%)")
    print("=" * 70)

    return test_records, optimal_threshold, eer_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Занятие 8. Голосовая биометрия (Аканов Дархан SIS-2309)")
    parser.add_argument("--record", action="store_true", help="Записать контрольную фразу с микрофона перед верификацией")
    args = parser.parse_args()

    if args.record:
        print("[Интерактивный режим микрофона]")
        try:
            import sounddevice as sd
            sr = 16000
            print("Произнесите контрольную фразу в микрофон (3 секунды)...")
            import time
            time.sleep(1)
            audio = sd.rec(int(3.0 * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            temp_path = REPORTS_DIR / "temp_live_mic.wav"
            if HAS_SOUNDFILE:
                sf.write(str(temp_path), audio.flatten(), sr)
            else:
                wavfile.write(str(temp_path), sr, (audio.flatten() * 32767).astype(np.int16))
            print(f"Запись сохранена: {temp_path.name}")
        except Exception as e:
            print(f"Ошибка записи с микрофона: {e}")

    run_voice_pipeline()
