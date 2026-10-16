"""
Скрипт создания подробного документа Microsoft Word (.docx) с руководством
и готовыми текстами для защиты лабораторных работ по биометрии (Лаб 1, 3, 4, 5, 6).
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_DOCX = REPORTS_DIR / "Biometrics_Defense_Guide.docx"


def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_speech_box(doc, text: str, title: str = "Готовая речь для защиты (что говорить преподавателю):"):
    """Создаёт стильный визуальный блок для заучивания/чтения речи."""
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.paragraph_format.left_indent = Inches(0.2)
    r_t = p_title.add_run(f"🎤 {title}")
    r_t.bold = True
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(11.5)
    r_t.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.right_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.15

    # Стиль рамки с синей вертикальной чертой слева
    pPr = p._p.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F0FDF4"/>')
    pPr.append(shd)
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="4" w:color="BBF7D0"/>'
        f'<w:left w:val="single" w:sz="24" w:space="8" w:color="16A34A"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="4" w:color="BBF7D0"/>'
        f'<w:right w:val="single" w:sz="4" w:space="4" w:color="BBF7D0"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)

    run = p.add_run(f'«{text.strip()}»')
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11.5)
    run.font.color.rgb = RGBColor(0x06, 0x4E, 0x3B)


def add_qa_item(doc, question: str, answer: str):
    """Добавляет вопрос-ответ преподавателя."""
    p_q = doc.add_paragraph()
    p_q.paragraph_format.space_before = Pt(6)
    p_q.paragraph_format.space_after = Pt(2)
    p_q.paragraph_format.left_indent = Inches(0.15)
    r_q = p_q.add_run(f"❓ {question}")
    r_q.bold = True
    r_q.font.name = "Times New Roman"
    r_q.font.size = Pt(11.5)
    r_q.font.color.rgb = RGBColor(0x99, 0x1B, 0x1B)

    p_a = doc.add_paragraph()
    p_a.paragraph_format.space_before = Pt(1)
    p_a.paragraph_format.space_after = Pt(6)
    p_a.paragraph_format.left_indent = Inches(0.35)
    r_a = p_a.add_run(f"💡 Ответ: {answer}")
    r_a.font.name = "Times New Roman"
    r_a.font.size = Pt(11)
    r_a.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)


def add_bullet_point(doc, bold_prefix: str, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25)
    r_b = p.add_run(f"• {bold_prefix} ")
    r_b.bold = True
    r_b.font.name = "Times New Roman"
    r_b.font.size = Pt(11.5)
    r_t = p.add_run(text)
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(11.5)


def add_section_header(doc, title: str, subtitle: str = ""):
    h = doc.add_heading(level=1)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(2)
    r = h.add_run(title)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    if subtitle:
        p_sub = doc.add_paragraph()
        p_sub.paragraph_format.space_before = Pt(1)
        p_sub.paragraph_format.space_after = Pt(6)
        r_sub = p_sub.add_run(subtitle)
        r_sub.italic = True
        r_sub.font.name = "Times New Roman"
        r_sub.font.size = Pt(11)
        r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)


def add_sub_header(doc, title: str):
    h = doc.add_heading(level=2)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(3)
    r = h.add_run(title)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)


def add_picture_if_exists(doc, rel_path: str, caption: str, width_in=6.0):
    p = REPORTS_DIR / rel_path
    if not p.exists():
        p = PROJECT_ROOT / rel_path
    if p.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        r = p_img.add_run()
        r.add_picture(str(p), width=Inches(width_in))

        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c.paragraph_format.space_before = Pt(1)
        p_c.paragraph_format.space_after = Pt(6)
        rc = p_c.add_run(f"Иллюстрация: {caption}")
        rc.italic = True
        rc.font.name = "Times New Roman"
        rc.font.size = Pt(10)
        rc.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)


def build_guide():
    doc = Document()

    # Поля страницы
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(0.7)

    # Титульный блок
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    r_top = title_p.add_run("РУКОВОДСТВО И ШПАРГАЛКА ДЛЯ ЗАЩИТЫ ЛАБОРАТОРНЫХ РАБОТ")
    r_top.bold = True
    r_top.font.name = "Times New Roman"
    r_top.font.size = Pt(18)
    r_top.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_before = Pt(2)
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Дисциплина: Биометрические системы и технологии | Лабораторные работы №1, 3, 4, 5, 6\nРепозиторий проекта: github.com/Dark-Ak06/Biometric")
    r_sub.italic = True
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Вводный блок с советами
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.space_before = Pt(2)
    p_intro.paragraph_format.space_after = Pt(8)
    r_in = p_intro.add_run(
        "Настоящее руководство предназначено для подготовки к успешной защите лабораторных работ перед преподавателем. "
        "Для каждого занятия приведён разбор внутренней инженерной логики (Что, Зачем, Почему), готовый выверенный текст ответа от первого лица "
        "с точными экспериментальными цифрами и формулами, а также ответы на каверзные вопросы."
    )
    r_in.font.name = "Times New Roman"
    r_in.font.size = Pt(11.5)

    # =========================================================================
    # ЛАБОРАТОРНАЯ 1
    # =========================================================================
    add_section_header(doc, "Лабораторная работа №1", "Введение в биометрические системы. Подготовка среды и анализ модальностей")

    add_sub_header(doc, "1. Суть работы (Что, Зачем, Почему)")
    add_bullet_point(doc, "Что сделано:", "Создано изолированное виртуальное окружение biometrics_env (Python 3.10) со специализированным стеком: OpenCV 5.0, NumPy, Matplotlib, scikit-learn, scikit-image. Написан скрипт src/test_environment.py, проведён захват тестового кадра с веб-камеры (640x480 пикселей). Составлена сравнительная таблица биометрических модальностей.")
    add_bullet_point(doc, "Зачем и почему:", "Биометрические алгоритмы работают с потоковыми изображениями и матричными операциями. Изоляция зависимостей гарантирует повторяемость вычислений. Проверка сенсора подтверждает готовность контура захвата (Capture). Сравнительный анализ показывает баланс между ценой сенсора, удобством и уязвимостью к спуфингу.")

    add_speech_box(
        doc,
        "Здравствуйте! В рамках первой лабораторной работы я подготовил инфраструктуру для биометрических исследований и провёл анализ модальностей.\n"
        "В практической части я развернул виртуальное окружение на базе Python 3.10 и библиотек OpenCV, Scikit-Image, Scikit-Learn и NumPy. "
        "Написал скрипт инициализации сенсора cv2.VideoCapture. Сенсор функционирует корректно, получен тестовый снимок разрешением 640 на 480 пикселей.\n"
        "В теоретической части я исследовал структуру биометрического тракта: Enrollment (регистрация), Capture (захват), Feature Extraction (извлечение признаков), "
        "Matching (сопоставление) и Decision (принятие решения).\n"
        "Также я сопоставил режимы 1:1 (Верификация) и 1:N (Идентификация). Верификация требует предъявления ID и выполняется за константное время O(1), "
        "а идентификация ищет по всей базе за линейное время O(N), при этом ошибка ложного допуска возрастает пропорционально размеру базы: FAR(1:N) ≈ N · FAR(1:1).\n"
        "При сравнении модальностей наиболее защищённой оказалась радужка глаза: её рисунок формируется случайно, имеет свыше 240 степеней свободы, "
        "стабилен на протяжении жизни и защищён роговицей, а активная подсветка в ближнем инфракрасном диапазоне с контролем реакции зрачка отсекает муляжи и фотографии."
    )

    add_sub_header(doc, "Козырные вопросы преподавателя по Лаб 1:")
    add_qa_item(doc, "Чем принципиально отличается 1:1 от 1:N?", "В 1:1 (верификация) пользователь заявляет личность (ID), и мы проверяем одно соответствие за O(1). В 1:N (идентификация) ID нет, система перебирает N шаблонов за O(N), и вероятность случайного ложного допуска накапливается как 1 - (1 - FAR)^N.")
    add_qa_item(doc, "Почему радужка надёжнее лица?", "Лицо меняется с возрастом, мимикой, бородой и легко атакуется фото/видео. Радужка глаза неизменна после 1–2 лет, защищена прозрачной роговицей, имеет высочайшую энтропию текстуры и даёт естественный Liveness по микроколебаниям зрачка.")

    # =========================================================================
    # ЛАБОРАТОРНАЯ 3
    # =========================================================================
    doc.add_page_break()
    add_section_header(doc, "Лабораторная работа №3", "Проектирование биометрической системы контроля доступа (СКУД)")

    add_sub_header(doc, "1. Суть работы (Что, Зачем, Почему)")
    add_bullet_point(doc, "Что сделано:", "Спроектирована двухфакторная СКУД в режимную лабораторию на 150 сотрудников (режим верификации 1:1, смарт-карта Mifare DESFire + Лицо RGB/NIR с Liveness Detection). Построена схема контуров в draw.io и программная блок-схема. Выполнено моделирование угроз по 8 точкам атак Ратха. Написан скрипт src/lab03_access_control.py, смоделировавший распределения сходства Score S для Genuine (μ=0.74) и Impostor (μ=0.36), рассчитавший кривые FAR(T), FRR(T), точку EER (3.52% при T=0.56) и обосновавший выбор порога T=0.68.")
    add_bullet_point(doc, "Зачем и почему:", "Биометрический шаблон нельзя хранить в открытом виде (фотографию нельзя сменить при компрометации как пароль). На режимном объекте нельзя работать в точке EER, так как 3.5% чужаков могут проникнуть внутрь. Порог безопасности сдвигается в FAR-зону (T=0.68, FAR ≤ 0.18%), а возникающий дискомфорт ложного отказа легитимным сотрудникам компенсируется правилом трёх повторных попыток.")

    add_picture_if_exists(doc, "lab03_architecture_diagram.png", "Архитектурная схема двухконтурной СКУД с точками атак", width_in=5.8)
    add_picture_if_exists(doc, "lab03_far_frr_decision_graph.png", "Моделирование Decision Module: распределение оценок сходства S и кривые FAR / FRR / EER", width_in=5.8)

    add_speech_box(
        doc,
        "В третьей лабораторной работе я спроектировал архитектуру биометрической СКУД для научно-исследовательской лаборатории повышенной секретности.\n"
        "Я выбрал режим двухфакторной верификации 1:1: бесконтактная карта передаёт Subject_ID, а камера сканирует лицо в видимом и ближнем инфракрасном диапазоне. "
        "Оба контура — регистрация (Enrollment) и верификация (Verification) — используют единый нейросетевой экстрактор ArcFace, благодаря чему векторы находятся "
        "в одном метрическом пространстве R^512. Шаблоны шифруются алгоритмом AES-256 с ключами в аппаратном модуле HSM с применением технологии Cancelable Biometrics.\n"
        "По модели угроз Ратха самым уязвимым элементом является сенсор, поскольку он вынесен в неконтролируемую зону. Для защиты от спуфинга я предусмотрел "
        "двухспектральную съёмку RGB+NIR на длине волны 850 нм и модуль Liveness Detection.\n"
        "Для настройки Decision-модуля я написал скрипт, смоделировавший распределения Genuine и Impostor и построивший кривые FAR и FRR. "
        "Рассчитанная точка EER составила 3.52% при пороге 0.56. Однако для режимного объекта такой допуск посторонних недопустим, "
        "поэтому я выбрал FAR-ориентированный режим с порогом T = 0.68. Вероятность ложного допуска злоумышленника снижена до 0.18%, "
        "а сопутствующий рост FRR до 27% нивелируется каскадной политикой повторных попыток — до трёх быстрых кадров перед вызовом тревоги."
    )

    add_sub_header(doc, "Козырные вопросы преподавателя по Лаб 3:")
    add_qa_item(doc, "Почему нельзя хранить сырое фото лица в БД?", "1) Приватность и необратимость: лицо человека нельзя сменить. Украденное фото скомпрометирует все биометрические аккаунты владельца. Отзываемый шаблон (Cancelable Biometrics) при утечке просто генерируется заново с новой солью/ключом. 2) Производительность: сырое фото весит 5 МБ, а вектор признаков — 2 КБ, и косинусное сходство двух векторов считается за доли миллисекунды.")
    add_qa_item(doc, "Чем архитектурно отличается FAR-система от FRR-системы в Decision-модуле?", "В FAR-системе порог T завышен (T >> T_EER, FAR -> 0), используется конъюнктивное правило слияния (AND — совпадение всех факторов) и каскадная логика повторов. В FRR-системе (смартфон, турникет метро) порог занижен (T < T_EER), чтобы не раздражать пользователя отказом.")

    # =========================================================================
    # ЛАБОРАТОРНАЯ 4
    # =========================================================================
    doc.add_page_break()
    add_section_header(doc, "Лабораторная работа №4", "Работа с биометрическими датасетами и Subject-Disjoint сплит")

    add_sub_header(doc, "1. Суть работы (Что, Зачем, Почему)")
    add_bullet_point(doc, "Что сделано:", "Загружены два датасета: ORL Faces (400 изображений 92x112, 40 человек, строгий баланс по 10 снимков) и Fingerprints FVC (255 изображений 128x128, 25 человек, дисбаланс от 6 до 14 снимков, повороты и шум). Скриптом src/load_dataset.py построены гистограммы классов и сетки образцов. Реализовано разбиение 80/20 по множеству субъектов (split_by_subjects).")
    add_bullet_point(doc, "Зачем и почему:", "Главная ошибка в биометрии — случайное перемешивание снимков между train и test. Это вызывает Subject-Level Data Leakage: модель запоминает конкретных 40 человек (их фон, одежду, очки) вместо обучения универсальной метрики различия лиц. В продакшене система должна верифицировать новых, ранее неизвестных людей (Open-Set), поэтому субъекты из Test ни в коем случае не должны пересекаться с Train.")

    add_picture_if_exists(doc, "dataset_class_distribution.png", "Распределение снимков по классам: идеальный баланс ORL Faces vs естественный разброс FVC", width_in=5.8)
    add_picture_if_exists(doc, "subject_split_comparison.png", "Корректный Subject-Disjoint сплит против некорректного случайного сплита изображений", width_in=5.8)

    add_speech_box(
        doc,
        "В четвёртой лабораторной работе я исследовал особенности подготовки данных для обучения и тестирования биометрических систем на примере датасетов ORL Faces и Fingerprints FVC.\n"
        "Я выполнил статистический анализ: датасет ORL Faces идеально сбалансирован — ровно по 10 снимков на каждого из 40 субъектов. "
        "В датасете отпечатков FVC наблюдается реалистичный разброс от 6 до 14 снимков на класс, а также контактные искажения: "
        "повороты пальца до 12.5 градусов, сдвиги центра узора и неравномерный прижим кожи к сенсору.\n"
        "Центральная задача работы — правильное разделение данных на обучающую и тестовую выборки. "
        "Классический случайный train_test_split по изображениям в биометрии категорически недопустим, так как приводит к утечке данных на уровне субъектов (Data Leakage). "
        "Модель просто выучит лица сорока человек и покажет 99% точности на тесте, но полностью провалится при появлении сорок первого, неизвестного человека.\n"
        "Я реализовал функцию split_by_subjects, которая делит выборку по множеству уникальных субъектов в пропорции 80 на 20. "
        "Для ORL 32 человека (320 изображений) вошли в Train, и 8 совершенно новых людей (80 изображений) — в Test с нулевым пересечением идентификаторов. "
        "Это гарантирует объективную оценку системы в реалистичном режиме открытого множества (Open-Set)."
    )

    add_sub_header(doc, "Козырные вопросы преподавателя по Лаб 4:")
    add_qa_item(doc, "Что такое Closed-Set и Open-Set в биометрии?", "Closed-Set — классификация среди заранее известного списка лиц (все люди из теста были в обучении). Open-Set — реальная верификация, где система проверяет произвольную пару лиц ранее неизвестных людей (Subject(Train) ∩ Subject(Test) = ∅).")
    add_qa_item(doc, "Как влияет дисбаланс классов на биометрическую модель?", "При сильном дисбалансе классификатор сдвигает границы в пользу классов с большим числом образцов, а для малопредставленных классов возрастает ошибка ложного отказа (FRR).")

    # =========================================================================
    # ЛАБОРАТОРНАЯ 5
    # =========================================================================
    doc.add_page_break()
    add_section_header(doc, "Лабораторная работа №5", "Распознавание отпечатков пальцев. Скелетизация и сопоставление")

    add_sub_header(doc, "1. Суть работы (Что, Зачем, Почему)")
    add_bullet_point(doc, "Что сделано:", "В src/fingerprint_matching.py реализован тракт предобработки: фильтр Гаусса 5x5, локальное контрастирование CLAHE, адаптивная инвертированная бинаризация Оцу, скелетизация папиллярных линий до 1 пикселя (skimage.morphology.skeletonize). Извлечены минуции Рутовица (CN=1 окончания, CN=3 ветвления) и 256-битные дескрипторы ORB. Проведено сопоставление 5 пар Genuine и 5 пар Impostor. Найден оптимальный порог T* = 0.292 (100% точность). Проведён эксперимент с поворотом отпечатка от 0° до 180°.")
    add_bullet_point(doc, "Зачем и почему:", "Толщина линий отпечатка нестабильна и зависит от силы нажатия и влажности кожи (от 2 до 8 пикселей). Скелетизация приводит все гребни к канонической толщине ровно в 1 пиксель. Дескриптор ORB инвариантен к повороту, так как вычисляет угол ориентации ключевой точки через центроид интенсивности патча, поэтому даже при повороте на 90° Score остаётся выше порога допуска.")

    add_picture_if_exists(doc, "lab05_preprocessing_pipeline.png", "Этапы предобработки отпечатка: оригинал -> CLAHE -> бинаризация -> скелет 1px -> минуции", width_in=5.8)
    add_picture_if_exists(doc, "lab05_genuine_vs_impostor_threshold.png", "Разделение оценок сходства Genuine и Impostor оптимальным порогом T* = 0.292", width_in=5.8)
    add_picture_if_exists(doc, "lab05_rotation_experiment.png", "Эксперимент с поворотом: зависимость Score от угла поворота отпечатка", width_in=5.8)

    add_speech_box(
        doc,
        "В пятой лабораторной работе я реализовал алгоритмы предобработки, скелетизации и сопоставления отпечатков пальцев.\n"
        "Разработанный пайплайн включает устранение шума сенсора фильтром Гаусса, выравнивание контраста CLAHE, бинаризацию по методу Оцу и морфологическую скелетизацию. "
        "Скелетизация является обязательным этапом: она редуцирует гребни до толщины в один пиксель, исключая влияние силы нажима пальца, и обеспечивает точное вычисление "
        "числа переходов Рутовица (Crossing Number) для поиска окончаний и бифуркаций.\n"
        "Для сопоставления я применил дескрипторы ключевых точек ORB и алгоритм BFMatcher с расстоянием Хэмминга. "
        "В эксперименте на 5 парах Genuine («свой-свой») и 5 парах Impostor («свой-чужой») средний Score легитимных пар составил 0.380 (диапазон от 0.326 до 0.453), "
        "а для самозванцев — 0.184 (диапазон от 0.137 до 0.258). Середина разделяющего зазора дала оптимальный порог T* = 0.292, обеспечивший 100% точность верификации.\n"
        "Кроме того, я исследовал устойчивость метода к вращению отпечатка от 0 до 180 градусов. "
        "Поскольку алгоритм ORB определяет локальную ориентацию особых точек методом центроида интенсивности, при повороте ровно на 90 градусов Score составил 0.390, "
        "что уверенно превышает порог 0.292. Это доказывает ротационную инвариантность алгоритма к поворотам пальца на сканере."
    )

    add_sub_header(doc, "Козырные вопросы преподавателя по Лаб 5:")
    add_qa_item(doc, "Что такое число переходов Рутовица (Crossing Number)?", "CN = 0.5 · Σ |P_i - P_{i+1}| при обходе 8-окрестности пикселя на однопиксельном скелете. Если CN=1 — это окончание папиллярной линии; если CN=3 — это бифуркация (разветвление).")
    add_qa_item(doc, "Почему при повороте на 30° Score падает сильнее, чем при 90°?", "При 90° растровая сетка пикселей поворачивается ортогонально без искажений. При 30° работает билинейная интерполяция, которая размывает наклонные гребни, порождая микросдвиги скелета и ключевых точек.")

    # =========================================================================
    # ЛАБОРАТОРНАЯ 6
    # =========================================================================
    doc.add_page_break()
    add_section_header(doc, "Лабораторная работа №6", "Детекция лиц: каскады Хаара против глубоких нейросетей DNN SSD")

    add_sub_header(doc, "1. Суть работы (Что, Зачем, Почему)")
    add_bullet_point(doc, "Что сделано:", "В src/face_detection.py реализованы два метода детекции лиц: классический каскад Виолы-Джонса (Haar Cascade) и глубокая свёрточная нейросеть DNN SSD (Single Shot MultiBox Detector на базе ResNet-10 с адаптером YuNet). Реализована детекция с веб-камеры с расчётом FPS. Проведён бенчмарк на 10 сценах с 14 истинными лицами (Ground Truth) по метрикам IoU (порог ≥ 0.35), TP, FP, FN и скорости инференса. Исследовано влияние гиперпараметров Хаара (scaleFactor и minNeighbors).")
    add_bullet_point(doc, "Зачем и почему:", "Каскад Хаара быстрее (124 FPS) за счёт интегрального изображения, но слеп к поворотам головы больше 15° и даёт ложные рамки на текстурированном фоне (жалюзи, решётки). Нейросеть DNN SSD на CPU работает чуть медленнее (115 FPS), но оперирует семантическими признаками лица, детектирует при наклонах до 45°, в очках и слабом свете, показав нулевой уровень ложных тревог (FP=0) и Mean IoU = 0.744 против 0.625 у Хаара.")

    add_picture_if_exists(doc, "lab06_metrics_and_fps_chart.png", "Сравнение метрик TP / FP / FN, среднего IoU и скорости обработки (FPS)", width_in=5.8)
    add_picture_if_exists(doc, "lab06_10_images_comparison.png", "Визуализация работы детекторов на 10 тестовых сценах", width_in=5.8)
    add_picture_if_exists(doc, "lab06_haar_params_experiment.png", "Влияние scaleFactor и minNeighbors на точность и быстродействие каскада Хаара", width_in=5.8)

    add_speech_box(
        doc,
        "В шестой лабораторной работе я исследовал и сопоставил два фундаментальных подхода к детекции лиц: классический каскадный метод Виолы-Джонса на признаках Хаара "
        "и нейросетевой детектор Single Shot MultiBox Detector (DNN SSD).\n"
        "Я протестировал оба алгоритма на видеопотоке веб-камеры в реальном времени, а также провёл объективный бенчмарк на выборке из 10 сцен с 14 размеченными лицами, "
        "включающими слабый свет, групповые фото, наклоны головы, очки и частичные перекрытия.\n"
        "Результаты бенчмарка показали явные преимущества нейросетевого детектора:\n"
        "1. По качеству локализации DNN SSD показал средний IoU 0.744 против 0.625 у Хаара.\n"
        "2. По ошибкам каскад Хаара допустил 2 ложных срабатывания на сложном фоне и 4 пропуска (TP=10, FP=2, FN=4). "
        "Нейросеть DNN SSD не допустила ни одной ложной тревоги и обнаружила 12 лиц из 14 (TP=12, FP=0, FN=2), уверенно справившись с поворотом головы в 22 градуса.\n"
        "3. По быстродействию на CPU каскад Хаара показал 124 кадра в секунду против 115 у DNN благодаря O(1) вычислению признаков через интегральное изображение.\n"
        "В исследовании гиперпараметров Хаара я доказал, что оптимальным компромиссом являются scaleFactor=1.1 и minNeighbors=5. "
        "Занижение minNeighbors порождает массу ложных срабатываний на текстурах, а завышение приводит к потере реальных лиц."
    )

    add_sub_header(doc, "Козырные вопросы преподавателя по Лаб 6:")
    add_qa_item(doc, "Что такое IoU (Intersection over Union)?", "IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt). Мера точности попадания предсказанной рамки в истинную. Если IoU ≥ 0.35, детекция считается истинно положительной (True Positive).")
    add_qa_item(doc, "Почему каскад Хаара ошибается на жалюзи и кирпичной кладке?", "Примитивы Хаара анализируют только разность яркости в смежных прямоугольниках на ч/б картинке. Чередование тёмных и светлых полос жалюзи математически совпадает с паттерном «глаза-переносица». Нейросеть DNN SSD обучается на цветных трёхмерных признаках лица и контекста, поэтому не путает фон с лицом.")

    # =========================================================================
    # СВОДНАЯ ТАБЛИЦА КОЗЫРНЫХ ОТВЕТОВ
    # =========================================================================
    doc.add_page_break()
    add_section_header(doc, "Экспресс-шпаргалка", "Сводная таблица моментальных ответов на контрольные вопросы")

    table_data = [
        ("Вопрос преподавателя", "Моментальный эталонный ответ студента"),
        (
            "Что такое EER и почему мы не используем его в спецлаборатории?",
            "EER — точка равенства FAR и FRR (у нас 3.52% при T=0.56). Для режимного объекта допуск 3.5% чужаков недопустим, поэтому порог сдвигают в FAR-зону (T=0.68, FAR ≤ 0.18%)."
        ),
        (
            "В чём фундаментальная опасность случайного сплита в биометрии?",
            "Возникает Subject-Level Data Leakage. Модель запоминает знакомые лица вместо обучения универсального метрического пространства, из-за чего на новых людях (Open-Set) система отказывает."
        ),
        (
            "Зачем нужна скелетизация отпечатка пальца?",
            "Она приводит гребни к канонической толщине ровно в 1 пиксель, устраняя влияние влажности и силы нажима пальца, и делает возможным корректный расчёт Crossing Number (CN)."
        ),
        (
            "Почему при повороте отпечатка на 90° он распознаётся?",
            "Дескриптор ORB инвариантен к повороту, вычисляя угол ориентации ключевой точки через центроид интенсивности патча, поэтому Score (0.390) остаётся выше порога T*=0.292."
        ),
        (
            "Почему каскад Хаара быстрее, но даёт больше False Positives?",
            "Он вычисляет признаки за O(1) через интегральное изображение (124 FPS), но оперирует простейшими ч/б прямоугольниками, реагируя на любые полосатые текстуры фона."
        ),
        (
            "Где хранится ваш код и отчёты?",
            "Репозиторий проекта опубликован на GitHub: github.com/Dark-Ak06/Biometric, полные отчёты со всеми графиками оформлены в Word docx и лежат в папке reports/."
        ),
    ]

    table = doc.add_table(rows=len(table_data), cols=2)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, (q_txt, a_txt) in enumerate(table_data):
        row = table.rows[r_idx]
        cell_q = row.cells[0]
        cell_a = row.cells[1]
        cell_q.width = Inches(2.2)
        cell_a.width = Inches(4.5)
        set_cell_margins(cell_q, top=70, bottom=70, left=100, right=100)
        set_cell_margins(cell_a, top=70, bottom=70, left=100, right=100)

        p_q = cell_q.paragraphs[0]
        p_a = cell_a.paragraphs[0]
        p_q.paragraph_format.space_before = Pt(2)
        p_q.paragraph_format.space_after = Pt(2)
        p_a.paragraph_format.space_before = Pt(2)
        p_a.paragraph_format.space_after = Pt(2)

        if r_idx == 0:
            set_cell_background(cell_q, "0F172A")
            set_cell_background(cell_a, "0F172A")
            rq = p_q.add_run(q_txt)
            ra = p_a.add_run(a_txt)
            rq.bold = True
            ra.bold = True
            rq.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            ra.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        else:
            if r_idx % 2 == 0:
                set_cell_background(cell_q, "F8FAFC")
                set_cell_background(cell_a, "F8FAFC")
            rq = p_q.add_run(q_txt)
            rq.bold = True
            rq.font.size = Pt(10)
            ra = p_a.add_run(a_txt)
            ra.font.size = Pt(10)

    doc.save(str(TARGET_DOCX))
    print(f"[OK] Сформирован документ для защиты: {TARGET_DOCX.name} ({TARGET_DOCX.stat().st_size / 1024.0:.1f} КБ)")


if __name__ == "__main__":
    build_guide()
