import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------------------------------------------------------
# Фикс несовместимости версий sklearn при загрузке Pipeline из pickle
# (нужен, если модель обучалась в одной версии sklearn, а запускается в другой)
# -----------------------------------------------------------------------------
try:
    import sklearn.compose._column_transformer as _ct  # type: ignore

    if not hasattr(_ct, "_RemainderColsList"):

        class _RemainderColsList(list):
            """Заглушка для старого внутреннего класса sklearn.
            Нужна только для корректной распаковки сохраненной модели."""

            pass

        _ct._RemainderColsList = _RemainderColsList
except Exception:
    # Если импорт не удался, просто продолжаем. загрузка модели может и так сработать.
    pass


# -----------------------------------------------------------------------------
# Общие настройки страницы
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduRisk. Прогноз исхода обучения",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "best_model.pkl"

# Имена признаков. порядок ДОЛЖЕН полностью совпадать с обучением модели.
FEATURE_COLS = [
    "Daytime/evening attendance",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Age at enrollment",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

# Словари для человекочитаемых значений
binary_map = {"Нет": 0, "Да": 1}
gender_map = {"Женский": 0, "Мужской": 1}
attendance_map = {
    "Дневная форма обучения": 1,
    "Вечерняя или другая форма": 0,
}


# -----------------------------------------------------------------------------
# Загрузка модели
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. "
            "Убедитесь, что best_model.pkl лежит рядом с app.py в репозитории."
        )
    model = joblib.load(MODEL_PATH)
    return model


# Пытаемся загрузить модель один раз при старте приложения
try:
    model = load_model()
except Exception as e:
    st.error(
        "Не удалось загрузить модель. "
        "Проверьте наличие файла best_model.pkl и версии зависимостей."
    )
    st.exception(e)
    st.stop()


# -----------------------------------------------------------------------------
# Заголовок и описание
# -----------------------------------------------------------------------------
st.title("EduRisk. Прогноз академического исхода студента")

st.markdown(
    """
EduRisk использует предобученную модель машинного обучения для оценки **вероятностей трех исходов обучения**:

* `Dropout` — отчисление  
* `Enrolled` — продолжает обучение  
* `Graduate` — успешно завершит обучение  

Модель работает только на понятных признаках:
форма обучения, пол, возраст, стипендия, задолженности,
статус иностранного студента и успеваемость за 1 и 2 семестры.
"""
)


# -----------------------------------------------------------------------------
# Форма ввода признаков
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Параметры студента")

    inputs: dict[str, float | int] = {}

    # Блок: общая информация
    st.subheader("Общая информация")

    attend_ru = st.selectbox(
        "Форма обучения",
        list(attendance_map.keys()),
        help=(
            "Дневная форма обучения — стандартная очная программа."
            " Вечерняя или другая форма — любые альтернативные форматы."
        ),
    )
    inputs["Daytime/evening attendance"] = attendance_map[attend_ru]

    gender_ru = st.selectbox("Пол", list(gender_map.keys()))
    inputs["Gender"] = gender_map[gender_ru]

    inputs["Age at enrollment"] = st.number_input(
        "Возраст на момент поступления",
        value=20,
        min_value=16,
        max_value=70,
        step=1,
        help="Чаще всего 17–30 лет, но возможны нетипичные случаи (позднее поступление).",
    )

    # Блок: социально экономические факторы
    st.subheader("Социально экономические факторы")

    debtor_ru = st.selectbox(
        "Есть задолженность по оплате обучения",
        list(binary_map.keys()),
        help="Да — у студента есть непогашенная задолженность по оплате.",
    )
    inputs["Debtor"] = binary_map[debtor_ru]

    fees_ru = st.selectbox(
        "Оплата обучения внесена вовремя",
        list(binary_map.keys()),
        help="Да — все платежи за обучение внесены в срок.",
    )
    inputs["Tuition fees up to date"] = binary_map[fees_ru]

    scholar_ru = st.selectbox(
        "Получает стипендию",
        list(binary_map.keys()),
        help="Да — студент получает академическую или социальную стипендию.",
    )
    inputs["Scholarship holder"] = binary_map[scholar_ru]

    intl_ru = st.selectbox(
        "Иностранный студент",
        list(binary_map.keys()),
        help="Да — у студента иностранное гражданство по отношению к вузу.",
    )
    inputs["International"] = binary_map[intl_ru]

    # Блок: успеваемость 1 семестра
    st.subheader("Успеваемость. 1 семестр")

    inputs["Curricular units 1st sem (enrolled)"] = st.number_input(
        "Количество записанных дисциплин в 1 семестре",
        value=6,
        min_value=3,
        max_value=10,
        step=1,
        help="Типичный диапазон нагрузки — 5–6 дисциплин, максимум 8–10.",
    )

    inputs["Curricular units 1st sem (approved)"] = st.number_input(
        "Количество успешно сданных дисциплин в 1 семестре",
        value=5,
        min_value=0,
        max_value=inputs["Curricular units 1st sem (enrolled)"],
        step=1,
        help="Не может превышать число записанных дисциплин.",
    )

    inputs["Curricular units 1st sem (grade)"] = st.number_input(
        "Средний балл за 1 семестр (шкала 0–20)",
        value=12.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        help="Средний итоговый балл. 10–14 — условный «средний» уровень.",
    )

    # Блок: успеваемость 2 семестра
    st.subheader("Успеваемость. 2 семестр")

    inputs["Curricular units 2nd sem (enrolled)"] = st.number_input(
        "Количество записанных дисциплин во 2 семестре",
        value=6,
        min_value=3,
        max_value=10,
        step=1,
        help="Обычно 5–6 дисциплин, при перегрузке — до 8–10.",
    )

    inputs["Curricular units 2nd sem (approved)"] = st.number_input(
        "Количество успешно сданных дисциплин во 2 семестре",
        value=6,
        min_value=0,
        max_value=inputs["Curricular units 2nd sem (enrolled)"],
        step=1,
        help="Не может быть больше числа записанных дисциплин.",
    )

    inputs["Curricular units 2nd sem (grade)"] = st.number_input(
        "Средний балл за 2 семестр (шкала 0–20)",
        value=12.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        help="Средний итоговый балл по дисциплинам второго семестра.",
    )

    predict_btn = st.button("Сделать прогноз")


# -----------------------------------------------------------------------------
# Логика прогноза
# -----------------------------------------------------------------------------
if predict_btn:
    # Мягкая проверка консистентности
    if (
        inputs["Curricular units 1st sem (approved)"]
        > inputs["Curricular units 1st sem (enrolled)"]
    ):
        st.warning("1 семестр. сдано не может быть больше, чем записано.")
    if (
        inputs["Curricular units 2nd sem (approved)"]
        > inputs["Curricular units 2nd sem (enrolled)"]
    ):
        st.warning("2 семестр. сдано не может быть больше, чем записано.")

    # Собираем признаки в нужном порядке
    row = {col: inputs[col] for col in FEATURE_COLS}
    X_input = pd.DataFrame([row])

    st.subheader("Введенные данные")
    st.dataframe(X_input)

    try:
        proba = model.predict_proba(X_input)[0]
        classes = model.classes_
        pred_idx = int(np.argmax(proba))
        pred_class = classes[pred_idx]

        st.subheader("Результат прогноза")

        proba_df = pd.DataFrame(
            {"Исход": classes, "Вероятность": np.round(proba, 3)}
        )
        st.dataframe(proba_df, use_container_width=True)

        st.success(f"Наиболее вероятный исход. **{pred_class}**")

        if pred_class == "Dropout":
            st.warning(
                "Модель оценивает высокий риск отчисления. "
                "Рекомендуется проанализировать успеваемость, академическую нагрузку "
                "и финансовую дисциплину студента. Возможны меры поддержки "
                "и снижение нагрузки."
            )
        elif pred_class == "Graduate":
            st.info(
                "Модель прогнозирует успешное завершение обучения. "
                "Важно поддерживать текущий уровень мотивации и успеваемости."
            )
        else:
            st.info(
                "Модель прогнозирует продолжение обучения. "
                "Рекомендуется следить за динамикой оценок и вовлеченности студента, "
                "чтобы не допустить перехода в группу риска."
            )

    except Exception as e:
        st.error("Произошла ошибка при расчете прогноза.")
        st.exception(e)
else:
    st.info(
        "Заполните параметры студента в левой панели и нажмите кнопку "
        "«Сделать прогноз»."
    )
