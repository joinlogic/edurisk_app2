import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Фикс несовместимости версий sklearn при загрузке Pipeline из pickle
import sklearn.compose._column_transformer as _ct


if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Заглушка для старого внутреннего класса sklearn.
        Нужна только для корректной распаковки сохраненной модели."""
        pass

    _ct._RemainderColsList = _RemainderColsList


st.set_page_config(page_title="EduRisk", layout="wide")

MODEL_PATH = Path(__file__).parent / "best_model.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


model = load_model()

# Имена признаков, должны совпадать с обучением компактной модели
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

binary_map = {"Нет": 0, "Да": 1}
gender_map = {"Женский": 0, "Мужской": 1}
attendance_map = {
    "Дневная форма обучения": 1,
    "Вечерняя или другая форма": 0,
}

st.title("EduRisk. Прогноз академического исхода студента")

st.markdown(
    """
EduRisk использует предобученную модель машинного обучения.
Цель. оценить вероятности трех исходов обучения.

* `Dropout`. отчисление  
* `Enrolled`. продолжает обучение  
* `Graduate`. успешно завершит обучение  

Модель работает только на понятных признаках. возраст. форма обучения. стипендия.
задолженности. статус иностранного студента. успеваемость за 1 и 2 семестры.
"""
)

with st.sidebar:
    st.header("Параметры студента")

    inputs = {}

    # Общая информация
    st.subheader("Общая информация")

    attend_ru = st.selectbox(
        "Форма обучения",
        list(attendance_map.keys()),
        help=(
            "Дневная. если студент учится по стандартной дневной программе."
            " Вечерняя или другая форма. если обучение в альтернативном формате."
        ),
    )
    inputs["Daytime/evening attendance"] = attendance_map[attend_ru]

    gender_ru = st.selectbox(
        "Пол",
        list(gender_map.keys()),
    )
    inputs["Gender"] = gender_map[gender_ru]

    inputs["Age at enrollment"] = st.number_input(
        "Возраст на момент поступления",
        value=20,
        min_value=16,
        max_value=70,
        step=1,
        help="Обычно 17–30 лет, возможны нетипичные случаи, например возврат к учебе.",
    )

    # Социально экономические факторы
    st.subheader("Социально экономические факторы")

    debtor_ru = st.selectbox(
        "Есть задолженность по оплате обучения",
        list(binary_map.keys()),
        help="Да. если у студента есть непогашенная задолженность по оплате.",
    )
    inputs["Debtor"] = binary_map[debtor_ru]

    fees_ru = st.selectbox(
        "Оплата обучения внесена вовремя",
        list(binary_map.keys()),
        help="Да. если все платежи за обучение внесены в срок.",
    )
    inputs["Tuition fees up to date"] = binary_map[fees_ru]

    scholar_ru = st.selectbox(
        "Получает стипендию",
        list(binary_map.keys()),
        help="Да. если студент получает академическую или социальную стипендию.",
    )
    inputs["Scholarship holder"] = binary_map[scholar_ru]

    intl_ru = st.selectbox(
        "Иностранный студент",
        list(binary_map.keys()),
        help="Да. если студент имеет иностранное гражданство по отношению к вузу.",
    )
    inputs["International"] = binary_map[intl_ru]

    # Успеваемость. 1 семестр
    st.subheader("Успеваемость. 1 семестр")

    inputs["Curricular units 1st sem (enrolled)"] = st.number_input(
        "Количество записанных дисциплин в 1 семестре",
        value=6,
        min_value=3,
        max_value=10,
        step=1,
        help="Типичный диапазон университетской нагрузки. 5–6 дисциплин. максимум 8–10.",
    )

    inputs["Curricular units 1st sem (approved)"] = st.number_input(
        "Количество успешно сданных дисциплин в 1 семестре",
        value=min(6, inputs["Curricular units 1st sem (enrolled)"]),
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
        help="Средний итоговый балл. 10–14 соответствует среднему уровню успеваемости.",
    )

    # Успеваемость. 2 семестр
    st.subheader("Успеваемость. 2 семестр")

    inputs["Curricular units 2nd sem (enrolled)"] = st.number_input(
        "Количество записанных дисциплин во 2 семестре",
        value=6,
        min_value=3,
        max_value=10,
        step=1,
        help="Типичный диапазон. 5–6 дисциплин. при перегрузке 8–10.",
    )

    inputs["Curricular units 2nd sem (approved)"] = st.number_input(
        "Количество успешно сданных дисциплин во 2 семестре",
        value=min(6, inputs["Curricular units 2nd sem (enrolled)"]),
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


if predict_btn:
    # Дополнительные мягкие проверки консистентности
    if inputs["Curricular units 1st sem (approved)"] > inputs["Curricular units 1st sem (enrolled)"]:
        st.warning("1 семестр. сдано не может быть больше чем записано.")
    if inputs["Curricular units 2nd sem (approved)"] > inputs["Curricular units 2nd sem (enrolled)"]:
        st.warning("2 семестр. сдано не может быть больше чем записано.")

    # Собираем данные в правильном порядке признаков
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
            {
                "Исход": classes,
                "Вероятность": proba,
            }
        )
        st.dataframe(proba_df)

        st.success(f"Наиболее вероятный исход. **{pred_class}**")

        if pred_class == "Dropout":
            st.warning(
                "Модель оценивает высокий риск отчисления."
                " Рекомендуется проанализировать успеваемость и академическую нагрузку."
                " Возможны меры поддержки. консультации. пересмотр количества дисциплин."
            )
        elif pred_class == "Graduate":
            st.info(
                "Модель прогнозирует успешное завершение обучения."
                " Важно поддерживать текущий уровень мотивации и успеваемости."
            )
        else:
            st.info(
                "Модель прогнозирует продолжение обучения."
                " Рекомендуется следить за динамикой оценок и вовлеченности студента."
            )

    except Exception as e:
        st.error(f"Ошибка при предсказании. {e}")
else:
    st.info(
        "Заполните параметры студента в левой панели и нажмите кнопку "
        "\"Сделать прогноз\"."
    )
