import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Фикс для старого Pipeline из sklearn (как в ноутбуке) :contentReference[oaicite:0]{index=0}
import sklearn.compose._column_transformer as _ct

if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Заглушка для внутреннего класса старой версии sklearn.
        Нужна только для корректной загрузки сохранённой модели."""
        pass

    _ct._RemainderColsList = _RemainderColsList


# ---------- Настройки страницы ----------
st.set_page_config(page_title="EduRisk", layout="centered")

MODEL_PATH = Path(__file__).parent / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Порядок признаков – как при обучении модели
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

# Маппинги для удобного ввода
binary_map = {"Нет": 0, "Да": 1}
gender_map = {"Женский": 0, "Мужской": 1}
attendance_map = {
    "Дневная форма обучения": 1,
    "Вечерняя или другая форма": 0,
}

# ---------- Заголовок и описание ----------
st.title("Прогноз академического исхода обучения (EduRisk)")

st.markdown(
    """
Приложение использует предобученную модель машинного обучения,  
чтобы оценить, чем с наибольшей вероятностью завершится обучение конкретного студента:

* **Dropout** – отчисление  
* **Enrolled** – продолжает обучение  
* **Graduate** – успешное завершение программы  

Укажите значения тех же признаков, которые использовались при обучении модели.
"""
)

st.markdown("---")

# ---------- Входные данные (форма) ----------
st.subheader("Входные характеристики студента")

with st.form("edurisk_form"):
    # Блок: общие данные
    st.markdown("### Общая информация")

    attend_ru = st.selectbox(
        "Форма обучения",
        list(attendance_map.keys()),
    )

    gender_ru = st.selectbox(
        "Пол",
        list(gender_map.keys()),
    )

    age = st.number_input(
        "Возраст на момент поступления",
        min_value=16,
        max_value=70,
        value=20,
        step=1,
    )

    # Блок: социально-экономические факторы
    st.markdown("### Социально-экономические факторы")

    debtor_ru = st.selectbox(
        "Есть задолженность по оплате обучения",
        list(binary_map.keys()),
    )

    fees_ru = st.selectbox(
        "Оплата обучения внесена вовремя",
        list(binary_map.keys()),
    )

    scholar_ru = st.selectbox(
        "Получает стипендию",
        list(binary_map.keys()),
    )

    intl_ru = st.selectbox(
        "Иностранный студент",
        list(binary_map.keys()),
    )

    # Блок: успеваемость 1 семестр
    st.markdown("### Успеваемость. 1 семестр")

    cu1_enrolled = st.number_input(
        "Количество записанных дисциплин в 1 семестре",
        min_value=3,
        max_value=10,
        value=6,
        step=1,
    )

    cu1_approved = st.number_input(
        "Количество успешно сданных дисциплин в 1 семестре",
        min_value=0,
        max_value=cu1_enrolled,
        value=min(6, cu1_enrolled),
        step=1,
    )

    cu1_grade = st.number_input(
        "Средний балл за 1 семестр (0–20)",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1,
    )

    # Блок: успеваемость 2 семестр
    st.markdown("### Успеваемость. 2 семестр")

    cu2_enrolled = st.number_input(
        "Количество записанных дисциплин во 2 семестре",
        min_value=3,
        max_value=10,
        value=6,
        step=1,
    )

    cu2_approved = st.number_input(
        "Количество успешно сданных дисциплин во 2 семестре",
        min_value=0,
        max_value=cu2_enrolled,
        value=min(6, cu2_enrolled),
        step=1,
    )

    cu2_grade = st.number_input(
        "Средний балл за 2 семестр (0–20)",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1,
    )

    submitted = st.form_submit_button("Спрогнозировать исход обучения")

# ---------- Обработка и вывод результата ----------
if submitted:
    # базовая проверка консистентности
    if cu1_approved > cu1_enrolled or cu2_approved > cu2_enrolled:
        st.error("Число сданных дисциплин не может быть больше числа записанных.")
    else:
        # собираем признаки в нужном порядке
        row = {
            "Daytime/evening attendance": attendance_map[attend_ru],
            "Debtor": binary_map[debtor_ru],
            "Tuition fees up to date": binary_map[fees_ru],
            "Gender": gender_map[gender_ru],
            "Scholarship holder": binary_map[scholar_ru],
            "International": binary_map[intl_ru],
            "Age at enrollment": age,
            "Curricular units 1st sem (enrolled)": cu1_enrolled,
            "Curricular units 1st sem (approved)": cu1_approved,
            "Curricular units 1st sem (grade)": cu1_grade,
            "Curricular units 2nd sem (enrolled)": cu2_enrolled,
            "Curricular units 2nd sem (approved)": cu2_approved,
            "Curricular units 2nd sem (grade)": cu2_grade,
        }

        X = pd.DataFrame([row], columns=FEATURE_COLS)

        try:
            proba = model.predict_proba(X)[0]
            classes = model.classes_

            pred_idx = int(np.argmax(proba))
            pred_class = classes[pred_idx]

            st.markdown("### Результат прогноза")
            st.write(f"**Наиболее вероятный исход. {pred_class}**")

            proba_df = pd.DataFrame(
                {"Исход": classes, "Вероятность": proba}
            )
            st.table(proba_df)

            if pred_class == "Dropout":
                st.warning(
                    "Модель оценивает высокий риск отчисления. "
                    "Рекомендуется обратить внимание на успеваемость и "
                    "объём академической нагрузки."
                )
            elif pred_class == "Graduate":
                st.info(
                    "Модель прогнозирует успешное завершение программы обучения."
                )
            else:
                st.info(
                    "Модель прогнозирует продолжение обучения. "
                    "Важно контролировать динамику оценок и долгов."
                )

        except Exception as e:
            st.error(f"Ошибка при вычислении прогноза. {e}")
